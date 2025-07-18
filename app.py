import streamlit as st
import numpy as np
import cv2
import librosa
import subprocess
import os
import gc

MAX_DURATION = 300  # Max 5 minuti
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def check_ffmpeg():
    import shutil
    return shutil.which("ffmpeg") is not None

def validate_audio_file(uploaded_file):
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"❌ File troppo grande ({uploaded_file.size / 1024 / 1024:.1f}MB). Limite: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
        return False
    return True

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    if duration > MAX_DURATION:
        y = y[:int(MAX_DURATION * sr)]
        duration = MAX_DURATION
    return y, sr, duration

def generate_melspectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(S, ref=np.max)
    mel_db_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    return mel_db_norm

def estimate_bpm(y, sr):
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return tempo
    except:
        return 120.0

class VideoGenerator:
    def __init__(self, resolution, level, style, fps, bg_color, line_color, bpm):
        self.width, self.height = resolution
        self.level = level
        self.style = style
        self.fps = fps
        self.bg_color = self.hex_to_bgr(bg_color)
        self.line_color = self.hex_to_bgr(line_color)
        self.bpm = bpm
        self.temp_video = "temp_output.mp4"
        self.final_video = "final_output.mp4"

    def hex_to_bgr(self, hex_color):
        hex_color = hex_color.lstrip('#')
        lv = len(hex_color)
        rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        return (rgb[2], rgb[1], rgb[0])  # OpenCV usa BGR

    def generate_barcode(self, frame, mel_spec, t_idx):
        num_bars = 40
        bar_width = self.width // num_bars
        bpm_frames = int((60 / self.bpm) * self.fps)
        phase = t_idx % bpm_frames
        for i in range(num_bars):
            freq_idx = int(i / num_bars * mel_spec.shape[0])
            energy = mel_spec[freq_idx, t_idx]
            if energy < 0.1:
                continue
            thickness = int(1 + energy * 10)
            if (phase // (bpm_frames // 4)) % 2 == 0:
                x = i * bar_width
                cv2.rectangle(frame, (x, 0), (x + thickness, self.height), self.line_color, -1)

    def generate_spectrum(self, frame, mel_spec, t_idx):
        num_lines = 40
        threshold = 0.05
        if self.level == "soft":  # linee verticali solo
            for i in range(num_lines):
                freq_idx = int(i / num_lines * mel_spec.shape[0])
                energy = mel_spec[freq_idx, t_idx]
                if energy < threshold:
                    continue
                x = int(i / num_lines * self.width)
                thickness = max(1, int(energy * 5))
                cv2.line(frame, (x, 0), (x, self.height), self.line_color, thickness)
        elif self.level == "medium":  # linee orizzontali solo
            for i in range(num_lines):
                freq_idx = int(i / num_lines * mel_spec.shape[0])
                energy = mel_spec[freq_idx, t_idx]
                if energy < threshold:
                    continue
                y = int(i / num_lines * self.height)
                thickness = max(1, int(energy * 5))
                cv2.line(frame, (0, y), (self.width, y), self.line_color, thickness)
        else:  # hard entrambe
            for i in range(num_lines):
                freq_idx = int(i / num_lines * mel_spec.shape[0])
                energy = mel_spec[freq_idx, t_idx]
                if energy < threshold:
                    continue
                x = int(i / num_lines * self.width)
                y = int(i / num_lines * self.height)
                thickness = max(1, int(energy * 5))
                cv2.line(frame, (x, 0), (x, self.height), self.line_color, thickness)
                cv2.line(frame, (0, y), (self.width, y), self.line_color, thickness)

    def generate_video(self, mel_spec, duration):
        if os.path.exists(self.temp_video):
            os.remove(self.temp_video)
        if os.path.exists(self.final_video):
            os.remove(self.final_video)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        total_frames = int(duration * self.fps)
        video_writer = cv2.VideoWriter(self.temp_video, fourcc, self.fps, (self.width, self.height))

        progress_bar = st.progress(0)
        status_text = st.empty()

        for frame_idx in range(total_frames):
            frame = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
            t_idx = min(int(frame_idx / total_frames * mel_spec.shape[1]), mel_spec.shape[1] - 1)

            if self.style == "Barcode":
                self.generate_barcode(frame, mel_spec, t_idx)
            else:
                self.generate_spectrum(frame, mel_spec, t_idx)

            video_writer.write(frame)

            if frame_idx % 10 == 0:
                progress_bar.progress((frame_idx + 1) / total_frames)
                status_text.text(f"Generazione frame {frame_idx + 1} / {total_frames}")

        video_writer.release()
        progress_bar.progress(1.0)
        status_text.text("Video generato!")

        if check_ffmpeg() and os.path.exists("input_audio.wav"):
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", self.temp_video,
                "-i", "input_audio.wav",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                self.final_video,
            ]
            subprocess.run(cmd)

        return True

def main():
    st.title("🎵 AudioLinee by Loop507")

    resolution = st.selectbox("Formato video", options=["16:9", "9:16", "1:1", "4:3"], index=0)
    res_dict = {"16:9": (1280,720), "9:16": (720,1280), "1:1": (720,720), "4:3": (800,600)}
    size = res_dict[resolution]

    level = st.selectbox("Livello (soft - medium - hard)", options=["soft", "medium", "hard"], index=0)
    style = st.selectbox("Stile", options=["Spectrum", "Barcode"], index=0)
    fps = st.selectbox("FPS", options=[24, 30, 48, 60, 72], index=1)

    bg_color = st.color_picker("Colore sfondo", "#FFFFFF")
    line_color = st.color_picker("Colore linee", "#000000")

    uploaded_file = st.file_uploader("Carica file audio (.wav o .mp3)", type=["wav","mp3"])

    if uploaded_file is not None:
        if not validate_audio_file(uploaded_file):
            return

        with open("input_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        y, sr, duration = load_audio("input_audio.wav")
        if y is None:
            return

        mel_spec = generate_melspectrogram(y, sr)
        if mel_spec is None:
            return

        bpm = estimate_bpm(y, sr)
        st.info(f"🎵 BPM stimati: {bpm:.1f}, FPS impostati: {fps}")

        video_gen = VideoGenerator(
            resolution=size,
            level=level,
            style=style,
            fps=fps,
            bg_color=bg_color,
            line_color=line_color,
            bpm=bpm
        )

        if video_gen.generate_video(mel_spec, duration):
            st.success("🎬 Video creato con successo!")
            video_file = open(video_gen.final_video, "rb").read()
            st.video(video_file)
        else:
            st.error("❌ Errore nella creazione del video")

if __name__ == "__main__":
    main()
