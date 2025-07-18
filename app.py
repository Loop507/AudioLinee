import streamlit as st
import numpy as np
import cv2
import librosa
import os
import shutil
import gc
from typing import Tuple, Optional

MAX_DURATION = 300  # 5 minuti max
MIN_DURATION = 1.0  # 1 secondo minimo
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB max

def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def validate_audio_file(uploaded_file) -> bool:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"❌ File troppo grande ({uploaded_file.size / 1024 / 1024:.1f}MB). Limite: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
        return False
    return True

def load_and_process_audio(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        if len(y) == 0:
            st.error("❌ File audio vuoto o non caricato correttamente.")
            return None, None, None
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < MIN_DURATION:
            st.error(f"❌ Audio troppo breve ({duration:.2f}s), serve almeno {MIN_DURATION}s.")
            return None, None, None
        if duration > MAX_DURATION:
            st.warning(f"⚠️ Audio troppo lungo ({duration:.1f}s), verrà troncato a {MAX_DURATION}s.")
            y = y[:int(MAX_DURATION * sr)]
            duration = MAX_DURATION
        return y, sr, duration
    except Exception as e:
        st.error(f"❌ Errore caricamento audio: {str(e)}")
        return None, None, None

def generate_melspectrogram(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
        if S.size == 0:
            st.error("❌ Impossibile generare spettrogramma.")
            return None
        mel_spec_db = librosa.power_to_db(S, ref=np.max)
        min_val, max_val = mel_spec_db.min(), mel_spec_db.max()
        if max_val == min_val:
            st.error("❌ Audio senza variazioni sufficienti per il video.")
            return None
        mel_spec_norm = (mel_spec_db - min_val) / (max_val - min_val)
        if mel_spec_norm.shape[1] == 0:
            st.error("❌ Spettrogramma vuoto.")
            return None
        return mel_spec_norm
    except Exception as e:
        st.error(f"❌ Errore spettrogramma: {str(e)}")
        return None

def estimate_bpm(y: np.ndarray, sr: int) -> float:
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return tempo
    except Exception:
        return 0.0

def hex_to_bgr(hex_color: str) -> Tuple[int,int,int]:
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i+lv//3], 16) for i in range(0, lv, lv//3))
    return (rgb[2], rgb[1], rgb[0])

class VideoGenerator:
    def __init__(self, resolution: Tuple[int,int], level: str, fps: int, bg_color: Tuple[int,int,int], line_color: Tuple[int,int,int], bpm: float):
        self.WIDTH, self.HEIGHT = resolution
        self.LEVEL = level
        self.FPS = fps
        self.bg_color = bg_color
        self.line_color = line_color
        self.bpm = bpm
        self.temp_video = "temp_output.mp4"
        self.final_video = "final_output.mp4"
        self.line_density = 30 if level == "soft" else 40 if level == "medium" else 50

    def draw_lines(self, frame, time_idx, mel_spec):
        center = self.HEIGHT // 2
        max_freq = mel_spec.shape[0]
        step = max(1, max_freq // self.line_density)
        for i in range(0, max_freq, step):
            energy = mel_spec[i, time_idx]
            thickness = int(np.interp(energy, [0,1],[1,10]))
            x = int(np.interp(i, [0,max_freq],[0,self.WIDTH]))
            cv2.line(frame, (x, center - 50), (x, center + 50), self.line_color, thickness)

    def draw_spectrum(self, frame, time_idx, mel_spec):
        center = self.HEIGHT // 2
        max_freq = mel_spec.shape[0]
        step = max(1, max_freq // self.line_density)
        if self.LEVEL == "soft":
            for i in range(0, max_freq, step):
                energy = mel_spec[i, time_idx]
                thickness = int(np.interp(energy, [0,1],[1,6]))
                x = int(np.interp(i, [0,max_freq],[0,self.WIDTH]))
                cv2.line(frame, (x,0), (x,self.HEIGHT), self.line_color, thickness)
        elif self.LEVEL == "medium":
            for i in range(0, max_freq, step):
                energy = mel_spec[i, time_idx]
                thickness = int(np.interp(energy, [0,1],[1,6]))
                y = int(np.interp(i, [0,max_freq],[0,self.HEIGHT]))
                cv2.line(frame, (0,y), (self.WIDTH,y), self.line_color, thickness)
        else:
            for i in range(0, max_freq, step):
                energy = mel_spec[i, time_idx]
                thickness = int(np.interp(energy, [0,1],[1,6]))
                x = int(np.interp(i, [0,max_freq],[0,self.WIDTH]))
                y = int(np.interp(i, [0,max_freq],[0,self.HEIGHT]))
                cv2.line(frame, (x,0), (x,self.HEIGHT), self.line_color, thickness)
                cv2.line(frame, (0,y), (self.WIDTH,y), self.line_color, thickness)

    def draw_barcode(self, frame, time_idx, mel_spec, frame_idx):
        bar_spacing = 10
        max_bars = self.WIDTH // bar_spacing
        energies = mel_spec[:max_bars, time_idx]
        threshold = 0.2
        if self.bpm > 0:
            beat_interval = max(1, int(self.FPS / (self.bpm / 60)))
        else:
            beat_interval = self.FPS
        if frame_idx % beat_interval != 0:
            return
        for i, e in enumerate(energies):
            if e < threshold:
                continue
            thickness = int(np.interp(i, [0, max_bars], [1, 10]))
            x = i * bar_spacing
            cv2.line(frame, (x, 0), (x, self.HEIGHT), self.line_color, thickness)

    def generate_video(self, mel_spec: np.ndarray, duration: float, effect_type: str, sync_audio: bool) -> bool:
        try:
            for f in [self.temp_video, self.final_video]:
                if os.path.exists(f):
                    os.remove(f)
            total_frames = int(duration * self.FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.temp_video, fourcc, self.FPS, (self.WIDTH, self.HEIGHT))
            if not writer.isOpened():
                st.error("❌ Impossibile aprire writer video.")
                return False
            progress = st.progress(0)
            status = st.empty()
            for idx in range(total_frames):
                frame = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8) * np.array(self.bg_color, dtype=np.uint8)
                time_idx = int((idx / total_frames) * mel_spec.shape[1])
                time_idx = max(0, min(time_idx, mel_spec.shape[1]-1))

                if effect_type == "Linee":
                    self.draw_lines(frame, time_idx, mel_spec)
                elif effect_type == "Spectrum":
                    self.draw_spectrum(frame, time_idx, mel_spec)
                elif effect_type == "Barcode":
                    self.draw_barcode(frame, time_idx, mel_spec, idx)

                writer.write(frame)
                if idx % 10 == 0 or idx == total_frames - 1:
                    progress.progress((idx+1) / total_frames)
                    status.text(f"🎬 Frame {idx+1} / {total_frames}")

            writer.release()
            progress.progress(1.0)
            status.text("✅ Video generato! Sincronizzazione audio...")

            if sync_audio:
                if not check_ffmpeg():
                    st.warning("⚠️ FFmpeg non trovato, video senza audio.")
                    os.rename(self.temp_video, self.final_video)
                    return True
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", self.temp_video,
                    "-i", "input_audio.wav",
                    "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
                    "-c:a", "aac", "-shortest",
                    self.final_video
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    st.error(f"❌ Errore FFmpeg: {result.stderr}")
                    os.rename(self.temp_video, self.final_video)
                else:
                    os.remove(self.temp_video)
            else:
                os.rename(self.temp_video, self.final_video)

            status.text("✅ Video pronto!")
            return True
        except Exception as e:
            st.error(f"❌ Errore nella generazione video: {str(e)}")
            return False
        finally:
            gc.collect()

def main():
    st.set_page_config(page_title="🎵 AudioLinee - by Loop507", layout="centered")
    st.title("🎨 AudioLinee")
    st.markdown("### by Loop507")
    st.markdown("Carica un file audio e genera un video sincronizzato ai BPM.")

    FORMAT_RESOLUTIONS = {
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "1:1": (720, 720),
        "4:3": (800, 600),
    }

    uploaded_file = st.file_uploader("🎧 Carica un audio (.wav o .mp3)", type=["wav","mp3"])
    if uploaded_file is None:
        return

    if not validate_audio_file(uploaded_file):
        return

    with open("input_audio.wav", "wb") as f:
        f.write(uploaded_file.read())
    st.success("🔊 Audio caricato correttamente!")

    y, sr, duration = load_and_process_audio("input_audio.wav")
    if y is None:
        return

    bpm = estimate_bpm(y, sr)
    bpm_display = f"{bpm:.1f}" if bpm > 0 else "non rilevato"
    st.info(f"🎵 BPM stimati: {bpm_display}")
    st.info(f"⏳ Durata audio: {duration:.2f} secondi")

    mel_spec = generate_melspectrogram(y, sr)
    if mel_spec is None:
        return
    st.success("✅ Analisi audio completata!")

    col1, col2, col3 = st.columns(3)
    with col1:
        video_format = st.selectbox("📐 Formato video", list(FORMAT_RESOLUTIONS.keys()))
    with col2:
        effect_level = st.selectbox("🎨 Livello effetti", ["soft", "medium", "hard"])
    with col3:
        fps_choice = st.selectbox("🎞️ FPS", [5, 10, 15, 24, 30, 60], index=3)

    effect_type = st.selectbox("🎭 Tipo effetto", ["Linee", "Spectrum", "Barcode"])

    bg_color_hex = st.color_picker("🎨 Colore sfondo", "#FFFFFF")
    line_color_hex = st.color_picker("🎨 Colore linee", "#000000")

    bitrate_map = {
        (1280, 720): {60: 6, 30:5, 24:4, 15:2.5, 10:1.5, 5:1},
        (720, 1280): {60: 6, 30:5, 24:4, 15:2.5, 10:1.5, 5:1},
        (720, 720): {60:5, 30:4, 24:3.5, 15:2, 10:1, 5:0.5},
        (800, 600): {60:4, 30:3.5, 24:3, 15:1.8, 10:1, 5:0.5}
    }
    bitrate = bitrate_map.get(FORMAT_RESOLUTIONS[video_format], {}).get(fps_choice, 3)
    est_size_mb = (bitrate * duration) / 8 if duration else 0
    st.info(f"📦 Stima dimensione video: {est_size_mb:.1f} MB")

    sync_audio = st.checkbox("🔊 Sincronizza audio nel video", value=True)
    if sync_audio and not check_ffmpeg():
        st.warning("⚠️ FFmpeg non disponibile, sincronizzazione audio disabilitata.")
        sync_audio = False

    if st.button("🎬 Genera Video"):
        bg_bgr = hex_to_bgr(bg_color_hex)
        line_bgr = hex_to_bgr(line_color_hex)
        gen = VideoGenerator(FORMAT_RESOLUTIONS[video_format], effect_level, fps_choice, bg_bgr, line_bgr, bpm)
        success = gen.generate_video(mel_spec, duration, effect_type, sync_audio)
        if success and os.path.exists(gen.final_video):
            with open(gen.final_video, "rb") as f:
                st.download_button("⬇️ Scarica video", f, file_name=f"audiolinee_{video_format}_{effect_level}.mp4", mime="video/mp4")
            size_mb = os.path.getsize(gen.final_video) / 1024 / 1024
            st.info(f"📁 Dimensione file: {size_mb:.1f} MB")
        else:
            st.error("❌ Errore durante la generazione del video.")

    if st.button("🧹 Pulisci file temporanei"):
        for f in ["input_audio.wav", "temp_output.mp4", "final_output.mp4"]:
            if os.path.exists(f):
                os.remove(f)
        st.success("✅ File temporanei eliminati!")

if __name__ == "__main__":
    main()
