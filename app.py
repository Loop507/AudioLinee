import streamlit as st
import numpy as np
import cv2
import librosa
import os
import subprocess
import gc
import shutil

MAX_DURATION = 300  # max 5 minuti
MIN_DURATION = 1.0
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max

def check_ffmpeg():
    return shutil.which("ffmpeg") is not None

def validate_audio_file(uploaded_file):
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"❌ File troppo grande ({uploaded_file.size / 1024 / 1024:.1f}MB). Limite: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
        return False
    return True

def load_and_process_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
        if len(y) == 0:
            st.error("❌ File audio vuoto o non valido.")
            return None, None, None
        audio_duration = librosa.get_duration(y=y, sr=sr)
        if audio_duration < MIN_DURATION:
            st.error(f"❌ Audio troppo corto ({audio_duration:.2f}s). Minimo {MIN_DURATION}s.")
            return None, None, None
        if audio_duration > MAX_DURATION:
            st.warning(f"⚠️ Audio troppo lungo ({audio_duration:.1f}s). Verrà troncato a {MAX_DURATION}s.")
            y = y[:int(MAX_DURATION * sr)]
            audio_duration = MAX_DURATION
        return y, sr, audio_duration
    except Exception as e:
        st.error(f"❌ Errore nel caricamento audio: {str(e)}")
        return None, None, None

def generate_melspectrogram(y, sr):
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
        if S.size == 0:
            st.error("❌ Spettrogramma vuoto.")
            return None
        mel_spec_db = librosa.power_to_db(S, ref=np.max)
        min_val = mel_spec_db.min()
        max_val = mel_spec_db.max()
        if max_val == min_val:
            st.error("❌ Audio senza variazioni.")
            return None
        mel_spec_norm = (mel_spec_db - min_val) / (max_val - min_val)
        if mel_spec_norm.shape[1] == 0:
            st.error("❌ Spettrogramma vuoto.")
            return None
        return mel_spec_norm
    except Exception as e:
        st.error(f"❌ Errore spettrogramma: {str(e)}")
        return None

def estimate_bpm(y, sr):
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if tempo is None or tempo <= 0:
            return 120.0
        return tempo
    except:
        return 120.0

class VideoGenerator:
    def __init__(self, format_res, effect_type, level, fps=30, bpm=120.0, line_color=(0,0,0), bg_color=(255,255,255)):
        self.W, self.H = format_res
        self.FPS = fps
        self.bpm = bpm
        self.effect_type = effect_type  # 'lines', 'spectrum', 'barcode'
        self.level = level  # 'soft', 'medium', 'hard'
        self.line_color = line_color
        self.bg_color = bg_color
        self.temp_video = "temp_output.mp4"
        self.final_video = "final_output.mp4"

    def draw_lines(self, frame, mel_spec_norm, time_index):
        points = []
        density = 30 if self.level == "soft" else 40 if self.level == "medium" else 50
        for _ in range(density):
            x = np.random.randint(0, self.W)
            y = np.random.randint(0, self.H)
            points.append((x,y))
        volume = np.mean(mel_spec_norm[:, time_index])
        if volume < 0.05:
            return
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                energy = mel_spec_norm[np.random.randint(0, mel_spec_norm.shape[0]), time_index]
                if energy > 0.3:
                    thickness = max(1, int(energy*4))
                    cv2.line(frame, points[i], points[j], self.line_color, thickness)

    def draw_spectrum(self, frame, mel_spec_norm, time_index):
        freq_bins = mel_spec_norm.shape[0]
        max_height = self.H
        max_width = self.W
        step = max_width // freq_bins

        if self.level == "soft":
            # Solo linee verticali sottili
            thickness = 1
            for i in range(freq_bins):
                energy = mel_spec_norm[i, time_index]
                height = int(energy * max_height)
                if height > 1:
                    x = i * step
                    cv2.line(frame, (x, max_height), (x, max_height - height), self.line_color, thickness)
        elif self.level == "medium":
            # Solo linee orizzontali medie
            thickness = 2
            for i in range(freq_bins):
                energy = mel_spec_norm[i, time_index]
                width = int(energy * max_width)
                if width > 1:
                    y = int(i * (max_height/freq_bins))
                    cv2.line(frame, (0, y), (width, y), self.line_color, thickness)
        else:
            # hard = entrambe
            thickness_v = 1
            thickness_h = 2
            for i in range(freq_bins):
                energy = mel_spec_norm[i, time_index]
                height = int(energy * max_height)
                width = int(energy * max_width)
                if height > 1:
                    x = i * step
                    cv2.line(frame, (x, max_height), (x, max_height - height), self.line_color, thickness_v)
                if width > 1:
                    y = int(i * (max_height/freq_bins))
                    cv2.line(frame, (0, y), (width, y), self.line_color, thickness_h)

    def draw_barcode(self, frame, mel_spec_norm, time_index):
        freq_bins = mel_spec_norm.shape[0]
        max_height = self.H
        max_width = self.W
        step = max_width // freq_bins

        bpm_period_frames = max(1, int((60 / self.bpm) * self.FPS))
        beat_pos = time_index % bpm_period_frames
        beat_ratio = beat_pos / bpm_period_frames

        for i in range(freq_bins):
            energy = mel_spec_norm[i, time_index]
            # Linee larghe per basse freq, sottili per alte freq
            if i < freq_bins // 3:
                thickness = int(5 + 5 * abs(np.sin(beat_ratio * np.pi * 2))) if self.level != 'soft' else 4
            elif i < 2 * freq_bins // 3:
                thickness = int(3 + 3 * abs(np.sin(beat_ratio * np.pi * 2))) if self.level != 'soft' else 2
            else:
                thickness = 1
            height = int(energy * max_height)
            if height > 1:
                x = i * step
                y0 = max_height
                y1 = max_height - height
                cv2.line(frame, (x, y0), (x, y1), self.line_color, thickness)

    def generate_video(self, mel_spec_norm, audio_duration, sync_audio=False):
        try:
            for f in [self.temp_video, self.final_video]:
                if os.path.exists(f):
                    os.remove(f)

            total_frames = int(audio_duration * self.FPS)
            speed_factor = self.bpm / 120.0
            adjusted_total_frames = int(total_frames * speed_factor)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(self.temp_video, fourcc, self.FPS, (self.W, self.H))
            if not video_writer.isOpened():
                st.error("❌ Impossibile inizializzare il writer video.")
                return False

            progress_bar = st.progress(0)
            status_text = st.empty()

            for frame_idx in range(adjusted_total_frames):
                frame = np.ones((self.H, self.W, 3), dtype=np.uint8) * np.array(self.bg_color, dtype=np.uint8)
                time_index = int((frame_idx / adjusted_total_frames) * mel_spec_norm.shape[1])
                time_index = max(0, min(time_index, mel_spec_norm.shape[1]-1))

                if self.effect_type == "lines":
                    self.draw_lines(frame, mel_spec_norm, time_index)
                elif self.effect_type == "spectrum":
                    self.draw_spectrum(frame, mel_spec_norm, time_index)
                elif self.effect_type == "barcode":
                    self.draw_barcode(frame, mel_spec_norm, time_index)

                video_writer.write(frame)

                if frame_idx % 10 == 0:
                    progress = (frame_idx+1)/adjusted_total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"🎬 Generazione frame {frame_idx+1}/{adjusted_total_frames} ({progress*100:.1f}%)")

            video_writer.release()
            progress_bar.progress(1.0)
            status_text.text("✅ Video generato! Sincronizzazione audio...")

            if sync_audio:
                if not check_ffmpeg():
                    st.warning("⚠️ FFmpeg non trovato. Video senza audio.")
                    os.rename(self.temp_video, self.final_video)
                    return True
                try:
                    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", self.temp_video, "-i", "input_audio.wav", "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", "-shortest", self.final_video]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if result.returncode != 0:
                        st.error(f"❌ Errore FFmpeg: {result.stderr}")
                        os.rename(self.temp_video, self.final_video)
                    else:
                        os.remove(self.temp_video)
                except subprocess.TimeoutExpired:
                    st.error("❌ Timeout FFmpeg.")
                    os.rename(self.temp_video, self.final_video)
                except Exception as e:
                    st.error(f"❌ Errore sincronizzazione audio: {str(e)}")
                    os.rename(self.temp_video, self.final_video)
            else:
                os.rename(self.temp_video, self.final_video)

            status_text.text("✅ Video completato con successo!")
            return True

        except Exception as e:
            st.error(f"❌ Errore generazione video: {str(e)}")
            return False
        finally:
            if 'video_writer' in locals():
                video_writer.release()
            gc.collect()

def main():
    FORMAT_RESOLUTIONS = {
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "1:1": (720, 720),
        "4:3": (800, 600)
    }

    st.set_page_config(page_title="🎵 AudioLinee - by Loop507", layout="centered")
    st.title("🎨 AudioLinee")
    st.markdown("### by Loop507")
    st.markdown("Carica un file audio e genera un video visivo sincronizzato.")

    uploaded_file = st.file_uploader("🎧 Carica un file audio (.wav o .mp3)", type=["wav", "mp3"])
    if uploaded_file is None:
        return

    if not validate_audio_file(uploaded_file):
        return

    with open("input_audio.wav", "wb") as f:
        f.write(uploaded_file.read())
    st.success("🔊 Audio caricato correttamente!")

    y, sr, audio_duration = load_and_process_audio("input_audio.wav")
    if y is None:
        return

    bpm = estimate_bpm(y, sr)
    if bpm is None or bpm <= 0:
        bpm = 120.0

    fps_options = [15, 24, 30, 48, 60, 72]
    fps = st.selectbox("📊 Seleziona FPS", fps_options, index=2)

    st.info(f"🎵 BPM stimati: {bpm:.1f}, FPS impostati: {fps}")

    with st.spinner("📊 Analisi audio in corso..."):
        mel_spec_norm = generate_melspectrogram(y, sr)
    if mel_spec_norm is None:
        return
    st.success("✅ Analisi audio completata!")

    col1, col2 = st.columns(2)
    with col1:
        video_format = st.selectbox("📐 Formato video", list(FORMAT_RESOLUTIONS.keys()))
    with col2:
        effect_type = st.selectbox("🎨 Tipo effetto", ["lines", "spectrum", "barcode"])

    level_options = ["soft", "medium", "hard"]
    level = st.selectbox("🎚️ Livello effetti", level_options)

    col3, col4 = st.columns(2)
    with col3:
        line_color = st.color_picker("🎨 Colore linee", "#000000")
    with col4:
        bg_color = st.color_picker("🎨 Colore sfondo", "#FFFFFF")

    sync_audio = st.checkbox("🔊 Sincronizza l'audio nel video")
    if not check_ffmpeg():
        st.warning("⚠️ FFmpeg non disponibile - La sincronizzazione audio è disabilitata")
        sync_audio = False

    if st.button("🎬 Genera Video"):
        def hex_to_bgr(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
        bgr_line = hex_to_bgr(line_color)
        bgr_bg = hex_to_bgr(bg_color)

        generator = VideoGenerator(FORMAT_RESOLUTIONS[video_format], effect_type, level, fps, bpm, bgr_line, bgr_bg)
        success = generator.generate_video(mel_spec_norm, audio_duration, sync_audio)
        if success and os.path.exists("final_output.mp4"):
            with open("final_output.mp4", "rb") as f:
                st.download_button("⬇️ Scarica il video", f, file_name=f"audiolinee_{video_format}_{effect_type}_{level}.mp4", mime="video/mp4")
            file_size = os.path.getsize("final_output.mp4")
            st.info(f"📁 Dimensione file: {file_size / 1024 / 1024:.1f} MB")
        else:
            st.error("❌ Errore nella generazione del video.")

    if st.button("🧹 Pulisci file temporanei"):
        temp_files = ["input_audio.wav", "temp_output.mp4", "final_output.mp4"]
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        st.success("🧹 File temporanei rimossi.")

if __name__ == "__main__":
    main()
