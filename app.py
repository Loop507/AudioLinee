# 🎵 AudioLinee - by Loop507 (Versione completa e stabile con BPM)

import streamlit as st
import numpy as np
import cv2
import librosa
import os
import subprocess
import gc
import shutil
from typing import Tuple, Optional

MAX_DURATION = 300
MIN_DURATION = 1.0
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

# --- Utility

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
            st.error("❌ Il file audio è vuoto o non è stato caricato correttamente.")
            return None, None, None
        audio_duration = librosa.get_duration(y=y, sr=sr)
        if audio_duration < MIN_DURATION:
            st.error(f"❌ L'audio deve essere lungo almeno {MIN_DURATION} secondi. Durata attuale: {audio_duration:.2f}s")
            return None, None, None
        if audio_duration > MAX_DURATION:
            st.warning(f"⚠️ Audio troppo lungo ({audio_duration:.1f}s). Verrà troncato a {MAX_DURATION}s.")
            y = y[:int(MAX_DURATION * sr)]
            audio_duration = MAX_DURATION
        return y, sr, audio_duration
    except Exception as e:
        st.error(f"❌ Errore nel caricamento dell'audio: {str(e)}")
        return None, None, None

def generate_melspectrogram(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
        mel_spec_db = librosa.power_to_db(S, ref=np.max)
        min_val = mel_spec_db.min()
        max_val = mel_spec_db.max()
        if max_val == min_val:
            st.error("❌ L'audio non contiene variazioni sufficienti per generare il video.")
            return None
        mel_spec_norm = (mel_spec_db - min_val) / (max_val - min_val)
        if mel_spec_norm.shape[1] == 0:
            st.error("❌ Lo spettrogramma è vuoto.")
            return None
        return mel_spec_norm
    except Exception as e:
        st.error(f"❌ Errore nella generazione dello spettrogramma: {str(e)}")
        return None

def estimate_bpm(y, sr) -> float:
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)
    except:
        return 0.0

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2 ,4))
    return (rgb[2], rgb[1], rgb[0])

# --- Generatore video
class VideoGenerator:
    def __init__(self, format_res, level, fps, bg_color, line_color, bpm=0):
        self.WIDTH, self.HEIGHT = format_res
        self.FPS = fps
        self.LEVEL = level
        self.LINE_DENSITY = 30 if level == "soft" else 40 if level == "medium" else 50
        self.bg_color = bg_color
        self.line_color = line_color
        self.TEMP_VIDEO = "temp_output.mp4"
        self.FINAL_VIDEO = "final_output.mp4"
        self.bpm = bpm

    def draw_network(self, frame, num_nodes, time_idx, mel_spec):
        points = [(np.random.randint(0, self.WIDTH), np.random.randint(0, self.HEIGHT)) for _ in range(num_nodes)]
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                energy = mel_spec[np.random.randint(0, mel_spec.shape[0]), time_idx]
                if energy > 0.3:
                    cv2.line(frame, points[i], points[j], self.line_color, 1)

    def generate(self, mel_spec, duration, sync_audio):
        try:
            for f in [self.TEMP_VIDEO, self.FINAL_VIDEO]:
                if os.path.exists(f): os.remove(f)
            total_frames = int(duration * self.FPS)
            writer = cv2.VideoWriter(self.TEMP_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (self.WIDTH, self.HEIGHT))
            if not writer.isOpened():
                st.error("❌ Impossibile inizializzare il writer video.")
                return False
            progress = st.progress(0)
            info = st.empty()
            beat_interval = int(self.FPS * 60 / self.bpm) if self.bpm > 0 else None
            for i in range(total_frames):
                frame = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
                frame[:] = self.bg_color
                t_idx = int((i / total_frames) * mel_spec.shape[1])
                self.draw_network(frame, 15, t_idx, mel_spec)
                if beat_interval and i % beat_interval == 0:
                    cv2.putText(frame, "BPM!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, self.line_color, 2)
                writer.write(frame)
                if i % 10 == 0:
                    progress.progress(i / total_frames)
                    info.text(f"🎬 Frame {i}/{total_frames}")
            writer.release()
            progress.progress(1.0)
            info.text("✅ Video generato")
            if sync_audio and check_ffmpeg():
                subprocess.run(["ffmpeg", "-y", "-i", self.TEMP_VIDEO, "-i", "input_audio.wav", "-c:v", "libx264", "-c:a", "aac", "-shortest", self.FINAL_VIDEO])
            else:
                os.rename(self.TEMP_VIDEO, self.FINAL_VIDEO)
            return True
        except Exception as e:
            st.error(f"❌ Errore video: {e}")
            return False
        finally:
            gc.collect()

# --- Interfaccia Streamlit

def main():
    st.set_page_config(page_title="🎵 AudioLinee", layout="centered")
    st.title("🎵 AudioLinee - by Loop507")
    st.markdown("Crea video visivi sincronizzati all'audio")

    uploaded_file = st.file_uploader("🎧 Carica un file audio", type=["mp3", "wav"])
    if not uploaded_file:
        return

    if not validate_audio_file(uploaded_file):
        return

    with open("input_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    y, sr, duration = load_and_process_audio("input_audio.wav")
    if y is None:
        return

    bpm = estimate_bpm(y, sr)
    bpm_display = f"{bpm:.1f}" if isinstance(bpm, (int, float)) else "non rilevato"
    st.info(f"🔊 Durata: {duration:.2f}s — BPM stimati: {bpm_display}")

    mel = generate_melspectrogram(y, sr)
    if mel is None:
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        res = st.selectbox("📐 Formato", {"16:9": (1280,720), "9:16":(720,1280), "1:1":(720,720), "4:3":(800,600)})
    with col2:
        level = st.selectbox("✨ Effetti", ["soft", "medium", "hard"])
    with col3:
        fps = st.selectbox("🎞 FPS", [5, 10, 12, 15, 20, 24, 30], index=6)

    bg_color = hex_to_bgr(st.color_picker("🎨 Sfondo", "#FFFFFF"))
    line_color = hex_to_bgr(st.color_picker("🎨 Linee", "#000000"))

    sync_audio = st.checkbox("🔊 Sincronizza audio nel video")

    if st.button("🎬 Genera Video"):
        gen = VideoGenerator(res, level, fps, bg_color, line_color, bpm)
        if gen.generate(mel, duration, sync_audio):
            with open("final_output.mp4", "rb") as f:
                st.download_button("⬇️ Scarica il video", f, file_name="audiolinee_output.mp4", mime="video/mp4")

    if st.button("🧹 Pulisci File Temporanei"):
        for f in ["input_audio.wav", "temp_output.mp4", "final_output.mp4"]:
            if os.path.exists(f): os.remove(f)
        st.success("✅ Pulizia completata")

if __name__ == "__main__":
    main()
