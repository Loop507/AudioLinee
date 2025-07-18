# 🎵 AudioLine by Loop507 - Sincronizzazione BPM e Effetto Barcode

import streamlit as st
import numpy as np
import cv2
import librosa
import os
import subprocess
import gc
import shutil
from typing import Tuple, Optional

MAX_DURATION = 300  # 5 minuti massimo
MIN_DURATION = 1.0  # 1 secondo minimo
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

FORMAT_RESOLUTIONS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (720, 720),
    "4:3": (800, 600)
}

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

def estimate_bpm(y, sr) -> float:
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return tempo
    except:
        return 0.0

def generate_melspectrogram(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
        mel_spec_db = librosa.power_to_db(S, ref=np.max)
        min_val = mel_spec_db.min()
        max_val = mel_spec_db.max()
        if max_val == min_val:
            return None
        mel_spec_norm = (mel_spec_db - min_val) / (max_val - min_val)
        return mel_spec_norm
    except:
        return None

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0])

class VideoGenerator:
    def __init__(self, format_res, level, fps, bg_color, line_color, bpm, mode):
        self.WIDTH, self.HEIGHT = format_res
        self.FPS = fps
        self.LEVEL = level
        self.bg_color = bg_color
        self.line_color = line_color
        self.TEMP_VIDEO = "temp_output.mp4"
        self.FINAL_VIDEO = "final_output.mp4"
        self.LINE_DENSITY = 30 if level == "soft" else 40 if level == "medium" else 50
        self.BPM = bpm
        self.MODE = mode

    def generate_video(self, mel_spec_norm: np.ndarray, duration: float) -> bool:
        try:
            for f in [self.TEMP_VIDEO, self.FINAL_VIDEO]:
                if os.path.exists(f): os.remove(f)
            total_frames = int(duration * self.FPS)
            video_writer = cv2.VideoWriter(self.TEMP_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (self.WIDTH, self.HEIGHT))
            progress_bar = st.progress(0)
            for i in range(total_frames):
                frame = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
                frame[:] = self.bg_color
                time_index = int((i / total_frames) * mel_spec_norm.shape[1])
                if self.MODE == "Barcode":
                    spacing = 10
                    max_bars = self.WIDTH // spacing
                    for j in range(max_bars):
                        energy = mel_spec_norm[j % mel_spec_norm.shape[0], time_index]
                        thickness = int(np.interp(energy, [0, 1], [1, 5]))
                        x = j * spacing
                        if self.BPM > 0:
                            beat_interval = int(self.FPS / (self.BPM / 60))
                            if i % beat_interval != 0:
                                continue
                        cv2.line(frame, (x, 0), (x, self.HEIGHT), self.line_color, thickness)
                video_writer.write(frame)
                if i % 5 == 0:
                    progress_bar.progress((i+1)/total_frames)
            video_writer.release()
            os.rename(self.TEMP_VIDEO, self.FINAL_VIDEO)
            return True
        except Exception as e:
            st.error(f"Errore generazione video: {e}")
            return False

def main():
    st.set_page_config(page_title="🎵 AudioLine by Loop507", layout="centered")
    st.title("🎵 AudioLine by Loop507")

    uploaded = st.file_uploader("🎧 Carica audio (.wav/.mp3)", type=["wav", "mp3"])
    if uploaded:
        if not validate_audio_file(uploaded): return
        with open("input_audio.wav", "wb") as f:
            f.write(uploaded.read())
        y, sr, duration = load_and_process_audio("input_audio.wav")
        if y is None: return
        mel = generate_melspectrogram(y, sr)
        if mel is None: return
        bpm = estimate_bpm(y, sr)
        bpm_display = f"{bpm:.1f}" if bpm > 0 else "non rilevato"
        st.info(f"🎵 BPM stimati: {bpm_display}")

        col1, col2, col3 = st.columns(3)
        with col1:
            fmt_key = st.selectbox("📐 Formato", list(FORMAT_RESOLUTIONS.keys()))
            format_res = FORMAT_RESOLUTIONS[fmt_key]
        with col2:
            level = st.selectbox("🎨 Livello effetto", ["soft", "medium", "hard"])
        with col3:
            fps = st.selectbox("🎞 FPS", [5, 10, 12, 15, 20, 24, 30], index=6)

        bg = st.color_picker("🎨 Sfondo", "#FFFFFF")
        line = st.color_picker("🎨 Linee", "#000000")
        mode = st.selectbox("✨ Modalità", ["Barcode"])

        if st.button("🎬 Genera Video"):
            gen = VideoGenerator(format_res, level, fps, hex_to_bgr(bg), hex_to_bgr(line), bpm, mode)
            if gen.generate_video(mel, duration):
                with open("final_output.mp4", "rb") as f:
                    st.download_button("⬇️ Scarica Video", f, file_name="audioline_video.mp4")

if __name__ == "__main__":
    main()
