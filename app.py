# 🎵 AudioLinee.py (by Loop507) - Versione con stile Barcode Grafico
# Generatore di video visivi sincronizzati con l'audio
# Aggiunta opzione "barcode style" mantenendo la struttura originale

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
MAX_FILE_SIZE = 50 * 1024 * 1024

def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def validate_audio_file(uploaded_file) -> bool:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("❌ File troppo grande")
        return False
    return True

def load_and_process_audio(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
    try:
        y, sr = librosa.load(file_path, sr=None)
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
        audio_duration = librosa.get_duration(y=y, sr=sr)
        if audio_duration < MIN_DURATION:
            st.error("❌ Audio troppo breve")
            return None, None, None
        if audio_duration > MAX_DURATION:
            y = y[:int(MAX_DURATION * sr)]
            audio_duration = MAX_DURATION
        return y, sr, audio_duration
    except Exception as e:
        st.error(f"❌ Errore audio: {str(e)}")
        return None, None, None

def generate_melspectrogram(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
        mel_db = librosa.power_to_db(S, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        return mel_norm
    except Exception as e:
        st.error(f"❌ Errore spettrogramma: {str(e)}")
        return None

class VideoGenerator:
    def __init__(self, format_res: Tuple[int, int], level: str, visual_type: str, fps: int = 30):
        self.FRAME_WIDTH, self.FRAME_HEIGHT = format_res
        self.FPS = fps
        self.LEVEL = level
        self.VISUAL_TYPE = visual_type
        self.TEMP_VIDEO = "temp_output.mp4"
        self.FINAL_VIDEO = "final_output.mp4"

    def draw_connected_network(self, frame, time_index, mel_spec_norm):
        points = []
        for _ in range(15):
            x = np.random.randint(0, self.FRAME_WIDTH)
            y = np.random.randint(0, self.FRAME_HEIGHT)
            points.append((x, y))
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                energy = mel_spec_norm[np.random.randint(0, mel_spec_norm.shape[0]), time_index]
                if energy > 0.3:
                    cv2.line(frame, points[i], points[j], (0, 0, 0), 1)

    def draw_barcode_style(self, frame, time_index, mel_spec_norm):
        num_bars = 100
        bar_width = self.FRAME_WIDTH // num_bars
        for i in range(num_bars):
            freq_bin = int(i / num_bars * mel_spec_norm.shape[0])
            energy = mel_spec_norm[freq_bin, time_index]
            line_height = int(energy * self.FRAME_HEIGHT)
            x = i * bar_width
            y_start = self.FRAME_HEIGHT - line_height
            color = (0, 0, 0)  # Nero stile china
            thickness = 1 + int(energy * 4)  # Spessore da 1 a 5
            cv2.line(frame, (x, y_start), (x, self.FRAME_HEIGHT), color, thickness)

    def generate_video(self, mel_spec_norm: np.ndarray, audio_duration: float, sync_audio: bool = False) -> bool:
        try:
            for f in [self.TEMP_VIDEO, self.FINAL_VIDEO]:
                if os.path.exists(f): os.remove(f)
            total_frames = int(audio_duration * self.FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.TEMP_VIDEO, fourcc, self.FPS, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            progress = st.progress(0)
            status = st.empty()
            for i in range(total_frames):
                frame = np.ones((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3), dtype=np.uint8) * 255
                time_index = min(int((i / total_frames) * mel_spec_norm.shape[1]), mel_spec_norm.shape[1] - 1)
                if self.VISUAL_TYPE == "Linee audio":
                    self.draw_connected_network(frame, time_index, mel_spec_norm)
                else:
                    self.draw_barcode_style(frame, time_index, mel_spec_norm)
                writer.write(frame)
                if i % 10 == 0:
                    progress.progress((i + 1) / total_frames)
                    status.text(f"Frame {i+1}/{total_frames}")
            writer.release()
            if sync_audio and check_ffmpeg():
                subprocess.run(["ffmpeg", "-y", "-i", self.TEMP_VIDEO, "-i", "input_audio.wav", "-c:v", "copy", "-c:a", "aac", "-shortest", self.FINAL_VIDEO])
                os.remove(self.TEMP_VIDEO)
            else:
                os.rename(self.TEMP_VIDEO, self.FINAL_VIDEO)
            status.text("✅ Video pronto!")
            return True
        except Exception as e:
            st.error(f"❌ Errore video: {str(e)}")
            return False


def main():
    FORMAT_RES = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (720, 720)}
    st.set_page_config(page_title="🎵 AudioLinee", layout="centered")
    st.title("🎨 AudioLinee")

    uploaded = st.file_uploader("🎧 Carica un file audio", type=["wav", "mp3"])
    if uploaded and validate_audio_file(uploaded):
        with open("input_audio.wav", "wb") as f:
            f.write(uploaded.read())
        y, sr, duration = load_and_process_audio("input_audio.wav")
        if y is None: return
        mel = generate_melspectrogram(y, sr)
        if mel is None: return
        st.success(f"Audio caricato! Durata: {duration:.2f}s")
        col1, col2 = st.columns(2)
        with col1:
            fmt = st.selectbox("📐 Formato", list(FORMAT_RES.keys()))
            fps = st.slider("🎞️ FPS", 12, 72, 30)
        with col2:
            level = st.selectbox("🎨 Livello", ["soft", "medium", "hard"])
            visual_type = st.selectbox("🎛️ Stile visivo", ["Linee audio", "Barcode style"])
        sync_audio = st.checkbox("🔊 Sincronizza audio")
        if st.button("🎬 Genera Video"):
            gen = VideoGenerator(FORMAT_RES[fmt], level, visual_type, fps)
            if gen.generate_video(mel, duration, sync_audio):
                with open("final_output.mp4", "rb") as f:
                    st.download_button("⬇️ Scarica il video", f, file_name="audiolinee_output.mp4", mime="video/mp4")

if __name__ == "__main__":
    main()
