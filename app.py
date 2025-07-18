# 🎵 AudioLinee.py - Versione stabile con sincronizzazione BPM e risoluzione corretta

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

FPS_OPTIONS = [5, 10, 15, 20, 24, 30, 40, 50, 60, 72]

# === Utility ===
def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def validate_audio_file(uploaded_file) -> bool:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("❌ File troppo grande. Limite: 200MB")
        return False
    return True

def load_and_process_audio(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        if len(y) == 0:
            st.error("❌ Il file audio è vuoto.")
            return None, None, None
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < MIN_DURATION:
            st.error("❌ L'audio è troppo corto.")
            return None, None, None
        if duration > MAX_DURATION:
            y = y[:int(MAX_DURATION * sr)]
            duration = MAX_DURATION
        return y, sr, duration
    except Exception as e:
        st.error(f"❌ Errore audio: {str(e)}")
        return None, None, None

def generate_melspectrogram(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr / 2)
        S_db = librosa.power_to_db(S, ref=np.max)
        norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        return norm if norm.shape[1] > 0 else None
    except:
        return None

def estimate_bpm(y, sr) -> float:
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return tempo
    except:
        return 0.0

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

class VideoGenerator:
    def __init__(self, format_res: Tuple[int, int], level: str, fps: int, bg_color: Tuple[int, int, int], line_color: Tuple[int, int, int], bpm: float = 0):
        self.WIDTH, self.HEIGHT = format_res
        self.LEVEL = level
        self.FPS = fps
        self.bg_color = bg_color
        self.line_color = line_color
        self.TEMP = "temp_output.mp4"
        self.FINAL = "final_output.mp4"
        self.LINE_DENSITY = 30 if level == "soft" else 40 if level == "medium" else 50
        self.beat_interval = int(fps / (bpm / 60)) if bpm > 0 else None

    def draw_frame(self, frame_idx: int, mel_spec_norm: np.ndarray) -> np.ndarray:
        frame = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        frame[:] = self.bg_color
        time_idx = int((frame_idx / (self.FPS * mel_spec_norm.shape[1])) * mel_spec_norm.shape[1])
        for i in range(0, self.WIDTH, self.LINE_DENSITY):
            energy = mel_spec_norm[np.random.randint(0, mel_spec_norm.shape[0]), time_idx]
            if self.beat_interval and frame_idx % self.beat_interval != 0:
                continue
            thickness = int(np.interp(energy, [0, 1], [1, 6]))
            cv2.line(frame, (i, 0), (i, self.HEIGHT), self.line_color, thickness)
        return frame

    def generate_video(self, mel_spec_norm: np.ndarray, duration: float, sync_audio: bool = False) -> bool:
        total_frames = int(duration * self.FPS)
        out = cv2.VideoWriter(self.TEMP, cv2.VideoWriter_fourcc(*'mp4v'), self.FPS, (self.WIDTH, self.HEIGHT))
        for idx in range(total_frames):
            frame = self.draw_frame(idx, mel_spec_norm)
            out.write(frame)
        out.release()
        if sync_audio and check_ffmpeg():
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", self.TEMP,
                    "-i", "input_audio.wav",
                    "-c:v", "libx264", "-crf", "23",
                    "-c:a", "aac", "-shortest",
                    self.FINAL
                ], check=True)
                os.remove(self.TEMP)
            except:
                os.rename(self.TEMP, self.FINAL)
        else:
            os.rename(self.TEMP, self.FINAL)
        return os.path.exists(self.FINAL)

def main():
    st.set_page_config(page_title="🎵 AudioLinee - Sync BPM", layout="centered")
    st.title("🎨 AudioLinee - Sincronizzato con BPM")

    uploaded_file = st.file_uploader("🎧 Carica un file audio", type=["wav", "mp3"])
    if uploaded_file:
        if not validate_audio_file(uploaded_file): return
        with open("input_audio.wav", "wb") as f:
            f.write(uploaded_file.read())
        y, sr, duration = load_and_process_audio("input_audio.wav")
        if y is None: return
        mel_spec = generate_melspectrogram(y, sr)
        if mel_spec is None:
            st.error("❌ Errore nella generazione dello spettrogramma.")
            return
        bpm = estimate_bpm(y, sr)
        bpm_display = f"{bpm:.1f}" if isinstance(bpm, (float, int)) else "non rilevato"
        st.info(f"🎵 BPM stimati: {bpm_display}")

        col1, col2, col3 = st.columns(3)
        with col1:
            video_format = st.selectbox("📐 Formato video", list(FORMAT_RESOLUTIONS.keys()))
        with col2:
            level = st.selectbox("✨ Livello effetti", ["soft", "medium", "hard"])
        with col3:
            fps = st.selectbox("🎞️ FPS", FPS_OPTIONS, index=FPS_OPTIONS.index(30))

        bg_color = hex_to_bgr(st.color_picker("🎨 Sfondo", "#FFFFFF"))
        line_color = hex_to_bgr(st.color_picker("🎨 Linee", "#000000"))
        sync_audio = st.checkbox("🔊 Sincronizza audio nel video")

        if st.button("🎬 Genera Video"):
            res = FORMAT_RESOLUTIONS[video_format]
            gen = VideoGenerator(res, level, fps, bg_color, line_color, bpm)
            if gen.generate_video(mel_spec, duration, sync_audio):
                with open("final_output.mp4", "rb") as f:
                    st.download_button("⬇️ Scarica il video", f, file_name="audiolinee_sync.mp4")
            else:
                st.error("❌ Errore durante la generazione del video.")

if __name__ == "__main__":
    main()
