# 🎵 AudioLine by Loop507 - Versione con effetti Linee, Barcode, Rete sincronizzati con BPM e frequenze

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

def estimate_bpm(y: np.ndarray, sr: int) -> float:
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)
    except:
        return 0.0

def generate_melspectrogram(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
        if S.size == 0:
            st.error("❌ Impossibile generare lo spettrogramma: l'audio è troppo breve o non è valido.")
            return None
        mel_spec_db = librosa.power_to_db(S, ref=np.max)
        min_val = mel_spec_db.min()
        max_val = mel_spec_db.max()
        if max_val == min_val:
            st.error("❌ L'audio non contiene variazioni sufficienti per generare il video.")
            return None
        mel_spec_norm = (mel_spec_db - min_val) / (max_val - min_val)
        if mel_spec_norm.shape[1] == 0:
            st.error("❌ Lo spettrogramma è vuoto: l'audio è troppo breve o non contiene dati validi.")
            return None
        return mel_spec_norm
    except Exception as e:
        st.error(f"❌ Errore nella generazione dello spettrogramma: {str(e)}")
        return None

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0])

class VideoGenerator:
    def __init__(self, format_res: Tuple[int, int], level: str, fps: int = 30, bg_color: Tuple[int,int,int]=(255,255,255), line_color: Tuple[int,int,int]=(0,0,0), bpm: float = 0.0, mode: str = "Rete"):
        self.WIDTH, self.HEIGHT = format_res
        self.FPS = fps
        self.LEVEL = level
        self.MODE = mode
        self.TEMP_VIDEO = "temp_output.mp4"
        self.FINAL_VIDEO = "final_output.mp4"
        self.LINE_DENSITY = 30 if level == "soft" else 40 if level == "medium" else 50
        self.bg_color = bg_color
        self.line_color = line_color
        self.BEAT_INTERVAL = int(fps / (bpm / 60)) if bpm > 0 else fps

    def draw_barcode(self, frame, time_index, mel_spec):
        spacing = 10
        max_bars = self.WIDTH // spacing
        for i in range(min(max_bars, mel_spec.shape[0])):
            energy = mel_spec[i, time_index]
            if energy > 0.2:
                thickness = int(np.interp(energy, [0, 1], [1, 5]))
                x = i * spacing
                cv2.line(frame, (x, 0), (x, self.HEIGHT), self.line_color, thickness)

    def draw_lines(self, frame, time_index, mel_spec):
        center = self.HEIGHT // 2
        spacing = self.WIDTH / mel_spec.shape[0]
        for i in range(mel_spec.shape[0]):
            energy = mel_spec[i, time_index]
            if energy > 0.05:
                thickness = int(np.interp(energy, [0, 1], [1, 8]))
                x = int(i * spacing)
                cv2.line(frame, (x, center - 50), (x, center + 50), self.line_color, thickness)

    def draw_connected_network(self, frame, time_index, mel_spec):
        points = [(np.random.randint(0, self.WIDTH), np.random.randint(0, self.HEIGHT)) for _ in range(15)]
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                energy = mel_spec[np.random.randint(0, mel_spec.shape[0]), time_index]
                if energy > 0.3:
                    cv2.line(frame, points[i], points[j], self.line_color, 1)

    def generate_video(self, mel_spec: np.ndarray, audio_duration: float, sync_audio: bool = False) -> bool:
        try:
            for f in [self.TEMP_VIDEO, self.FINAL_VIDEO]:
                if os.path.exists(f): os.remove(f)
            total_frames = int(audio_duration * self.FPS)
            writer = cv2.VideoWriter(self.TEMP_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (self.WIDTH, self.HEIGHT))
            if not writer.isOpened():
                st.error("❌ Impossibile inizializzare il writer video.")
                return False
            progress = st.progress(0)
            for idx in range(total_frames):
                frame = np.full((self.HEIGHT, self.WIDTH, 3), self.bg_color, dtype=np.uint8)
                t_idx = int((idx / total_frames) * mel_spec.shape[1])
                if self.MODE == "Barcode" and idx % self.BEAT_INTERVAL == 0:
                    self.draw_barcode(frame, t_idx, mel_spec)
                elif self.MODE == "Linee":
                    self.draw_lines(frame, t_idx, mel_spec)
                else:
                    self.draw_connected_network(frame, t_idx, mel_spec)
                writer.write(frame)
                if idx % 5 == 0:
                    progress.progress((idx + 1) / total_frames)
            writer.release()
            progress.progress(1.0)
            if sync_audio and check_ffmpeg():
                try:
                    subprocess.run([
                        "ffmpeg", "-y", "-i", self.TEMP_VIDEO, "-i", "input_audio.wav",
                        "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-shortest", self.FINAL_VIDEO
                    ], check=True)
                    os.remove(self.TEMP_VIDEO)
                except:
                    os.rename(self.TEMP_VIDEO, self.FINAL_VIDEO)
            else:
                os.rename(self.TEMP_VIDEO, self.FINAL_VIDEO)
            return True
        except Exception as e:
            st.error(f"❌ Errore nella generazione: {str(e)}")
            return False

def main():
    st.set_page_config(page_title="🎵 AudioLine by Loop507", layout="centered")
    st.title("🎨 AudioLine by Loop507")
    uploaded_file = st.file_uploader("🎧 Carica file audio", type=["wav", "mp3"])
    if uploaded_file and validate_audio_file(uploaded_file):
        with open("input_audio.wav", "wb") as f:
            f.write(uploaded_file.read())
        y, sr, duration = load_and_process_audio("input_audio.wav")
        if y is None:
            return
        bpm = estimate_bpm(y, sr)
        bpm_display = f"{bpm:.1f}" if bpm and bpm > 0 else "non rilevato"
        st.info(f"🔊 Durata: {duration:.2f}s | BPM: {bpm_display}")
        mel_spec = generate_melspectrogram(y, sr)
        if mel_spec is None:
            return
        col1, col2, col3 = st.columns(3)
        with col1:
            video_format = st.selectbox("📐 Formato", {"16:9": (1280,720), "9:16": (720,1280), "1:1": (720,720), "4:3": (800,600)})
        with col2:
            level = st.selectbox("🎨 Effetto", ["soft", "medium", "hard"])
        with col3:
            fps = st.selectbox("🎞️ FPS", [5, 8, 10, 15, 24, 30], index=5)
        mode = st.selectbox("✨ Stile Video", ["Rete", "Linee", "Barcode"])
        bg = st.color_picker("🎨 Sfondo", "#FFFFFF")
        line = st.color_picker("🎨 Linee", "#000000")
        sync = st.checkbox("🔊 Sincronizza Audio")
        if st.button("🎬 Genera Video"):
            gen = VideoGenerator(video_format, level, fps, hex_to_bgr(bg), hex_to_bgr(line), bpm, mode)
            ok = gen.generate_video(mel_spec, duration, sync)
            if ok and os.path.exists("final_output.mp4"):
                with open("final_output.mp4", "rb") as f:
                    st.download_button("⬇️ Scarica Video", f, file_name="audioline_by_loop507.mp4", mime="video/mp4")

if __name__ == "__main__":
    main()
