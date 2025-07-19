# 🎵 AudioLinee - by Loop507 - Versione Estesa con Effetti e Colori per Frequenze

import streamlit as st
import numpy as np
import cv2
import librosa
import os
import subprocess
import gc
import shutil
from typing import Tuple, Optional
from scipy.ndimage import uniform_filter1d  # <--- AGGIUNTO PER SMOOTHING

MAX_DURATION = 300
MIN_DURATION = 1.0
MAX_FILE_SIZE = 50 * 1024 * 1024


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
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr / 2)
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
        mel_spec_norm = smooth_mel(mel_spec_norm, smoothing=5)  # <--- SMOOTHING INSERITO QUI
        return mel_spec_norm
    except Exception as e:
        st.error(f"❌ Errore nella generazione dello spettrogramma: {str(e)}")
        return None


def smooth_mel(mel, smoothing=5):
    return uniform_filter1d(mel, size=smoothing, axis=1)


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0])


class VideoGenerator:
    def __init__(self, format_res, level, fps=30, bg_color=(255, 255, 255), freq_colors=None, effect_mode="connessioni"):
        self.W, self.H = format_res
        self.FPS = fps
        self.LEVEL = level
        self.bg_color = bg_color
        self.freq_colors = freq_colors or {
            'low': (0, 0, 0),     # nero
            'mid': (0, 0, 255),   # rosso
            'high': (255, 0, 0)   # blu
        }
        self.effect_mode = effect_mode
        self.TEMP = "temp_output.mp4"
        self.FINAL = "final_output.mp4"
        self.density = 30 if level == "soft" else 45 if level == "medium" else 60

    def freq_to_color(self, i):
        if i < 42:
            return self.freq_colors['low']
        elif i < 85:
            return self.freq_colors['mid']
        return self.freq_colors['high']

    def draw_connected_lines(self, frame, mel, idx):
        pts = [(np.random.randint(0, self.W), np.random.randint(0, self.H)) for _ in range(15)]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                val = mel[np.random.randint(0, mel.shape[0]), idx]
                if val > 0.3:
                    color = self.freq_to_color(np.random.randint(0, mel.shape[0]))
                    thick = int(1 + val * 4)
                    cv2.line(frame, pts[i], pts[j], color, thick)

    def draw_burst_lines(self, frame, mel, idx):
        center = (self.W // 2, self.H // 2)
        for i in range(0, mel.shape[0], 5):
            angle = np.random.uniform(0, 2 * np.pi)
            length = int(mel[i, idx] * self.W / 2)
            x = int(center[0] + np.cos(angle) * length)
            y = int(center[1] + np.sin(angle) * length)
            color = self.freq_to_color(i)
            cv2.line(frame, center, (x, y), color, 1)

    def draw_jagged_lines(self, frame, mel, idx):
        for i in range(0, mel.shape[0], 10):
            y = int((i / mel.shape[0]) * self.H)
            x_start = 0
            for j in range(5):
                x_end = x_start + np.random.randint(10, 40)
                color = self.freq_to_color(i)
                volume = mel[i, idx]
                cv2.line(frame, (x_start, y), (x_end, y + np.random.randint(-10, 10)), color, max(1, int(volume * 5)))
                x_start = x_end

    def generate_video(self, mel, duration, sync_audio=True):
        for f in [self.TEMP, self.FINAL]:
            if os.path.exists(f): os.remove(f)
        total_frames = int(duration * self.FPS)
        writer = cv2.VideoWriter(self.TEMP, cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (self.W, self.H))
        if not writer.isOpened():
            st.error("❌ Impossibile inizializzare il writer video.")
            return False
        progress = st.progress(0)
        status = st.empty()
        for i in range(total_frames):
            try:
                frame = np.ones((self.H, self.W, 3), dtype=np.uint8) * np.array(self.bg_color, dtype=np.uint8)
                t_idx = int((i / total_frames) * mel.shape[1])
                if self.effect_mode == "connessioni":
                    self.draw_connected_lines(frame, mel, t_idx)
                elif self.effect_mode == "esplosione":
                    self.draw_burst_lines(frame, mel, t_idx)
                elif self.effect_mode == "frastagliate":
                    self.draw_jagged_lines(frame, mel, t_idx)
                writer.write(frame)
                if i % 10 == 0:
                    progress.progress((i + 1) / total_frames)
                    status.text(f"🎮 Frame {i + 1}/{total_frames}")
            except Exception as e:
                st.warning(f"⚠️ Errore al frame {i}: {str(e)}")
        writer.release()
        if sync_audio and check_ffmpeg():
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", self.TEMP, "-i", "input_audio.wav",
                    "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
                    "-c:a", "aac", "-shortest", self.FINAL
                ], capture_output=True)
                os.remove(self.TEMP)
            except Exception as e:
                st.error(f"Errore FFmpeg: {str(e)}")
                os.rename(self.TEMP, self.FINAL)
        else:
            os.rename(self.TEMP, self.FINAL)
        status.text("✅ Video completato!")
        gc.collect()
        return True


def main():
    st.set_page_config(page_title="🎵 AudioLinee - ColorFX", layout="centered")
    st.title("🎨 AudioLinee - ColorFX Edition")
    st.markdown("Carica un file audio per generare un video artistico basato sulle frequenze.")

    uploaded_file = st.file_uploader("🎧 Carica un file audio (.wav, .mp3)", type=["wav", "mp3"])
    if uploaded_file is None:
        return

    if not validate_audio_file(uploaded_file):
        return

    with open("input_audio.wav", "wb") as f:
        f.write(uploaded_file.read())
    st.success("🔊 Audio caricato!")

    y, sr, audio_duration = load_and_process_audio("input_audio.wav")
    if y is None:
        return

    st.info(f"Durata audio: {audio_duration:.2f} secondi")

    mel_spec_norm = generate_melspectrogram(y, sr)
    if mel_spec_norm is None:
        return

    with st.expander("🎧 Impostazioni video"):
        format_res = st.selectbox("📀 Formato", {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (720, 720), "4:3": (800, 600)})
        level = st.selectbox("🎨 Livello effetto", ["soft", "medium", "hard"], index=1)
        fps = st.selectbox("🎞️ FPS", [5, 10, 15, 24, 30], index=3)
        bg_color = hex_to_bgr(st.color_picker("🖌️ Colore sfondo", "#FFFFFF"))
        effect_mode = st.selectbox("✨ Tipo di effetto", ["connessioni", "esplosione", "frastagliate"])

        st.markdown("🎨 **Colore per gamma di frequenze**")
        low = hex_to_bgr(st.color_picker("Basse frequenze", "#000000"))
        mid = hex_to_bgr(st.color_picker("Medie frequenze", "#FF0000"))
        high = hex_to_bgr(st.color_picker("Alte frequenze", "#0000FF"))
        freq_colors = {"low": low, "mid": mid, "high": high}

    sync_audio = st.checkbox("🔊 Aggiungi audio al video", value=True)

    if st.button("🎬 Genera Video"):
        generator = VideoGenerator(format_res, level, fps, bg_color, freq_colors, effect_mode)
        success = generator.generate_video(mel_spec_norm, audio_duration, sync_audio)
        if success and os.path.exists("final_output.mp4"):
            with open("final_output.mp4", "rb") as f:
                st.download_button("⬇️ Scarica il video", f, file_name="audiolinee_output.mp4", mime="video/mp4")
            size_mb = os.path.getsize("final_output.mp4") / (1024 * 1024)
            st.info(f"📁 Dimensione video: {size_mb:.1f} MB")

    if st.button("🪑 Pulisci file temporanei"):
        for f in ["input_audio.wav", "temp_output.mp4", "final_output.mp4"]:
            if os.path.exists(f):
                os.remove(f)
        st.success("✅ Pulizia completata!")


if __name__ == "__main__":
    main()
