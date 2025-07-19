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
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

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
    def __init__(self, format_res: Tuple[int, int], level: str, fps: int = 30, bg_color: Tuple[int,int,int]=(255,255,255), freq_colors=None, effect_mode="connessioni"):
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
        self.TEMP_VIDEO = "temp_output.mp4"
        self.FINAL_VIDEO = "final_output.mp4"
        self.LINE_DENSITY = 30 if level == "soft" else 45 if level == "medium" else 60

    def freq_to_color(self, i):
        if i < 42:
            return self.freq_colors['low']
        elif i < 85:
            return self.freq_colors['mid']
        else:
            return self.freq_colors['high']

    def draw_connected_lines(self, frame, mel_spec_norm, time_index):
        points = [(np.random.randint(0, self.W), np.random.randint(0, self.H)) for _ in range(15)]
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                val = mel_spec_norm[np.random.randint(0, mel_spec_norm.shape[0]), time_index]
                if val > 0.3:
                    color = self.freq_to_color(np.random.randint(0, mel_spec_norm.shape[0]))
                    thickness = max(1, int(val * 4))
                    cv2.line(frame, points[i], points[j], color, thickness)

    def draw_burst_lines(self, frame, mel_spec_norm, time_index):
        center = (self.W // 2, self.H // 2)
        for i in range(0, mel_spec_norm.shape[0], 5):
            angle = np.random.uniform(0, 2 * np.pi)
            length = int(mel_spec_norm[i, time_index] * self.W / 2)
            x = int(center[0] + np.cos(angle) * length)
            y = int(center[1] + np.sin(angle) * length)
            color = self.freq_to_color(i)
            thickness = max(1, int(mel_spec_norm[i, time_index] * 5))
            cv2.line(frame, center, (x, y), color, thickness)

    def draw_jagged_lines(self, frame, mel_spec_norm, time_index):
        for i in range(0, mel_spec_norm.shape[0], 10):
            y = int((i / mel_spec_norm.shape[0]) * self.H)
            x_start = 0
            for _ in range(5):
                x_end = x_start + np.random.randint(10, 40)
                color = self.freq_to_color(i)
                volume = mel_spec_norm[i, time_index]
                thickness = max(1, int(volume * 5))
                cv2.line(frame, (x_start, y), (x_end, y + np.random.randint(-10, 10)), color, thickness)
                x_start = x_end

    def generate_video(self, mel_spec_norm: np.ndarray, audio_duration: float, sync_audio: bool = False) -> bool:
        try:
            for f in [self.TEMP_VIDEO, self.FINAL_VIDEO]:
                if os.path.exists(f):
                    os.remove(f)
            total_frames = int(audio_duration * self.FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(self.TEMP_VIDEO, fourcc, self.FPS, (self.W, self.H))
            if not video_writer.isOpened():
                st.error("❌ Impossibile inizializzare il writer video.")
                return False
            progress_bar = st.progress(0)
            status_text = st.empty()
            for frame_idx in range(total_frames):
                frame = np.ones((self.H, self.W, 3), dtype=np.uint8) * np.array(self.bg_color, dtype=np.uint8)
                time_index = int((frame_idx / total_frames) * mel_spec_norm.shape[1])
                time_index = max(0, min(time_index, mel_spec_norm.shape[1] - 1))
                if self.effect_mode == "connessioni":
                    self.draw_connected_lines(frame, mel_spec_norm, time_index)
                elif self.effect_mode == "esplosione":
                    self.draw_burst_lines(frame, mel_spec_norm, time_index)
                elif self.effect_mode == "frastagliate":
                    self.draw_jagged_lines(frame, mel_spec_norm, time_index)
                video_writer.write(frame)
                if frame_idx % 10 == 0:
                    progress = (frame_idx + 1) / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"🎬 Generazione frame {frame_idx + 1}/{total_frames} ({progress * 100:.1f}%)")
            video_writer.release()
            progress_bar.progress(1.0)
            status_text.text("✅ Video generato! Sincronizzazione audio...")
            if sync_audio:
                if not check_ffmpeg():
                    st.warning("⚠️ FFmpeg non trovato. Video senza audio.")
                    os.rename(self.TEMP_VIDEO, self.FINAL_VIDEO)
                    return True
                try:
                    cmd = [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", self.TEMP_VIDEO,
                        "-i", "input_audio.wav",
                        "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
                        "-c:a", "aac", "-shortest",
                        self.FINAL_VIDEO
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if result.returncode != 0:
                        st.error(f"❌ Errore FFmpeg: {result.stderr}")
                        os.rename(self.TEMP_VIDEO, self.FINAL_VIDEO)
                    else:
                        os.remove(self.TEMP_VIDEO)
                except subprocess.TimeoutExpired:
                    st.error("❌ Timeout FFmpeg.")
                    os.rename(self.TEMP_VIDEO, self.FINAL_VIDEO)
                except Exception as e:
                    st.error(f"❌ Errore nella sincronizzazione audio: {str(e)}")
                    os.rename(self.TEMP_VIDEO, self.FINAL_VIDEO)
            else:
                os.rename(self.TEMP_VIDEO, self.FINAL_VIDEO)
            status_text.text("✅ Video completato con successo!")
            return True
        except Exception as e:
            st.error(f"❌ Errore nella generazione del video: {str(e)}")
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
    if uploaded_file is not None:
        if not validate_audio_file(uploaded_file):
            return
        with open("input_audio.wav", "wb") as f:
            f.write(uploaded_file.read())
        st.success("🔊 Audio caricato correttamente!")

        y, sr, audio_duration = load_and_process_audio("input_audio.wav")
        if y is None:
            return
        st.info(f"🔊 Durata audio: {audio_duration:.2f} secondi")

        with st.spinner("📊 Analisi audio in corso..."):
            mel_spec_norm = generate_melspectrogram(y, sr)
        if mel_spec_norm is None:
            return
        st.success("✅ Analisi audio completata!")

        col1, col2, col3 = st.columns(3)
        with col1:
            video_format = st.selectbox("📐 Formato video", list(FORMAT_RESOLUTIONS.keys()))
        with col2:
            effect_level = st.selectbox("🎨 Livello effetti", ["soft", "medium", "hard"])
        with col3:
            fps_choice = st.selectbox("🎞️ Fotogrammi al secondo (FPS)", [10, 15, 24, 30], index=3)

        effect_mode = st.selectbox("✨ Effetto artistico", ["connessioni", "esplosione", "frastagliate"])

        st.markdown("🎨 Scegli i colori per le frequenze (basso, medio, alto):")
        col_low, col_mid, col_high = st.columns(3)
        with col_low:
            low_color = st.color_picker("Basse frequenze", "#000000")
        with col_mid:
            mid_color = st.color_picker("Medie frequenze", "#FF0000")
        with col_high:
            high_color = st.color_picker("Alte frequenze", "#0000FF")

        bg_color_hex = st.color_picker("🎨 Colore sfondo", "#FFFFFF")

        sync_audio = st.checkbox("🔊 Sincronizza audio nel video", value=True)
        if sync_audio and not check_ffmpeg():
            st.warning("⚠️ FFmpeg non disponibile, la sincronizzazione audio è disabilitata.")
            sync_audio = False

        if st.button("🎬 Genera Video"):
            freq_colors = {
                'low': hex_to_bgr(low_color),
                'mid': hex_to_bgr(mid_color),
                'high': hex_to_bgr(high_color)
            }
            bg_color_bgr = hex_to_bgr(bg_color_hex)
            generator = VideoGenerator(
                FORMAT_RESOLUTIONS[video_format],
                effect_level,
                fps=fps_choice,
                bg_color=bg_color_bgr,
                freq_colors=freq_colors,
                effect_mode=effect_mode
            )
            success = generator.generate_video(mel_spec_norm, audio_duration, sync_audio)
            if success and os.path.exists(generator.FINAL_VIDEO):
                with open(generator.FINAL_VIDEO, "rb") as f:
                    st.download_button("⬇️ Scarica il video", f, file_name=f"audio_linee_{video_format}_{effect_level}_{effect_mode}.mp4", mime="video/mp4")
                file_size = os.path.getsize(generator.FINAL_VIDEO)
                st.info(f"📁 Dimensione file: {file_size / 1024 / 1024:.1f} MB")
            else:
                st.error("❌ Errore nella generazione del video.")

        if st.button("🧹 Pulisci file temporanei"):
            temp_files = ["input_audio.wav", "temp_output.mp4", "final_output.mp4"]
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
            st.success("✅ File temporanei eliminati!")

if __name__ == "__main__":
    main()
