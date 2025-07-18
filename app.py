# 🎵 AudioLinee.py (by Loop507) - Versione Modificata con linee audio-reactive
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
        y, sr = librosa.load(file_path, sr=None)
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
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

class VideoGenerator:
    def __init__(self, format_res: Tuple[int, int], level: str, fps: int = 30):
        self.FRAME_WIDTH, self.FRAME_HEIGHT = format_res
        self.FPS = fps
        self.LEVEL = level
        self.TEMP_VIDEO = "temp_output.mp4"
        self.FINAL_VIDEO = "final_output.mp4"
        self.LINE_DENSITY = 30 if level == "soft" else 40 if level == "medium" else 50

    def energy_to_color(self, energy: float) -> Tuple[int, int, int]:
        return (0, 0, 0)  # Nero puro per stile a china

    def draw_connected_network(self, frame, time_index, mel_spec_norm):
        n_mels = mel_spec_norm.shape[0]
        freq_energies = mel_spec_norm[:, time_index]
        height_step = self.FRAME_HEIGHT / n_mels

        volume = np.mean(freq_energies)  # volume complessivo
        volume_threshold = 0.1  # soglia minima volume per iniziare a disegnare

        if volume < volume_threshold:
            # volume troppo basso: niente linee
            return

        points = []
        thicknesses = []

        for i in range(n_mels):
            energy = freq_energies[i]
            if energy > 0.05:  # soglia minima energia locale
                x = int(np.interp(energy, [0, 1], [0, self.FRAME_WIDTH]))
                y = int(i * height_step + height_step / 2)
                points.append((x, y))
                # spessore proporzionale all’energia locale, da 1 a 5
                thickness = int(np.interp(energy, [0, 1], [1, 5]))
                thicknesses.append(thickness)

        # Disegna linee tra i punti con spessore basato sull’energia locale
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                # spessore linea = max spessore dei due punti collegati
                line_thickness = max(thicknesses[i], thicknesses[j])
                cv2.line(frame, points[i], points[j], (0, 0, 0), line_thickness)

    def generate_video(self, mel_spec_norm: np.ndarray, audio_duration: float, sync_audio: bool = False) -> bool:
        try:
            for f in [self.TEMP_VIDEO, self.FINAL_VIDEO]:
                if os.path.exists(f):
                    os.remove(f)
            total_frames = int(audio_duration * self.FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(self.TEMP_VIDEO, fourcc, self.FPS, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            if not video_writer.isOpened():
                st.error("❌ Impossibile inizializzare il writer video.")
                return False
            progress_bar = st.progress(0)
            status_text = st.empty()
            for frame_idx in range(total_frames):
                try:
                    frame = np.ones((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3), dtype=np.uint8) * 255  # sfondo bianco
                    time_index = int((frame_idx / total_frames) * mel_spec_norm.shape[1])
                    time_index = max(0, min(time_index, mel_spec_norm.shape[1] - 1))
                    self.draw_connected_network(frame, time_index=time_index, mel_spec_norm=mel_spec_norm)
                    video_writer.write(frame)
                    if frame_idx % 10 == 0:
                        progress = (frame_idx + 1) / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"🎬 Generazione frame {frame_idx + 1}/{total_frames} ({progress * 100:.1f}%)")
                except Exception as e:
                    st.warning(f"⚠️ Errore nel frame {frame_idx}: {str(e)}")
                    continue
            video_writer.release()
            progress_bar.progress(1.0)
            status_text.text("✅ Video generato! Sincronizzazione audio...")
            if sync_audio:
                if not check_ffmpeg():
                    st.warning("⚠️ FFmpeg non trovato. Video senza audio.")
                    os.rename(self.TEMP_VIDEO, self.FINAL_VIDEO)
                    return True
                try:
                    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", self.TEMP_VIDEO, "-i", "input_audio.wav",
                           "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", "-shortest", self.FINAL_VIDEO]
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
        "16:9": (1920, 1080),
        "9:16": (1080, 1920),
        "1:1": (1080, 1080),
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
        col1, col2 = st.columns(2)
        with col1:
            video_format = st.selectbox("📐 Formato video", ["16:9", "9:16", "1:1", "4:3"])
        with col2:
            effect_level = st.selectbox("🎨 Livello effetti", ["soft", "medium", "hard"])
        sync_audio = st.checkbox("🔊 Sincronizza l'audio nel video")
        if not check_ffmpeg():
            st.warning("⚠️ FFmpeg non disponibile - La sincronizzazione audio è disabilitata")
            sync_audio = False
        if st.button("🎬 Genera Video"):
            generator = VideoGenerator(FORMAT_RESOLUTIONS[video_format], effect_level)
            success = generator.generate_video(mel_spec_norm, audio_duration, sync_audio)
            if success and os.path.exists("final_output.mp4"):
                with open("final_output.mp4", "rb") as f:
                    st.download_button("⬇️ Scarica il video", f,
                                       file_name=f"audiolinee_{video_format}_{effect_level}.mp4",
                                       mime="video/mp4")
                file_size = os.path.getsize("final_output.mp4")
                st.info(f"📁 Dimensione file: {file_size / 1024 / 1024:.1f} MB")
            else:
                st.error("❌ Si è verificato un errore nella generazione del video.")
        if st.button("🧹 Pulisci file temporanei"):
            temp_files = ["input_audio.wav", "temp_output.mp4", "final_output.mp4"]
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
            st.success("✅ File temporanei eliminati!")

if __name__ == "__main__":
    main()
