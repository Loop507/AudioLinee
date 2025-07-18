# 🎵 AudioLinee.py (by Loop507) - Versione Corretta
# Generatore di video visivi sincronizzati con l'audio
# Realizzato con Streamlit - Funziona online su Streamlit Cloud

import streamlit as st
import numpy as np
import cv2
import librosa
import matplotlib.cm as cm
import matplotlib.colors as colors
import os
import subprocess
import gc
from typing import Tuple, Optional

# Configurazione pagina
st.set_page_config(page_title="🎵 AudioLinee - by Loop507", layout="centered")
st.title("🎨 AudioLinee")
st.markdown("### by Loop507")
st.markdown("Carica un file audio e genera un video visivo sincronizzato.")

# Costanti globali
MAX_DURATION = 300  # 5 minuti massimo
MIN_DURATION = 1.0  # 1 secondo minimo
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def validate_audio_file(uploaded_file) -> bool:
    """Valida il file audio caricato"""
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"❌ File troppo grande ({uploaded_file.size / 1024 / 1024:.1f}MB). Limite: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
        return False
    return True

def load_and_process_audio(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
    """Carica e processa l'audio con controlli di sicurezza"""
    try:
        # Carica l'audio
        y, sr = librosa.load(file_path, sr=None)
        
        # Converti in mono se necessario
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
        
        # Controlla se l'audio è vuoto
        if len(y) == 0:
            st.error("❌ Il file audio è vuoto o non è stato caricato correttamente.")
            return None, None, None
        
        # Calcola la durata
        audio_duration = librosa.get_duration(y=y, sr=sr)
        
        # Valida la durata
        if audio_duration < MIN_DURATION:
            st.error(f"❌ L'audio deve essere lungo almeno {MIN_DURATION} secondi. Durata attuale: {audio_duration:.2f}s")
            return None, None, None
        
        # Tronca se troppo lungo
        if audio_duration > MAX_DURATION:
            st.warning(f"⚠️ Audio troppo lungo ({audio_duration:.1f}s). Verrà troncato a {MAX_DURATION}s.")
            y = y[:int(MAX_DURATION * sr)]
            audio_duration = MAX_DURATION
        
        return y, sr, audio_duration
        
    except Exception as e:
        st.error(f"❌ Errore nel caricamento dell'audio: {str(e)}")
        return None, None, None

def generate_melspectrogram(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """Genera lo spettrogramma mel con controlli di sicurezza"""
    try:
        # Genera lo spettrogramma
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
        
        if S.size == 0:
            st.error("❌ Impossibile generare lo spettrogramma: l'audio è troppo breve o non è valido.")
            return None
        
        # Converti in dB
        mel_spec_db = librosa.power_to_db(S, ref=np.max)
        
        # Normalizza con controllo divisione per zero
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
    """Classe per la generazione video con ottimizzazioni"""
    
    def __init__(self, format_res: Tuple[int, int], level: str, fps: int = 30):
        self.FRAME_WIDTH, self.FRAME_HEIGHT = format_res
        self.FPS = fps
        self.LEVEL = level
        self.TEMP_VIDEO = "temp_output.mp4"
        self.FINAL_VIDEO = "final_output.mp4"
        
        # Configurazioni basate sul livello
        self.COLOR_GRADIENT = level in ["medium", "hard"]
        self.ROTATE_LINES = level == "hard"
        self.MULTI_BAND_FREQUENCIES = level == "hard"
        self.LINE_DENSITY = 30 if level == "soft" else 40 if level == "medium" else 50
        
        # Cache per i colori
        self.colors_cache = {}
        
        # Inizializza la mappa colori
        self.cmap = cm.get_cmap("plasma")
        self.norm = colors.Normalize(vmin=0, vmax=1)
        self.scalar_map = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
    
    def energy_to_color(self, energy: float) -> Tuple[int, int, int]:
        """Converte l'energia in colore con cache"""
        # Quantizza per cache
        key = int(energy * 100)
        if key not in self.colors_cache:
            rgba = self.scalar_map.to_rgba(energy)
            self.colors_cache[key] = tuple(int(x * 255) for x in rgba[:3])
        return self.colors_cache[key]
    
    def generate_distorted_line(self, frame: np.ndarray, x1: float, y1: float, 
                              x2: float, y2: float, distortion: float, 
                              energy: float = 1.0, angle: float = 0):
        """Genera una linea distorta con ottimizzazioni"""
        try:
            num_points = 25 if self.LEVEL == "soft" else 35 if self.LEVEL == "medium" else 50
            x = np.linspace(x1, x2, num_points)
            y = np.linspace(y1, y2, num_points)
            
            # Distorsione sinusoidale
            y += distortion * np.sin(np.linspace(0, 2 * np.pi, num_points))
            
            # Rotazione se abilitata
            if self.ROTATE_LINES:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                theta = np.radians(angle)
                x_centered = x - cx
                y_centered = y - cy
                xr = x_centered * np.cos(theta) - y_centered * np.sin(theta)
                yr = x_centered * np.sin(theta) + y_centered * np.cos(theta)
                x, y = xr + cx, yr + cy
            
            # Disegna la linea
            for i in range(num_points - 1):
                color = self.energy_to_color(energy) if self.COLOR_GRADIENT else (255, 255, 255)
                pt1 = (max(0, min(int(x[i]), self.FRAME_WIDTH-1)), 
                       max(0, min(int(y[i]), self.FRAME_HEIGHT-1)))
                pt2 = (max(0, min(int(x[i+1]), self.FRAME_WIDTH-1)), 
                       max(0, min(int(y[i+1]), self.FRAME_HEIGHT-1)))
                cv2.line(frame, pt1, pt2, color, 1)
                
        except Exception as e:
            # Ignora errori di singole linee per non bloccare il video
            pass
    
    def generate_video(self, mel_spec_norm: np.ndarray, audio_duration: float, 
                      sync_audio: bool = False) -> bool:
        """Genera il video con progress tracking"""
        try:
            # Rimuovi file esistenti
            for f in [self.TEMP_VIDEO, self.FINAL_VIDEO]:
                if os.path.exists(f):
                    os.remove(f)
            
            total_frames = int(audio_duration * self.FPS)
            
            # Inizializza il writer del video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                self.TEMP_VIDEO, fourcc, self.FPS, 
                (self.FRAME_WIDTH, self.FRAME_HEIGHT)
            )
            
            if not video_writer.isOpened():
                st.error("❌ Impossibile inizializzare il writer video.")
                return False
            
            # Prepara le bande di frequenza se necessario
            if self.MULTI_BAND_FREQUENCIES:
                bands = np.array_split(mel_spec_norm, 4)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Genera i frame
            for frame_idx in range(total_frames):
                try:
                    frame = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3), dtype=np.uint8)
                    
                    # Calcolo time_index con protezione completa
                    time_index = int((frame_idx / total_frames) * mel_spec_norm.shape[1])
                    time_index = max(0, min(time_index, mel_spec_norm.shape[1] - 1))
                    
                    # Linee verticali
                    for i in range(self.LINE_DENSITY):
                        y1 = i * (self.FRAME_HEIGHT / self.LINE_DENSITY)
                        y2 = y1 + (self.FRAME_HEIGHT / self.LINE_DENSITY)
                        x1, x2 = 0, self.FRAME_WIDTH
                        
                        if self.MULTI_BAND_FREQUENCIES:
                            band_index = i % len(bands)
                            freq_index = min(i, bands[band_index].shape[0] - 1)
                            energy = bands[band_index][freq_index, time_index]
                            distortion = energy * 30
                        else:
                            freq_index = i % mel_spec_norm.shape[0]
                            energy = mel_spec_norm[freq_index, time_index]
                            distortion = energy * 30
                        
                        angle = int(energy * 20) if self.ROTATE_LINES else 0
                        self.generate_distorted_line(frame, x1, y1, x2, y2, distortion, energy, angle)
                    
                    # Linee orizzontali
                    for j in range(self.LINE_DENSITY):
                        x1 = j * (self.FRAME_WIDTH / self.LINE_DENSITY)
                        x2 = x1 + (self.FRAME_WIDTH / self.LINE_DENSITY)
                        y1, y2 = 0, self.FRAME_HEIGHT
                        
                        if self.MULTI_BAND_FREQUENCIES:
                            band_index = j % len(bands)
                            freq_index = min(j, bands[band_index].shape[0] - 1)
                            energy = bands[band_index][freq_index, time_index]
                            distortion = energy * 30
                        else:
                            freq_index = j % mel_spec_norm.shape[0]
                            energy = mel_spec_norm[freq_index, time_index]
                            distortion = energy * 30
                        
                        angle = int(energy * 20) if self.ROTATE_LINES else 0
                        self.generate_distorted_line(frame, x1, y1, x2, y2, distortion, energy, angle)
                    
                    # Effetto scala basato sull'energia media
                    if self.LEVEL in ["medium", "hard"]:
                        energy_frame = np.mean(mel_spec_norm[:, time_index])
                        scale_factor = 1 + 0.05 * (energy_frame - 0.5)
                        scale_factor = max(0.95, min(1.05, scale_factor))  # Limita il scaling
                        
                        if abs(scale_factor - 1.0) > 0.01:  # Solo se significativo
                            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, 
                                             interpolation=cv2.INTER_LINEAR)
                            
                            # Centra il frame ridimensionato
                            new_h, new_w = frame.shape[:2]
                            if new_h != self.FRAME_HEIGHT or new_w != self.FRAME_WIDTH:
                                top = max(0, (self.FRAME_HEIGHT - new_h) // 2)
                                bottom = max(0, self.FRAME_HEIGHT - new_h - top)
                                left = max(0, (self.FRAME_WIDTH - new_w) // 2)
                                right = max(0, self.FRAME_WIDTH - new_w - left)
                                
                                frame = cv2.copyMakeBorder(
                                    frame, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
                                )
                    
                    video_writer.write(frame)
                    
                    # Aggiorna progress
                    if frame_idx % 10 == 0:  # Aggiorna ogni 10 frame
                        progress = (frame_idx + 1) / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"🎬 Generazione frame {frame_idx + 1}/{total_frames} ({progress*100:.1f}%)")
                    
                except Exception as e:
                    st.warning(f"⚠️ Errore nel frame {frame_idx}: {str(e)}")
                    continue
            
            video_writer.release()
            progress_bar.progress(1.0)
            status_text.text("✅ Video generato! Sincronizzazione audio...")
            
            # Sincronizza l'audio al video se richiesto
            if sync_audio:
                try:
                    cmd = [
                        "ffmpeg", "-y", "-loglevel", "error",
                        "-i", self.TEMP_VIDEO,
                        "-i", "input_audio.wav",
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-strict", "experimental",
                        "-shortest",  # Usa la durata più breve
                        self.FINAL_VIDEO
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        st.error(f"❌ Errore FFmpeg: {result.stderr}")
                        return False
                    
                    os.remove(self.TEMP_VIDEO)
                except Exception as e:
                    st.error(f"❌ Errore nella sincronizzazione audio: {str(e)}")
                    return False
            else:
                os.rename(self.TEMP_VIDEO, self.FINAL_VIDEO)
            
            status_text.text("✅ Video completato con successo!")
            return True
            
        except Exception as e:
            st.error(f"❌ Errore nella generazione del video: {str(e)}")
            return False
        finally:
            # Cleanup
            if 'video_writer' in locals():
                video_writer.release()
            gc.collect()

def main():
    """Funzione principale dell'applicazione"""
    FORMAT_RESOLUTIONS = {
        "16:9": (1920, 1080),
        "9:16": (1080, 1920),
        "1:1": (1080, 1080),
        "4:3": (800, 600)
    }
    
    # Interfaccia utente
    uploaded_file = st.file_uploader("🎧 Carica un file audio (.wav o .mp3)", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        # Valida il file
        if not validate_audio_file(uploaded_file):
            return
        
        # Salva il file
        with open("input_audio.wav", "wb") as f:
            f.write(uploaded_file.read())
        st.success("🔊 Audio caricato correttamente!")
        
        # Carica e processa l'audio
        y, sr, audio_duration = load_and_process_audio("input_audio.wav")
        if y is None:
            return
        
        st.info(f"🔊 Durata audio: {audio_duration:.2f} secondi")
        
        # Genera lo spettrogramma
        with st.spinner("📊 Analisi audio in corso..."):
            mel_spec_norm = generate_melspectrogram(y, sr)
        
        if mel_spec_norm is None:
            return
        
        st.success("✅ Analisi audio completata!")
        
        # Controlli utente
        col1, col2 = st.columns(2)
        with col1:
            video_format = st.selectbox("📐 Formato video", ["16:9", "9:16", "1:1", "4:3"])
        with col2:
            effect_level = st.selectbox("🎨 Livello effetti", ["soft", "medium", "hard"])
        
        sync_audio = st.checkbox("🔊 Sincronizza l'audio nel video")
        
        if st.button("🎬 Genera Video"):
            # Crea il generatore video
            generator = VideoGenerator(FORMAT_RESOLUTIONS[video_format], effect_level)
            
            # Genera il video
            success = generator.generate_video(mel_spec_norm, audio_duration, sync_audio)
            
            if success and os.path.exists("final_output.mp4"):
                # Offri il download
                with open("final_output.mp4", "rb") as f:
                    st.download_button(
                        "⬇️ Scarica il video",
                        f,
                        file_name=f"audiolinee_{video_format}_{effect_level}.mp4",
                        mime="video/mp4"
                    )
                
                # Mostra informazioni sul file
                file_size = os.path.getsize("final_output.mp4")
                st.info(f"📁 Dimensione file: {file_size / 1024 / 1024:.1f} MB")
            else:
                st.error("❌ Si è verificato un errore nella generazione del video.")
        
        # Cleanup automatico
        if st.button("🧹 Pulisci file temporanei"):
            temp_files = ["input_audio.wav", "temp_output.mp4", "final_output.mp4"]
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
            st.success("✅ File temporanei eliminati!")

if __name__ == "__main__":
    main()
