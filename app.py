# 🎵 AudioLine by Loop507 - Versione Corretta

import streamlit as st
import numpy as np
import cv2
import librosa
import os
import subprocess
import gc
import shutil
from typing import Tuple, Optional
import contextlib

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
    """Verifica se FFmpeg è disponibile"""
    return shutil.which("ffmpeg") is not None

def validate_audio_file(uploaded_file) -> bool:
    """Valida il file audio caricato"""
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"❌ File troppo grande ({uploaded_file.size / 1024 / 1024:.1f}MB). Limite: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
        return False
    return True

def load_and_process_audio(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
    """Carica e processa il file audio con gestione errori migliorata"""
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
    """Stima il BPM dell'audio"""
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo) if tempo > 0 else 120.0  # Default fallback
    except Exception:
        return 120.0  # BPM di fallback

def generate_melspectrogram(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """Genera mel-spettrogramma normalizzato con controlli migliorati"""
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
        mel_spec_db = librosa.power_to_db(S, ref=np.max)
        
        min_val = mel_spec_db.min()
        max_val = mel_spec_db.max()
        
        if max_val == min_val or np.isnan(max_val) or np.isnan(min_val):
            st.error("❌ Spettrogramma non valido")
            return None
        
        mel_spec_norm = (mel_spec_db - min_val) / (max_val - min_val)
        
        # Verifica che abbiamo dati validi
        if mel_spec_norm.shape[1] == 0:
            st.error("❌ Spettrogramma vuoto")
            return None
            
        return mel_spec_norm
    except Exception as e:
        st.error(f"❌ Errore nella generazione dello spettrogramma: {e}")
        return None

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Converte colore hex in BGR per OpenCV"""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0])

def cleanup_files(*files):
    """Pulisce i file temporanei"""
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception:
            pass

class VideoGenerator:
    def __init__(self, format_res, level, fps, bg_color, line_color, bpm, mode):
        self.WIDTH, self.HEIGHT = format_res
        self.FPS = fps
        self.LEVEL = level
        self.bg_color = bg_color
        self.line_color = line_color
        self.TEMP_VIDEO = "temp_output.mp4"
        self.FINAL_VIDEO = "final_output.mp4"
        
        # Densità basata sul livello
        density_map = {"soft": 20, "medium": 35, "hard": 50}
        self.LINE_DENSITY = density_map.get(level, 35)
        
        self.BPM = max(60.0, min(200.0, bpm))  # Clamp BPM tra 60-200
        self.MODE = mode

    def generate_barcode_frame(self, mel_spec_norm: np.ndarray, time_index: int, beat_intensity: float) -> np.ndarray:
        """Genera un singolo frame in modalità barcode"""
        frame = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        frame[:] = self.bg_color
        
        # Calcola spaziatura dinamica basata sulla larghezza
        spacing = max(2, self.WIDTH // self.LINE_DENSITY)
        max_bars = self.WIDTH // spacing
        
        for j in range(max_bars):
            # Mappa le barre su tutto lo spettro di frequenze
            freq_idx = int((j / max_bars) * mel_spec_norm.shape[0])
            freq_idx = min(freq_idx, mel_spec_norm.shape[0] - 1)
            
            energy = mel_spec_norm[freq_idx, time_index]
            
            # Modula l'energia con il beat (invece di saltare frame)
            energy_modulated = energy * beat_intensity
            
            # Calcola altezza e spessore della barra
            bar_height = int(np.interp(energy_modulated, [0, 1], [10, self.HEIGHT - 10]))
            thickness = max(1, int(np.interp(energy_modulated, [0, 1], [1, spacing - 1])))
            
            x = j * spacing
            y_start = (self.HEIGHT - bar_height) // 2
            y_end = y_start + bar_height
            
            # Disegna la barra
            cv2.rectangle(frame, (x, y_start), (x + thickness, y_end), self.line_color, -1)
        
        return frame

    def calculate_beat_intensity(self, frame_number: int) -> float:
        """Calcola l'intensità basata sul BPM per modulazione smooth"""
        if self.BPM <= 0:
            return 1.0
        
        # Converti BPM in radianti per frame
        beat_frequency = (self.BPM / 60.0) * 2 * np.pi / self.FPS
        
        # Usa una sinusoide per modulazione smooth invece di step discreti
        beat_phase = frame_number * beat_frequency
        intensity = 0.7 + 0.3 * np.sin(beat_phase)  # Varia tra 0.4 e 1.0
        
        return max(0.4, min(1.0, intensity))

    def generate_video(self, mel_spec_norm: np.ndarray, duration: float) -> bool:
        """Genera il video con gestione errori migliorata"""
        video_writer = None
        try:
            # Pulizia file esistenti
            cleanup_files(self.TEMP_VIDEO, self.FINAL_VIDEO)
            
            total_frames = int(duration * self.FPS)
            if total_frames <= 0:
                st.error("❌ Durata video non valida")
                return False
            
            # Inizializza video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(self.TEMP_VIDEO, fourcc, self.FPS, (self.WIDTH, self.HEIGHT))
            
            if not video_writer.isOpened():
                st.error("❌ Impossibile creare il video writer")
                return False
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(total_frames):
                try:
                    # Calcola indice temporale con bounds checking
                    time_index = min(int((i / total_frames) * mel_spec_norm.shape[1]), 
                                   mel_spec_norm.shape[1] - 1)
                    
                    # Calcola intensità beat per modulazione smooth
                    beat_intensity = self.calculate_beat_intensity(i)
                    
                    # Genera frame basato sulla modalità
                    if self.MODE == "Barcode":
                        frame = self.generate_barcode_frame(mel_spec_norm, time_index, beat_intensity)
                    else:
                        # Modalità di fallback
                        frame = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8) * 128
                    
                    video_writer.write(frame)
                    
                    # Aggiorna progress ogni 10 frame per performance
                    if i % 10 == 0:
                        progress = (i + 1) / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Generazione frame {i+1}/{total_frames} ({progress*100:.1f}%)")
                
                except Exception as e:
                    st.error(f"❌ Errore nel frame {i}: {e}")
                    return False
            
            video_writer.release()
            video_writer = None
            
            # Rinomina file finale
            if os.path.exists(self.TEMP_VIDEO):
                os.rename(self.TEMP_VIDEO, self.FINAL_VIDEO)
                status_text.text("✅ Video generato con successo!")
                return True
            else:
                st.error("❌ File temporaneo non trovato")
                return False
                
        except Exception as e:
            st.error(f"❌ Errore nella generazione video: {e}")
            return False
        finally:
            # Cleanup garantito
            if video_writer:
                video_writer.release()
            cleanup_files(self.TEMP_VIDEO)

def main():
    st.set_page_config(page_title="🎵 AudioLine by Loop507", layout="centered")
    st.title("🎵 AudioLine by Loop507")
    st.markdown("*Trasforma la tua musica in visualizzazioni sincronizzate*")
    
    # Verifica FFmpeg
    if not check_ffmpeg():
        st.warning("⚠️ FFmpeg non trovato. Alcune funzionalità potrebbero non funzionare.")
    
    uploaded = st.file_uploader("🎧 Carica il tuo file audio", type=["wav", "mp3"])
    
    if uploaded:
        if not validate_audio_file(uploaded):
            return
        
        # Salva file temporaneo
        temp_audio = "input_audio.wav"
        try:
            with open(temp_audio, "wb") as f:
                f.write(uploaded.read())
            
            # Processa audio
            with st.spinner("🔄 Elaborazione audio in corso..."):
                y, sr, duration = load_and_process_audio(temp_audio)
                
            if y is None:
                return
            
            # Genera spettrogramma
            with st.spinner("📊 Analisi spettrale..."):
                mel = generate_melspectrogram(y, sr)
                
            if mel is None:
                return
            
            # Stima BPM
            with st.spinner("🎵 Rilevamento BPM..."):
                bpm = estimate_bpm(y, sr)
            
            # Libera memoria audio
            del y
            gc.collect()
            
            st.success(f"✅ Audio elaborato: {duration:.1f}s, BPM: {bpm:.1f}")
            
            # Interfaccia controlli
            st.markdown("### ⚙️ Configurazione Video")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                fmt_key = st.selectbox("📐 Formato", list(FORMAT_RESOLUTIONS.keys()), index=0)
                format_res = FORMAT_RESOLUTIONS[fmt_key]
                st.caption(f"Risoluzione: {format_res[0]}×{format_res[1]}")
            
            with col2:
                level = st.selectbox("🎨 Intensità effetto", ["soft", "medium", "hard"], index=1)
                level_desc = {"soft": "Delicato", "medium": "Bilanciato", "hard": "Intenso"}
                st.caption(level_desc[level])
            
            with col3:
                fps = st.selectbox("🎞 FPS", [12, 15, 20, 24, 30], index=2)
                st.caption(f"Frame al secondo")
            
            # Controlli colore
            col4, col5 = st.columns(2)
            with col4:
                bg = st.color_picker("🎨 Colore sfondo", "#000000")
            with col5:
                line = st.color_picker("🎨 Colore linee", "#FFFFFF")
            
            mode = st.selectbox("✨ Modalità visualizzazione", ["Barcode"], index=0)
            
            # Generazione video
            if st.button("🎬 Genera Video", type="primary"):
                with st.spinner("🎬 Creazione video in corso..."):
                    generator = VideoGenerator(
                        format_res, level, fps, 
                        hex_to_bgr(bg), hex_to_bgr(line), 
                        bpm, mode
                    )
                    
                    if generator.generate_video(mel, duration):
                        st.balloons()
                        
                        # Download button
                        try:
                            with open("final_output.mp4", "rb") as video_file:
                                st.download_button(
                                    label="⬇️ Scarica Video",
                                    data=video_file,
                                    file_name=f"audioline_{uploaded.name}_{fmt_key}.mp4",
                                    mime="video/mp4",
                                    type="primary"
                                )
                        except FileNotFoundError:
                            st.error("❌ File video non trovato")
                    
                    # Cleanup
                    del mel
                    gc.collect()
        
        finally:
            # Pulizia file temporanei
            cleanup_files(temp_audio, "final_output.mp4")

if __name__ == "__main__":
    main()
