# 🎵 AudioLine by Loop507 - Versione Completa

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
import tempfile

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
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def validate_audio_file(uploaded_file) -> bool:
    """Valida il file audio caricato"""
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"❌ File troppo grande ({uploaded_file.size / 1024 / 1024:.1f}MB). Limite: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
        return False
    return True

def load_and_process_audio(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
    """Carica e processa il file audio con gestione errori migliorata"""
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)  # Standardizza SR
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
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        # Converte numpy scalar a float python
        tempo_val = float(tempo.item()) if hasattr(tempo, 'item') else float(tempo)
        return tempo_val if tempo_val > 0 else 120.0
    except Exception as e:
        print(f"BPM estimation error: {e}")
        return 120.0

def generate_audio_features(y: np.ndarray, sr: int, fps: int) -> dict:
    """Genera tutte le features audio necessarie"""
    try:
        duration = len(y) / sr
        
        # Mel-spectrogram per visualizzazioni generali
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalizza mel-spectrogram
        mel_min, mel_max = float(mel_spec_db.min()), float(mel_spec_db.max())
        if mel_max != mel_min:
            mel_norm = (mel_spec_db - mel_min) / (mel_max - mel_min)
        else:
            mel_norm = np.zeros_like(mel_spec_db)
        
        # STFT per spectrum analyzer
        stft = librosa.stft(y, hop_length=512, n_fft=2048)
        magnitude = np.abs(stft)
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Normalizza STFT
        mag_min, mag_max = float(magnitude_db.min()), float(magnitude_db.max())
        if mag_max != mag_min:
            stft_norm = (magnitude_db - mag_min) / (mag_max - mag_min)
        else:
            stft_norm = np.zeros_like(magnitude_db)
        
        # RMS energy per linee
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        rms_min, rms_max = float(rms.min()), float(rms.max())
        rms_norm = (rms - rms_min) / (rms_max - rms_min) if rms_max != rms_min else rms
        
        # Beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        
        # Converte numpy scalars/arrays a tipi Python standard
        tempo_val = float(tempo.item()) if hasattr(tempo, 'item') else float(tempo)
        beats_array = beats.astype(np.float32) if hasattr(beats, 'astype') else beats
        
        return {
            'mel_spectrogram': mel_norm,
            'stft_magnitude': stft_norm,
            'rms_energy': rms_norm,
            'beats': beats_array,
            'tempo': tempo_val,
            'hop_length': 512,
            'sr': sr,
            'duration': duration
        }
    except Exception as e:
        st.error(f"❌ Errore nell'estrazione features: {e}")
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
    def __init__(self, format_res, level, fps, bg_color, line_color, mode):
        self.WIDTH, self.HEIGHT = format_res
        self.FPS = fps
        self.LEVEL = level
        self.bg_color = bg_color
        self.line_color = line_color
        self.TEMP_VIDEO = "temp_video.mp4"
        self.FINAL_VIDEO = "final_video_with_audio.mp4"
        
        # Parametri basati sul livello
        level_params = {
            "soft": {"density": 20, "sensitivity": 0.6, "thickness": 2},
            "medium": {"density": 35, "sensitivity": 0.8, "thickness": 3},
            "hard": {"density": 50, "sensitivity": 1.0, "thickness": 4}
        }
        self.params = level_params.get(level, level_params["medium"])
        self.MODE = mode

    def generate_barcode_frame(self, features: dict, frame_idx: int, beat_intensity: float) -> np.ndarray:
        """Genera frame barcode - linee verticali che rappresentano le frequenze"""
        frame = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        frame[:] = self.bg_color
        
        # Calcola time index
        time_idx = min(frame_idx, features['mel_spectrogram'].shape[1] - 1)
        
        # Parametri barcode
        num_bars = min(self.params['density'], self.WIDTH // 4)
        bar_width = max(1, self.WIDTH // num_bars - 1)
        
        for i in range(num_bars):
            # Mappa su frequenze mel
            freq_idx = int((i / num_bars) * features['mel_spectrogram'].shape[0])
            freq_idx = min(freq_idx, features['mel_spectrogram'].shape[0] - 1)
            
            # Energia per questa frequenza
            energy = features['mel_spectrogram'][freq_idx, time_idx]
            energy *= beat_intensity * self.params['sensitivity']
            
            # Calcola altezza barra
            bar_height = int(energy * self.HEIGHT * 0.8)
            if bar_height < 5:
                continue
                
            # Posizione
            x = int(i * (self.WIDTH / num_bars))
            y_center = self.HEIGHT // 2
            y_start = y_center - bar_height // 2
            y_end = y_center + bar_height // 2
            
            # Disegna barra
            cv2.rectangle(frame, (x, y_start), (x + bar_width, y_end), 
                         self.line_color, -1)
        
        return frame

    def generate_lines_frame(self, features: dict, frame_idx: int, beat_intensity: float) -> np.ndarray:
        """Genera frame linee - onde che seguono l'energia audio"""
        frame = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        frame[:] = self.bg_color
        
        # Calcola time index
        time_idx = min(frame_idx, len(features['rms_energy']) - 1)
        
        # Numero di linee basato sul livello
        num_lines = self.params['density'] // 5
        
        for line_idx in range(num_lines):
            # Offset temporale per ogni linea
            offset = line_idx * 5
            current_idx = max(0, time_idx - offset)
            
            # Energia per questa linea
            energy = features['rms_energy'][current_idx] if current_idx < len(features['rms_energy']) else 0
            energy *= beat_intensity * self.params['sensitivity']
            
            # Calcola punti della linea ondulata
            points = []
            amplitude = int(energy * self.HEIGHT * 0.3)
            
            for x in range(0, self.WIDTH, 8):
                # Onda sinusoidale modulata dall'energia
                wave = np.sin((x / self.WIDTH) * 4 * np.pi + line_idx) * amplitude
                y = int(self.HEIGHT // 2 + wave * (1 - line_idx * 0.1))
                points.append((x, y))
            
            # Disegna linea spezzata
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(frame, points[i], points[i + 1], 
                            self.line_color, self.params['thickness'])
        
        return frame

    def generate_spectrum_frame(self, features: dict, frame_idx: int, beat_intensity: float) -> np.ndarray:
        """Genera frame spectrum - analizzatore di spettro classico"""
        frame = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        frame[:] = self.bg_color
        
        # Calcola time index
        time_idx = min(frame_idx, features['stft_magnitude'].shape[1] - 1)
        
        # Numero di bande di frequenza
        num_bands = min(self.WIDTH // 8, features['stft_magnitude'].shape[0] // 2)
        band_width = self.WIDTH // num_bands
        
        for i in range(num_bands):
            # Mappa su frequenze STFT (solo metà inferiore, più interessante)
            freq_start = int((i / num_bands) * features['stft_magnitude'].shape[0] // 2)
            freq_end = int(((i + 1) / num_bands) * features['stft_magnitude'].shape[0] // 2)
            
            # Media energia per questa banda
            band_energy = np.mean(features['stft_magnitude'][freq_start:freq_end, time_idx])
            band_energy *= beat_intensity * self.params['sensitivity']
            
            # Altezza barra
            bar_height = int(band_energy * self.HEIGHT * 0.9)
            if bar_height < 2:
                continue
            
            # Posizione (dal basso verso l'alto)
            x_start = i * band_width
            x_end = x_start + band_width - 1
            y_start = self.HEIGHT - bar_height
            y_end = self.HEIGHT
            
            # Colore gradiente basato sulla frequenza
            color_intensity = min(255, int(band_energy * 255))
            if i < num_bands // 3:  # Bassi - più rosso
                color = (0, 0, color_intensity)
            elif i < 2 * num_bands // 3:  # Medi - più verde
                color = (0, color_intensity, 0)
            else:  # Alti - più blu
                color = (color_intensity, 0, 0)
            
            # Usa colore personalizzato se non gradiente
            color = self.line_color
            
            # Disegna barra
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, -1)
        
        return frame

    def calculate_beat_intensity(self, frame_idx: int, features: dict) -> float:
        """Calcola intensità basata sui beat rilevati"""
        try:
            beats = features.get('beats', [])
            if len(beats) == 0:
                return 1.0
            
            # Converti frame in tempo
            hop_length = features.get('hop_length', 512)
            sr = features.get('sr', 22050)
            current_time = frame_idx * (hop_length / sr)
            
            # Calcola tempi dei beat
            beat_times = np.array(beats) * (hop_length / sr)
            
            # Trova beat più vicino
            if len(beat_times) > 0:
                distances = np.abs(beat_times - current_time)
                closest_beat_dist = float(np.min(distances))
            else:
                closest_beat_dist = 1.0
            
            # Intensità inversamente proporzionale alla distanza dal beat
            beat_window = 0.1  # 100ms window
            if closest_beat_dist < beat_window:
                intensity = 1.0 + 0.5 * (1 - closest_beat_dist / beat_window)
            else:
                intensity = 1.0
            
            return min(1.5, intensity)
            
        except Exception as e:
            print(f"Beat intensity calculation error: {e}")
            return 1.0

    def generate_video(self, audio_features: dict, audio_file_path: str) -> bool:
        """Genera video con audio sincronizzato"""
        video_writer = None
        try:
            cleanup_files(self.TEMP_VIDEO, self.FINAL_VIDEO)
            
            # Calcola parametri video
            duration = audio_features['duration']
            total_frames = int(duration * self.FPS)
            
            if total_frames <= 0:
                st.error("❌ Durata video non valida")
                return False
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(self.TEMP_VIDEO, fourcc, self.FPS, 
                                         (self.WIDTH, self.HEIGHT))
            
            if not video_writer.isOpened():
                st.error("❌ Impossibile creare video writer")
                return False
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Genera frames
            for frame_idx in range(total_frames):
                try:
                    # Calcola indice temporale per features audio
                    time_ratio = frame_idx / total_frames
                    
                    # Beat intensity
                    beat_intensity = self.calculate_beat_intensity(frame_idx, audio_features)
                    
                    # Genera frame basato sulla modalità
                    if self.MODE == "Barcode":
                        frame = self.generate_barcode_frame(audio_features, 
                                                          int(time_ratio * audio_features['mel_spectrogram'].shape[1]), 
                                                          beat_intensity)
                    elif self.MODE == "Lines":
                        frame = self.generate_lines_frame(audio_features, 
                                                        int(time_ratio * len(audio_features['rms_energy'])), 
                                                        beat_intensity)
                    elif self.MODE == "Spectrum":
                        frame = self.generate_spectrum_frame(audio_features, 
                                                           int(time_ratio * audio_features['stft_magnitude'].shape[1]), 
                                                           beat_intensity)
                    else:
                        frame = np.ones((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8) * 128
                    
                    video_writer.write(frame)
                    
                    # Update progress
                    if frame_idx % 10 == 0:
                        progress = (frame_idx + 1) / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"Generazione: {frame_idx+1}/{total_frames} frame ({progress*100:.1f}%)")
                
                except Exception as e:
                    st.error(f"❌ Errore frame {frame_idx}: {e}")
                    return False
            
            video_writer.release()
            video_writer = None
            
            # Combina video con audio usando FFmpeg
            status_text.text("🔊 Aggiunta audio al video...")
            
            if not self._add_audio_to_video(self.TEMP_VIDEO, audio_file_path, self.FINAL_VIDEO):
                st.error("❌ Errore nell'aggiunta dell'audio")
                return False
            
            status_text.text("✅ Video completato!")
            return True
            
        except Exception as e:
            st.error(f"❌ Errore generazione video: {e}")
            return False
        finally:
            if video_writer:
                video_writer.release()

    def _add_audio_to_video(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Combina video e audio usando FFmpeg"""
        try:
            cmd = [
                'ffmpeg', '-y',  # Sovrascrivi file esistente
                '-i', video_path,  # Video input
                '-i', audio_path,  # Audio input
                '-c:v', 'libx264',  # Codec video
                '-c:a', 'aac',      # Codec audio
                '-strict', 'experimental',
                '-shortest',        # Taglia al più corto
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                st.error(f"FFmpeg error: {result.stderr}")
                return False
                
            return os.path.exists(output_path)
            
        except subprocess.TimeoutExpired:
            st.error("❌ Timeout durante la combinazione audio/video")
            return False
        except Exception as e:
            st.error(f"❌ Errore FFmpeg: {e}")
            return False

def main():
    st.set_page_config(page_title="🎵 AudioLine by Loop507", layout="centered")
    st.title("🎵 AudioLine by Loop507")
    st.markdown("*Trasforma la tua musica in visualizzazioni sincronizzate*")
    
    # Verifica FFmpeg
    if not check_ffmpeg():
        st.error("❌ **FFmpeg non trovato!** Installa FFmpeg per continuare.")
        st.markdown("**Come installare FFmpeg:**")
        st.code("# Ubuntu/Debian:\nsudo apt install ffmpeg\n\n# macOS:\nbrew install ffmpeg\n\n# Windows:\n# Scarica da https://ffmpeg.org/download.html")
        return
    
    uploaded = st.file_uploader("🎧 Carica il tuo file audio", type=["wav", "mp3"])
    
    if uploaded:
        if not validate_audio_file(uploaded):
            return
        
        # Salva file temporaneo
        temp_audio = f"temp_audio_{uploaded.name}"
        try:
            with open(temp_audio, "wb") as f:
                f.write(uploaded.read())
            
            # Processa audio
            with st.spinner("🔄 Caricamento audio..."):
                y, sr, duration = load_and_process_audio(temp_audio)
                
            if y is None:
                return
            
            # Estrai features audio
            with st.spinner("📊 Analisi audio completa..."):
                features = generate_audio_features(y, sr, fps=30)  # Default FPS per analisi
                
            if features is None:
                return
            
            # Info audio
            tempo_val = float(features['tempo']) if hasattr(features['tempo'], 'item') else features['tempo']
            st.success(f"✅ **Audio elaborato:** {duration:.1f}s | **BPM:** {tempo_val:.1f} | **SR:** {sr}Hz")
            
            # Libera memoria
            del y
            gc.collect()
            
            # Interfaccia controlli
            st.markdown("### ⚙️ Configurazione Video")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                fmt_key = st.selectbox("📐 **Formato**", list(FORMAT_RESOLUTIONS.keys()), index=0)
                format_res = FORMAT_RESOLUTIONS[fmt_key]
                st.caption(f"📏 {format_res[0]}×{format_res[1]} px")
            
            with col2:
                level = st.selectbox("🎨 **Intensità**", ["soft", "medium", "hard"], index=1)
                level_desc = {"soft": "Delicato", "medium": "Bilanciato", "hard": "Intenso"}
                st.caption(f"⚡ {level_desc[level]}")
            
            with col3:
                fps = st.selectbox("🎞 **FPS**", [15, 20, 24, 30, 60], index=3)
                st.caption(f"⏱️ {fps} frame/sec")
            
            # Controlli colore
            col4, col5 = st.columns(2)
            with col4:
                bg = st.color_picker("🎨 **Sfondo**", "#000000")
                st.caption("🖤 Colore di fondo")
            with col5:
                line = st.color_picker("🎨 **Elementi**", "#00FF41")
                st.caption("💚 Colore visualizzazione")
            
            # Modalità visualizzazione
            mode = st.selectbox("✨ **Modalità Visualizzazione**", 
                               ["Barcode", "Lines", "Spectrum"], index=0)
            
            mode_desc = {
                "Barcode": "📊 Linee verticali per frequenze",
                "Lines": "〰️ Onde sincronizzate con energia",
                "Spectrum": "📈 Analizzatore di spettro classico"
            }
            st.caption(mode_desc[mode])
            
            # Generazione video
            st.markdown("---")
            if st.button("🎬 **Genera Video**", type="primary", use_container_width=True):
                
                # Rigenera features con FPS corretto
                with st.spinner("🔄 Ottimizzazione per FPS selezionato..."):
                    features = generate_audio_features(y if 'y' in locals() else librosa.load(temp_audio)[0], 
                                                     sr, fps)
                
                with st.spinner("🎬 Generazione video in corso..."):
                    generator = VideoGenerator(
                        format_res, level, fps, 
                        hex_to_bgr(bg), hex_to_bgr(line), 
                        mode
                    )
                    
                    if generator.generate_video(features, temp_audio):
                        st.balloons()
                        st.success("🎉 **Video generato con successo!**")
                        
                        # Mostra info file
                        if os.path.exists("final_video_with_audio.mp4"):
                            file_size = os.path.getsize("final_video_with_audio.mp4") / 1024 / 1024
                            st.info(f"📁 **Dimensione file:** {file_size:.1f} MB")
                            
                            # Download button
                            with open("final_video_with_audio.mp4", "rb") as video_file:
                                st.download_button(
                                    label="⬇️ **Scarica Video Completo**",
                                    data=video_file,
                                    file_name=f"audioline_{mode.lower()}_{fmt_key}_{fps}fps.mp4",
                                    mime="video/mp4",
                                    type="primary",
                                    use_container_width=True
                                )
                        else:
                            st.error("❌ File video non trovato dopo la generazione")
                    
                    # Cleanup
                    if 'features' in locals():
                        del features
                    gc.collect()
        
        except Exception as e:
            st.error(f"❌ Errore generale: {e}")
        finally:
            # Pulizia garantita
            cleanup_files(temp_audio, "temp_video.mp4", "final_video_with_audio.mp4")

    # Footer
    st.markdown("---")
    st.markdown("🎵 **AudioLine by Loop507** - Trasforma la musica in arte visiva")
    st.caption("Supporta file MP3/WAV fino a 200MB e 5 minuti")

if __name__ == "__main__":
    main()
