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

def analyze_audio_features(y: np.ndarray, sr: int, fps: int, duration: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Analizza volume, tempo e spettrogramma per la sincronizzazione"""
    try:
        # Genera melspettrogramma
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr / 2)
        if S.size == 0:
            st.error("❌ Impossibile generare lo spettrogramma: l'audio è troppo breve o non è valido.")
            return None, None, None
        mel_spec_db = librosa.power_to_db(S, ref=np.max)
        min_val = mel_spec_db.min()
        max_val = mel_spec_db.max()
        if max_val == min_val:
            st.error("❌ L'audio non contiene variazioni sufficienti per generare il video.")
            return None, None, None
        mel_spec_norm = (mel_spec_db - min_val) / (max_val - min_val)
        
        # Analisi del volume (RMS energy)
        hop_length = 512
        frame_length = 2048
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Normalizza RMS
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
        
        # Analisi del tempo musicale (onset strength)
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_norm = (onset_strength - onset_strength.min()) / (onset_strength.max() - onset_strength.min() + 1e-8)
        
        # Sincronizza con i frame del video
        total_frames = int(duration * fps)
        
        # Interpola RMS per matchare i frame del video
        rms_frames = np.interp(
            np.linspace(0, len(rms_norm)-1, total_frames),
            np.arange(len(rms_norm)),
            rms_norm
        )
        
        # Interpola onset strength per matchare i frame del video
        onset_frames = np.interp(
            np.linspace(0, len(onset_norm)-1, total_frames),
            np.arange(len(onset_norm)),
            onset_norm
        )
        
        if mel_spec_norm.shape[1] == 0:
            st.error("❌ Lo spettrogramma è vuoto: l'audio è troppo breve o non contiene dati validi.")
            return None, None, None
        
        return mel_spec_norm, rms_frames, onset_frames
        
    except Exception as e:
        st.error(f"❌ Errore nell'analisi audio: {str(e)}")
        return None, None, None

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0])

class VideoGenerator:
    def __init__(self, format_res, level, fps=30, bg_color=(255, 255, 255), freq_colors=None, effect_mode="connessioni"):
        self.W, self.H = format_res
        self.FPS = fps
        self.LEVEL = level  # "soft", "medium", "hard"
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

    def get_mood_factor(self):
        return {"soft": 0.5, "medium": 1.0, "hard": 1.5}.get(self.LEVEL, 1.0)

    def freq_to_color(self, i):
        if i < 42:
            return self.freq_colors['low']
        elif i < 85:
            return self.freq_colors['mid']
        return self.freq_colors['high']

    def draw_connected_lines(self, frame, mel, idx, volume_factor, tempo_factor):
        mood = self.get_mood_factor()
        
        # Adatta il numero di punti al volume (più volume = più punti)
        base_pts = int(15 * mood)
        volume_pts = max(5, int(base_pts * (0.3 + 0.7 * volume_factor)))
        
        # Adatta la densità delle connessioni al tempo
        connection_probability = 0.3 + (0.4 * tempo_factor)
        
        pts = [(np.random.randint(0, self.W), np.random.randint(0, self.H)) for _ in range(volume_pts)]
        
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                # Probabilità di connessione basata sul tempo musicale
                if np.random.random() < connection_probability:
                    val = mel[np.random.randint(0, mel.shape[0]), idx]
                    threshold = 0.3 * mood * (1.0 - volume_factor * 0.3)  # Soglia dinamica basata sul volume
                    
                    if val > threshold:
                        color = self.freq_to_color(np.random.randint(0, mel.shape[0]))
                        # Spessore basato su volume e intensità spettrale
                        thick = max(1, int((1 + val * 4) * mood * (0.5 + 0.5 * volume_factor)))
                        cv2.line(frame, pts[i], pts[j], color, thick)

    def draw_rectangular_grid(self, frame, mel, idx, volume_factor, tempo_factor):
        mood = self.get_mood_factor()
        
        # Adatta la griglia al volume
        base_size = 5
        grid_size = max(3, int(base_size * (0.6 + 0.4 * volume_factor)))
        rows, cols = grid_size, grid_size
        
        margin_x = self.W // (cols + 1)
        margin_y = self.H // (rows + 1)
        pts = []
        
        # Jitter basato sul tempo musicale
        jitter_scale = int((margin_x // 3) * mood * (0.5 + 0.5 * tempo_factor))
        
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                jitter_x = np.random.randint(-jitter_scale, jitter_scale)
                jitter_y = np.random.randint(-jitter_scale, jitter_scale)
                x = c * margin_x + jitter_x
                y = r * margin_y + jitter_y
                pts.append((x, y))

        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dist = np.linalg.norm(np.array(pts[i]) - np.array(pts[j]))
                max_dist = max(margin_x, margin_y) * 1.5
                
                if dist < max_dist:
                    val = mel[np.random.randint(0, mel.shape[0]), idx]
                    # Soglia dinamica basata su volume e tempo
                    threshold = 0.1 * mood * (1.2 - volume_factor * 0.4 - tempo_factor * 0.2)
                    
                    if val > threshold:
                        color = self.freq_to_color(np.random.randint(0, mel.shape[0]))
                        # Spessore basato su volume, tempo e intensità
                        thickness = max(1, int(val * 5 * mood * (0.4 + 0.3 * volume_factor + 0.3 * tempo_factor)))
                        cv2.line(frame, pts[i], pts[j], color, thickness)

    def draw_complex_geometric_network(self, frame, mel, idx, volume_factor, tempo_factor):
        mood = self.get_mood_factor()
        
        # Numero di punti basato sul volume
        base_pts = int(25 * mood)
        volume_pts = max(10, int(base_pts * (0.4 + 0.6 * volume_factor)))
        
        pts = [(np.random.randint(0, self.W), np.random.randint(0, self.H)) for _ in range(volume_pts)]
        
        # Probabilità di connessione basata sul tempo
        connection_prob = 0.2 + (0.3 * tempo_factor)
        
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if np.random.random() < connection_prob:
                    val = mel[np.random.randint(0, mel.shape[0]), idx]
                    # Soglia adattiva
                    threshold = 0.2 * mood * (1.1 - volume_factor * 0.3)
                    
                    if val > threshold:
                        color = self.freq_to_color(np.random.randint(0, mel.shape[0]))
                        # Spessore che combina tutti i fattori
                        thickness = max(1, int(val * 6 * mood * (0.3 + 0.4 * volume_factor + 0.3 * tempo_factor)))
                        cv2.line(frame, pts[i], pts[j], color, thickness)

    def generate_video(self, mel, rms_frames, onset_frames, duration, sync_audio=True):
        for f in [self.TEMP, self.FINAL]:
            if os.path.exists(f): 
                os.remove(f)
                
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
                
                # Calcola indici per mel spectrogram
                t_idx = int((i / total_frames) * mel.shape[1])
                t_idx = min(t_idx, mel.shape[1] - 1)  # Assicura che l'indice sia valido
                
                # Ottieni fattori di volume e tempo per questo frame
                volume_factor = rms_frames[min(i, len(rms_frames) - 1)]
                tempo_factor = onset_frames[min(i, len(onset_frames) - 1)]
                
                # Applica smoothing per evitare cambi troppo bruschi
                if i > 0:
                    prev_vol = rms_frames[min(i-1, len(rms_frames) - 1)]
                    prev_tempo = onset_frames[min(i-1, len(onset_frames) - 1)]
                    volume_factor = 0.7 * volume_factor + 0.3 * prev_vol
                    tempo_factor = 0.7 * tempo_factor + 0.3 * prev_tempo
                
                # Genera gli effetti visuali sincronizzati
                if self.effect_mode == "connessioni":
                    self.draw_connected_lines(frame, mel, t_idx, volume_factor, tempo_factor)
                elif self.effect_mode == "rettangoli":
                    self.draw_rectangular_grid(frame, mel, t_idx, volume_factor, tempo_factor)
                elif self.effect_mode == "geometriche":
                    self.draw_complex_geometric_network(frame, mel, t_idx, volume_factor, tempo_factor)
                
                writer.write(frame)
                
                if i % 10 == 0:
                    progress.progress((i + 1) / total_frames)
                    status.text(f"🎬 Frame {i + 1}/{total_frames} (Vol: {volume_factor:.2f}, Tempo: {tempo_factor:.2f})")
                    
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
    FORMAT_RESOLUTIONS = {
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "1:1": (720, 720),
        "4:3": (800, 600)
    }

    st.set_page_config(page_title="🎵 AudioLinee - by Loop507", layout="centered")
    st.title("🎨 AudioLinee")
    st.markdown("### by Loop507")
    st.markdown("Carica un file audio e genera un video visivo sincronizzato con volume e tempo musicale.")

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

        # Selezione parametri video prima dell'analisi
        col1, col2, col3 = st.columns(3)
        with col1:
            video_format = st.selectbox("📐 Formato video", list(FORMAT_RESOLUTIONS.keys()))
        with col2:
            effect_level = st.selectbox("🎨 Livello effetti", ["soft", "medium", "hard"])
        with col3:
            fps_choice = st.selectbox("🎞️ Fotogrammi al secondo (FPS)", [5, 10, 15, 24, 30], index=3)

        with st.spinner("📊 Analisi audio avanzata in corso (volume, tempo, spettrogramma)..."):
            mel_spec_norm, rms_frames, onset_frames = analyze_audio_features(y, sr, fps_choice, audio_duration)
        
        if mel_spec_norm is None or rms_frames is None or onset_frames is None:
            return
        
        st.success("✅ Analisi audio completata con sincronizzazione volume/tempo!")
        
        # Mostra statistiche audio
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.metric("📊 Volume medio", f"{np.mean(rms_frames):.2f}")
            st.metric("🎵 Dinamica volume", f"{np.std(rms_frames):.2f}")
        with col_stats2:
            st.metric("⚡ Energia tempo media", f"{np.mean(onset_frames):.2f}")
            st.metric("🎼 Variazioni tempo", f"{np.std(onset_frames):.2f}")

        effect_mode = st.selectbox("✨ Effetto artistico", ["connessioni", "rettangoli", "geometriche"])

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

        if st.button("🎬 Genera Video Sincronizzato"):
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
            
            success = generator.generate_video(mel_spec_norm, rms_frames, onset_frames, audio_duration, sync_audio)
            
            if success and os.path.exists(generator.FINAL):
                with open(generator.FINAL, "rb") as f:
                    st.download_button("⬇️ Scarica il video", f, file_name=f"audio_linee_sync_{video_format}_{effect_level}_{effect_mode}.mp4", mime="video/mp4")
                file_size = os.path.getsize(generator.FINAL)
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
