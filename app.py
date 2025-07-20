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
    """Verifica se FFmpeg è installato e disponibile nel PATH."""
    return shutil.which("ffmpeg") is not None

def validate_audio_file(uploaded_file) -> bool:
    """Valida le dimensioni del file audio caricato."""
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"❌ File troppo grande ({uploaded_file.size / 1024 / 1024:.1f}MB). Limite: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
        return False
    return True

def load_and_process_audio(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
    """
    Carica e processa il file audio, gestendo la durata e i potenziali errori.
    """
    try:
        # Carica l'audio con librosa, mantenendo il sample rate originale e convertendo a mono
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
            audio_duration = MAX_DURATION # Aggiorna la durata dopo il troncamento
            
        return y, sr, audio_duration
    except Exception as e:
        st.error(f"❌ Errore nel caricamento dell'audio: {str(e)}")
        return None, None, None

def analyze_audio_features(y: np.ndarray, sr: int, fps: int, duration: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Analizza il volume (RMS), il tempo musicale (onset strength) e genera lo spettrogramma
    per la sincronizzazione visiva.
    """
    try:
        # Genera melspettrogramma per rappresentare le frequenze
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr / 2)
        if S.size == 0:
            st.error("❌ Impossibile generare lo spettrogramma: l'audio è troppo breve o non è valido.")
            return None, None, None
        
        # Converte lo spettrogramma in scala di decibel e normalizza tra 0 e 1
        mel_spec_db = librosa.power_to_db(S, ref=np.max)
        min_val = mel_spec_db.min()
        max_val = mel_spec_db.max()
        if max_val == min_val: # Gestisce il caso di audio piatto
            st.error("❌ L'audio non contiene variazioni sufficienti per generare il video.")
            return None, None, None
        mel_spec_norm = (mel_spec_db - min_val) / (max_val - min_val)
        
        # Analisi del volume (RMS energy) su finestre brevi
        hop_length = 512 # Numero di campioni tra frame consecutivi di analisi
        frame_length = 2048 # Dimensione della finestra per l'analisi RMS
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Normalizza RMS tra 0 e 1
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8) # Aggiunge epsilon per evitare divisione per zero
        
        # Analisi del tempo musicale (onset strength) per rilevare gli attacchi
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onset_norm = (onset_strength - onset_strength.min()) / (onset_strength.max() - onset_strength.min() + 1e-8)
        
        # Sincronizza i dati audio con il numero totale di frame del video
        total_frames = int(duration * fps)
        
        # Interpola RMS per avere un valore per ogni frame del video
        rms_frames = np.interp(
            np.linspace(0, len(rms_norm)-1, total_frames), # Punti target (frame video)
            np.arange(len(rms_norm)), # Punti sorgente (campioni RMS)
            rms_norm # Valori sorgente
        )
        
        # Interpola onset strength per avere un valore per ogni frame del video
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
    """Converte un colore esadecimale (es. '#RRGGBB') in un tuple BGR (Blue, Green, Red)."""
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0]) # OpenCV usa BGR

class VideoGenerator:
    """
    Classe per generare effetti visivi sincronizzati con l'audio.
    """
    def __init__(self, format_res: Tuple[int, int], level: str, fps: int = 30, 
                 bg_color: Tuple[int, int, int] = (255, 255, 255), 
                 freq_colors: Optional[dict] = None, effect_mode: str = "connessioni"):
        self.W, self.H = format_res # Larghezza e Altezza del video
        self.FPS = fps
        self.LEVEL = level  # "soft", "medium", "hard" per modulare l'intensità degli effetti
        self.bg_color = bg_color
        self.freq_colors = freq_colors or { # Colori di default per le bande di frequenza
            'low': (0, 0, 0),     # nero (BGR)
            'mid': (0, 0, 255),   # rosso (BGR)
            'high': (255, 0, 0)   # blu (BGR)
        }
        self.effect_mode = effect_mode # Modalità dell'effetto grafico
        self.TEMP = "temp_output.mp4" # File temporaneo per il video senza audio
        self.FINAL = "final_output.mp4" # File finale con audio sincronizzato

    def get_mood_factor(self) -> float:
        """Restituisce un fattore di moltiplicazione basato sul livello di difficoltà/intensità scelto."""
        return {"soft": 0.5, "medium": 1.0, "hard": 1.5}.get(self.LEVEL, 1.0)

    def freq_to_color(self, i: int) -> Tuple[int, int, int]:
        """Mappa un indice di frequenza a un colore predefinito."""
        # Questi indici sono basati su n_mels=128, dividendo approssimativamente in 3 fasce
        if i < 42: # Basse frequenze
            return self.freq_colors['low']
        elif i < 85: # Medie frequenze
            return self.freq_colors['mid']
        return self.freq_colors['high'] # Alte frequenze

    def draw_connected_lines(self, frame: np.ndarray, mel: np.ndarray, idx: int, volume_factor: float, tempo_factor: float):
        """Disegna un effetto di linee connesse basato sull'audio."""
        mood = self.get_mood_factor()
        
        # Numero di punti base, influenzato dal mood e altamente dal volume
        base_pts = int(8 * mood) 
        volume_pts = max(3, int(base_pts * (0.4 + 1.2 * volume_factor))) # Maggiore sensibilità al volume
        
        # Probabilità di connessione, influenzata dal tempo musicale
        connection_probability = 0.15 + (0.7 * tempo_factor * mood) 
        
        pts = [(np.random.randint(0, self.W), np.random.randint(0, self.H)) for _ in range(volume_pts)]
        
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if np.random.random() < connection_probability:
                    # Preleva un valore dallo spettrogramma (random per varietà)
                    val = mel[np.random.randint(0, mel.shape[0]), idx]
                    
                    # Soglia dinamica: più alta di base, ma abbassata da volume e tempo elevati
                    threshold = 0.25 * mood * (1.2 - volume_factor * 0.6 - tempo_factor * 0.2) 
                    
                    if val > threshold:
                        color = self.freq_to_color(np.random.randint(0, mel.shape[0]))
                        # Spessore della linea: molto dipendente dal volume e dal valore spettrale
                        thick = max(1, int((0.5 + val * 6) * mood * (0.2 + 0.8 * volume_factor)))
                        cv2.line(frame, pts[i], pts[j], color, thick)

    def draw_rectangular_grid(self, frame: np.ndarray, mel: np.ndarray, idx: int, volume_factor: float, tempo_factor: float):
        """Disegna un effetto a griglia rettangolare basato sull'audio."""
        mood = self.get_mood_factor()
        
        # Dimensione della griglia, influenzata dal volume
        base_size = 4
        grid_density = max(2, int(base_size * (0.5 + 0.5 * volume_factor))) # Più volume = griglia più densa
        rows, cols = grid_density, grid_density
        
        margin_x = self.W // (cols + 1)
        margin_y = self.H // (rows + 1)
        pts = []
        
        # Jitter (spostamento casuale dei punti) basato sul tempo musicale
        jitter_scale = int((margin_x // 4) * mood * (0.3 + 0.7 * tempo_factor)) # Più tempo = più jitter
        
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
                max_dist_threshold = max(margin_x, margin_y) * 1.8 # Distanza massima per connettere i punti
                
                if dist < max_dist_threshold:
                    val = mel[np.random.randint(0, mel.shape[0]), idx]
                    # Soglia dinamica: rende le connessioni più difficili a basso volume/tempo
                    threshold = 0.15 * mood * (1.3 - volume_factor * 0.5 - tempo_factor * 0.3)
                    
                    if val > threshold:
                        color = self.freq_to_color(np.random.randint(0, mel.shape[0]))
                        # Spessore basato su volume, tempo e intensità spettrale
                        thickness = max(1, int(val * 4 * mood * (0.3 + 0.4 * volume_factor + 0.3 * tempo_factor)))
                        cv2.line(frame, pts[i], pts[j], color, thickness)

    def draw_complex_geometric_network(self, frame: np.ndarray, mel: np.ndarray, idx: int, volume_factor: float, tempo_factor: float):
        """Disegna un effetto di rete geometrica complessa basata sull'audio."""
        mood = self.get_mood_factor()
        
        # Numero di punti basato sul volume, con una base più conservativa
        base_pts = int(20 * mood) 
        volume_pts = max(8, int(base_pts * (0.3 + 0.8 * volume_factor))) # Forte influenza del volume
        
        pts = [(np.random.randint(0, self.W), np.random.randint(0, self.H)) for _ in range(volume_pts)]
        
        # Probabilità di connessione basata sul tempo
        connection_prob = 0.18 + (0.4 * tempo_factor * mood) # La probabilità aumenta con il tempo
        
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if np.random.random() < connection_prob:
                    val = mel[np.random.randint(0, mel.shape[0]), idx]
                    # Soglia adattiva: influenza maggiore di volume e tempo
                    threshold = 0.22 * mood * (1.15 - volume_factor * 0.4 - tempo_factor * 0.25)
                    
                    if val > threshold:
                        color = self.freq_to_color(np.random.randint(0, mel.shape[0]))
                        # Spessore che combina tutti i fattori, con forte peso sul volume e spettro
                        thickness = max(1, int(val * 5 * mood * (0.2 + 0.5 * volume_factor + 0.3 * tempo_factor)))
                        cv2.line(frame, pts[i], pts[j], color, thickness)

    def draw_horizontal_lines(self, frame: np.ndarray, mel: np.ndarray, idx: int, volume_factor: float, tempo_factor: float):
        """Disegna un effetto di linee orizzontali pulsanti basato sull'audio."""
        mood = self.get_mood_factor()
        
        # Numero di linee proporzionale al volume e alla difficoltà
        num_lines = int(5 + 20 * volume_factor * mood) 
        
        for i in range(num_lines):
            # Calcola posizione Y della linea con un piccolo jitter
            y_base = int((i / num_lines) * self.H)
            
            # Movimento verticale dinamico basato sul tempo e sull'indice del frame
            # Usiamo sin per un movimento ondulatorio
            y_offset = int(20 * tempo_factor * mood * np.sin(idx * 0.1 + i * 0.5)) 
            y = y_base + y_offset
            y = np.clip(y, 0, self.H - 1) # Assicura che la linea rimanga nel frame
            
            # Preleva valore spettro per questa linea. Per semplicità, usiamo un indice randomico del mel.
            # Potresti volere mappare l'indice 'i' a una specifica banda di frequenza qui.
            spec_val_idx = np.random.randint(0, mel.shape[0])
            spec_val = mel[spec_val_idx, idx] 
            
            # Colore in base alla frequenza e alla sua intensità
            line_color_base = self.freq_to_color(spec_val_idx)
            
            # Regola l'intensità del colore (simulando opacità/luminosità)
            # Più volume e valore spettrale = più intensa
            intensity_factor = 0.5 + 0.5 * spec_val * volume_factor 
            color_bgr = (int(line_color_base[0] * intensity_factor), 
                         int(line_color_base[1] * intensity_factor), 
                         int(line_color_base[2] * intensity_factor))
            
            # Spessore che combina tutti i fattori
            thickness = max(1, int(2 + spec_val * 5 * mood * volume_factor))
            
            cv2.line(frame, (0, y), (self.W, y), color_bgr, thickness)

    def draw_vertical_lines(self, frame: np.ndarray, mel: np.ndarray, idx: int, volume_factor: float, tempo_factor: float):
        """Disegna un effetto di linee verticali pulsanti basato sull'audio."""
        mood = self.get_mood_factor()
        
        # Numero di linee proporzionale al volume e alla difficoltà
        num_lines = int(5 + 20 * volume_factor * mood) 
        
        for i in range(num_lines):
            # Calcola posizione X della linea con un piccolo jitter
            x_base = int((i / num_lines) * self.W)
            
            # Movimento orizzontale dinamico basato sul tempo e sull'indice del frame
            x_offset = int(20 * tempo_factor * mood * np.sin(idx * 0.1 + i * 0.5)) 
            x = x_base + x_offset
            x = np.clip(x, 0, self.W - 1) # Assicura che la linea rimanga nel frame
            
            # Preleva valore spettro per questa linea (usiamo un indice randomico del mel per varietà)
            spec_val_idx = np.random.randint(0, mel.shape[0])
            spec_val = mel[spec_val_idx, idx] 
            
            # Colore in base alla frequenza e alla sua intensità
            line_color_base = self.freq_to_color(spec_val_idx)
            
            # Regola l'intensità del colore (simulando opacità/luminosità)
            intensity_factor = 0.5 + 0.5 * spec_val * volume_factor 
            color_bgr = (int(line_color_base[0] * intensity_factor), 
                         int(line_color_base[1] * intensity_factor), 
                         int(line_color_base[2] * intensity_factor))
            
            # Spessore che combina tutti i fattori
            thickness = max(1, int(2 + spec_val * 5 * mood * volume_factor))
            
            cv2.line(frame, (x, 0), (x, self.H), color_bgr, thickness)


    def generate_video(self, mel: np.ndarray, rms_frames: np.ndarray, onset_frames: np.ndarray, duration: float, sync_audio: bool = True) -> bool:
        """
        Genera il video frame per frame, applicando gli effetti visivi sincronizzati
        e opzionalmente unendo l'audio alla fine.
        """
        # Pulisce i file temporanei precedenti
        for f in [self.TEMP, self.FINAL]:
            if os.path.exists(f): 
                os.remove(f)
                
        total_frames = int(duration * self.FPS)
        # Inizializza il VideoWriter di OpenCV
        writer = cv2.VideoWriter(self.TEMP, cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (self.W, self.H))
        
        if not writer.isOpened():
            st.error("❌ Impossibile inizializzare il writer video.")
            return False
            
        progress = st.progress(0) # Barra di progresso per Streamlit
        status = st.empty() # Placeholder per messaggi di stato
        
        # Loop per generare ogni frame del video
        for i in range(total_frames):
            try:
                # Inizializza il frame con il colore di sfondo
                frame = np.ones((self.H, self.W, 3), dtype=np.uint8) * np.array(self.bg_color, dtype=np.uint8)
                
                # Calcola l'indice dello spettrogramma corrispondente al frame corrente
                t_idx = int((i / total_frames) * mel.shape[1])
                t_idx = min(t_idx, mel.shape[1] - 1)  # Assicura che l'indice sia valido
                
                # Ottieni i fattori di volume e tempo per questo frame
                current_volume_factor = rms_frames[min(i, len(rms_frames) - 1)]
                current_tempo_factor = onset_frames[min(i, len(onset_frames) - 1)]
                
                # --- Applicazione dello Smoothing per fluidificare i movimenti ---
                # Peso maggiore per il frame precedente per un effetto più smussato
                if i > 0:
                    prev_vol = rms_frames[min(i-1, len(rms_frames) - 1)]
                    prev_tempo = onset_frames[min(i-1, len(onset_frames) - 1)]
                    volume_factor = 0.5 * current_volume_factor + 0.5 * prev_vol
                    tempo_factor = 0.5 * current_tempo_factor + 0.5 * prev_tempo
                else: # Primo frame, nessun precedente per lo smoothing
                    volume_factor = current_volume_factor
                    tempo_factor = current_tempo_factor
                # --- Fine Smoothing ---
                
                # Genera gli effetti visuali in base alla modalità selezionata
                if self.effect_mode == "connessioni":
                    self.draw_connected_lines(frame, mel, t_idx, volume_factor, tempo_factor)
                elif self.effect_mode == "rettangoli":
                    self.draw_rectangular_grid(frame, mel, t_idx, volume_factor, tempo_factor)
                elif self.effect_mode == "geometriche":
                    self.draw_complex_geometric_network(frame, mel, t_idx, volume_factor, tempo_factor)
                elif self.effect_mode == "linee_orizzontali":
                    self.draw_horizontal_lines(frame, mel, t_idx, volume_factor, tempo_factor)
                elif self.effect_mode == "linee_verticali":
                    self.draw_vertical_lines(frame, mel, t_idx, volume_factor, tempo_factor)
                
                writer.write(frame) # Scrive il frame nel video
                
                # Aggiorna la barra di progresso e lo stato in Streamlit
                if i % 10 == 0 or i == total_frames - 1: # Aggiorna ogni 10 frame o all'ultimo
                    progress.progress((i + 1) / total_frames)
                    status.text(f"🎬 Generazione Frame {i + 1}/{total_frames} (Vol: {volume_factor:.2f}, Tempo: {tempo_factor:.2f})")
                    
            except Exception as e:
                st.warning(f"⚠️ Errore al frame {i}: {str(e)}")
                
        writer.release() # Rilascia il writer video
        
        # Sincronizzazione audio con FFmpeg
        if sync_audio and check_ffmpeg():
            try:
                status.text("🎶 Sincronizzazione audio in corso (potrebbe richiedere qualche istante)...")
                subprocess.run([
                    "ffmpeg", "-y", # Sovrascrivi il file di output se esiste
                    "-i", self.TEMP, # Input video (senza audio)
                    "-i", "input_audio.wav", # Input audio
                    "-c:v", "libx264", "-crf", "28", "-preset", "veryfast", # Codec video H.264, qualità, preset di velocità
                    "-c:a", "aac", "-b:a", "192k", "-shortest", # Codec audio AAC, bitrate, termina con il flusso più corto
                    self.FINAL
                ], capture_output=True, check=True) # Cattura output e solleva errore se il comando fallisce
                os.remove(self.TEMP) # Rimuovi il video temporaneo senza audio
            except subprocess.CalledProcessError as e:
                st.error(f"❌ Errore FFmpeg durante la sincronizzazione audio: {e.stderr.decode()}")
                os.rename(self.TEMP, self.FINAL) # Se FFmpeg fallisce, salva comunque il video senza audio
            except Exception as e:
                st.error(f"❌ Errore generico durante la sincronizzazione audio: {str(e)}")
                os.rename(self.TEMP, self.FINAL) # Se fallisce, salva comunque il video senza audio
        else:
            # Se FFmpeg non c'è o la sincronizzazione è disabilitata, rinomina il file temporaneo
            os.rename(self.TEMP, self.FINAL)
            
        status.text("✅ Video completato!")
        gc.collect() # Esegue il garbage collection per liberare memoria
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
            
        # Salva il file audio caricato localmente per librosa e FFmpeg
        with open("input_audio.wav", "wb") as f:
            f.write(uploaded_file.read())
        st.success("🔊 Audio caricato correttamente!")

        # Carica e processa l'audio
        y, sr, audio_duration = load_and_process_audio("input_audio.wav")
        if y is None: # Se il caricamento o la validazione falliscono
            return
        st.info(f"🔊 Durata audio: {audio_duration:.2f} secondi")

        # Selezione parametri video prima dell'analisi per influenzare l'interpolazione
        col1, col2, col3 = st.columns(3)
        with col1:
            video_format = st.selectbox("📐 Formato video", list(FORMAT_RESOLUTIONS.keys()))
        with col2:
            effect_level = st.selectbox("🎨 Livello effetti", ["soft", "medium", "hard"])
        with col3:
            fps_choice = st.selectbox("🎞️ Fotogrammi al secondo (FPS)", [5, 10, 15, 24, 30], index=4) # Default 30 FPS

        with st.spinner("📊 Analisi audio avanzata in corso (volume, tempo, spettrogramma)..."):
            mel_spec_norm, rms_frames, onset_frames = analyze_audio_features(y, sr, fps_choice, audio_duration)
        
        if mel_spec_norm is None or rms_frames is None or onset_frames is None:
            return # Se l'analisi fallisce, interrompi
        
        st.success("✅ Analisi audio completata con sincronizzazione volume/tempo!")
        
        # Mostra statistiche audio per informazione
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.metric("📊 Volume medio", f"{np.mean(rms_frames):.2f}")
            st.metric("🎵 Dinamica volume", f"{np.std(rms_frames):.2f}")
        with col_stats2:
            st.metric("⚡ Energia tempo media", f"{np.mean(onset_frames):.2f}")
            st.metric("🎼 Variazioni tempo", f"{np.std(onset_frames):.2f}")

        # Selezione dell'effetto artistico
        effect_mode = st.selectbox("✨ Effetto artistico", ["connessioni", "rettangoli", "geometriche", "linee_orizzontali", "linee_verticali"])

        st.markdown("🎨 Scegli i colori per le frequenze (basso, medio, alto):")
        col_low, col_mid, col_high = st.columns(3)
        with col_low:
            low_color = st.color_picker("Basse frequenze", "#000000") # Nero
        with col_mid:
            mid_color = st.color_picker("Medie frequenze", "#FF0000") # Rosso
        with col_high:
            high_color = st.color_picker("Alte frequenze", "#0000FF") # Blu

        bg_color_hex = st.color_picker("🎨 Colore sfondo", "#FFFFFF") # Bianco

        # Opzione per sincronizzare l'audio (richiede FFmpeg)
        sync_audio = st.checkbox("🔊 Sincronizza audio nel video", value=True, 
                                 help="Richiede l'installazione di FFmpeg sul sistema.")
        if sync_audio and not check_ffmpeg():
            st.warning("⚠️ FFmpeg non disponibile sul tuo sistema. La sincronizzazione audio sarà disabilitata. Assicurati che FFmpeg sia installato e nel PATH.")
            sync_audio = False

        if st.button("🎬 Genera Video Sincronizzato"):
            # Converte i colori esadecimali in formato BGR per OpenCV
            freq_colors = {
                'low': hex_to_bgr(low_color),
                'mid': hex_to_bgr(mid_color),
                'high': hex_to_bgr(high_color)
            }
            bg_color_bgr = hex_to_bgr(bg_color_hex)
            
            # Inizializza il generatore di video
            generator = VideoGenerator(
                FORMAT_RESOLUTIONS[video_format],
                effect_level,
                fps=fps_choice,
                bg_color=bg_color_bgr,
                freq_colors=freq_colors,
                effect_mode=effect_mode
            )
            
            # Genera il video
            success = generator.generate_video(mel_spec_norm, rms_frames, onset_frames, audio_duration, sync_audio)
            
            if success and os.path.exists(generator.FINAL):
                # Pulsante per scaricare il video finale
                with open(generator.FINAL, "rb") as f:
                    st.download_button(
                        "⬇️ Scarica il video", 
                        f, 
                        file_name=f"audio_linee_sync_{video_format}_{effect_level}_{effect_mode}.mp4", 
                        mime="video/mp4"
                    )
                file_size = os.path.getsize(generator.FINAL)
                st.info(f"📁 Dimensione file: {file_size / 1024 / 1024:.1f} MB")

                # --- Inizio Sezione Anteprima Video ---
                PREVIEW_DURATION = 5 # Durata dell'anteprima in secondi
                preview_file = "preview_output.mp4"
                
                st.subheader("Guarda un'Anteprima del Video")
                if check_ffmpeg():
                    try:
                        st.info(f"Generazione anteprima di {PREVIEW_DURATION} secondi con audio...")
                        # Usa FFmpeg per estrarre i primi PREVIEW_DURATION secondi
                        subprocess.run([
                            "ffmpeg", "-y", # Sovrascrivi il file di output se esiste
                            "-i", generator.FINAL, # Input è il video finale
                            "-t", str(PREVIEW_DURATION), # Durata dell'anteprima
                            "-c:v", "libx264", "-crf", "30", "-preset", "ultrafast", # Compressione leggera e veloce per anteprima
                            "-c:a", "copy", # Copia il flusso audio originale senza ricodifica
                            preview_file
                        ], capture_output=True, check=True) # Cattura output e solleva errore se il comando fallisce
                        
                        if os.path.exists(preview_file):
                            st.video(preview_file) # Mostra l'anteprima
                            os.remove(preview_file) # Pulisci il file di anteprima dopo la visualizzazione
                        else:
                            st.warning("⚠️ Impossibile creare il file di anteprima. Verifica i log per errori FFmpeg.")

                    except subprocess.CalledProcessError as e:
                        st.error(f"❌ Errore FFmpeg durante la creazione dell'anteprima: {e.stderr.decode()}")
                    except Exception as e:
                        st.error(f"❌ Errore durante la generazione dell'anteprima: {str(e)}")
                else:
                    st.warning("⚠️ FFmpeg non è disponibile. Impossibile generare l'anteprima.")
                # --- Fine Sezione Anteprima Video ---

            else:
                st.error("❌ Errore nella generazione del video.")

        # Pulsante per pulire i file temporanei
        if st.button("🧹 Pulisci file temporanei"):
            temp_files = ["input_audio.wav", "temp_output.mp4", "final_output.mp4", "preview_output.mp4"]
            for f in temp_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception as e:
                        st.error(f"❌ Impossibile eliminare il file {f}: {str(e)}")
            st.success("✅ File temporanei eliminati!")

if __name__ == "__main__":
    main()
