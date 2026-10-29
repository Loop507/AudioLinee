import streamlit as st
import numpy as np
import cv2
import librosa
import os
import math
import subprocess
import gc
import shutil
import glob
import tempfile
from typing import Tuple, Optional

MAX_DURATION = 300
MIN_DURATION = 1.0
MAX_FILE_SIZE = 50 * 1024 * 1024
DEFAULT_SEED = 507
# Soglie sul picco reale del segnale (dBFS, 0 dBFS = fondo scala digitale)
PEAK_SUB_CLIPPING_DBFS = -0.3   # picco >= -0.3 dBFS  -> Sub_Clipping
PEAK_MID_SATURATION_DBFS = -3.0 # picco >= -3 dBFS    -> Mid_Saturation, altrimenti Clean_Signal

def check_ffmpeg() -> bool:
    """Verifica se FFmpeg è installato e disponibile nel PATH."""
    return shutil.which("ffmpeg") is not None

def validate_audio_file(uploaded_file) -> bool:
    """Valida le dimensioni del file audio caricato."""
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"❌ File troppo grande ({uploaded_file.size / 1024 / 1024:.1f}MB). Limite: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
        return False
    return True

def _decode_with_ffmpeg(file_path: str) -> Optional[str]:
    """Fallback: decodifica il file in WAV con FFmpeg quando librosa/libsndfile non riesce a leggerlo.
    librosa 1.0 ha rimosso il backend audioread, quindi alcuni mp3 non letti da libsndfile
    passano da qui."""
    if not check_ffmpeg():
        return None
    out_path = os.path.splitext(file_path)[0] + "_decoded.wav"
    try:
        subprocess.run(["ffmpeg", "-y", "-i", file_path, "-vn", "-ac", "1", out_path],
                       capture_output=True, check=True)
        return out_path
    except Exception:
        return None

def load_and_process_audio(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[float]]:
    """
    Carica e processa il file audio, gestendo la durata e i potenziali errori.
    """
    try:
        # Carica l'audio con librosa, mantenendo il sample rate originale e convertendo a mono.
        # Se la lettura diretta fallisce, riprova passando da una decodifica FFmpeg.
        try:
            y, sr = librosa.load(file_path, sr=None, mono=True)
        except Exception:
            decoded = _decode_with_ffmpeg(file_path)
            if decoded is None:
                raise
            y, sr = librosa.load(decoded, sr=None, mono=True)

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

def analyze_audio_features(y: np.ndarray, sr: int, fps: int, duration: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Analizza il volume (RMS), il tempo musicale (onset strength) e genera lo spettrogramma
    per la sincronizzazione visiva.
    """
    try:
        # Genera melspettrogramma per rappresentare le frequenze
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr / 2)
        if S.size == 0 or S.shape[1] == 0:
            st.error("❌ Impossibile generare lo spettrogramma: l'audio è troppo breve o non è valido.")
            return None, None, None, None
        
        # Converte lo spettrogramma in scala di decibel e normalizza tra 0 e 1
        mel_spec_db = librosa.power_to_db(S, ref=np.max)
        min_val = mel_spec_db.min()
        max_val = mel_spec_db.max()
        if max_val == min_val: # Gestisce il caso di audio piatto
            st.error("❌ L'audio non contiene variazioni sufficienti per generare il video.")
            return None, None, None, None
        mel_spec_norm = ((mel_spec_db - min_val) / (max_val - min_val)).astype(np.float32)
        
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
        
        # Stima BPM con librosa
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            bpm = float(np.atleast_1d(tempo)[0])
        except Exception:
            bpm = 0.0
        
        return mel_spec_norm, rms_frames, onset_frames, bpm
        
    except Exception as e:
        st.error(f"❌ Errore nell'analisi audio: {str(e)}")
        return None, None, None, None

def measure_signal_level(y: np.ndarray) -> Tuple[float, str]:
    """Misura il picco reale del segnale (prima di qualsiasi normalizzazione) e restituisce
    (picco in dBFS, etichetta). Un mp3 decodificato può superare 0 dBFS: in quel caso è clipping."""
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-10))
    if peak_dbfs >= PEAK_SUB_CLIPPING_DBFS:
        label = "Sub_Clipping"
    elif peak_dbfs >= PEAK_MID_SATURATION_DBFS:
        label = "Mid_Saturation"
    else:
        label = "Clean_Signal"
    return peak_dbfs, label

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Converte un colore esadecimale (es. '#RRGGBB') in un tuple BGR (Blue, Green, Red).
    Fallback a bianco se il formato non è valido (difesa contro input malformati)."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (255, 255, 255)
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        return (255, 255, 255)
    return (b, g, r) # OpenCV usa BGR

class VideoGenerator:
    """
    Classe per generare effetti visivi sincronizzati con l'audio.
    """
    def __init__(self, format_res: Tuple[int, int], level: str, fps: int = 30, 
                 bg_color: Tuple[int, int, int] = (255, 255, 255), 
                 freq_colors: Optional[dict] = None, effect_mode: str = "connessioni",
                 seed: int = DEFAULT_SEED, workdir: str = ".", audio_path: Optional[str] = None):
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
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed) # Generatore con seed: stesso audio + stesso seed = stesso video
        self.workdir = workdir
        self.audio_path = audio_path or os.path.join(workdir, "input_audio.wav")
        self.TEMP = os.path.join(workdir, "temp_output.mp4") # File temporaneo per il video senza audio
        self.FINAL = os.path.join(workdir, "final_output.mp4") # File finale con audio sincronizzato
        self.muxed = False # True solo se FFmpeg ha davvero codificato H.264 + AAC

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

    # ------------------------------------------------------------------
    # Helper comuni (sostituiscono il codice ripetuto in ogni effetto)
    # ------------------------------------------------------------------
    def _rand_pt(self) -> Tuple[int, int]:
        """Punto casuale dentro il frame, come interi Python."""
        return (int(self.rng.integers(0, self.W)), int(self.rng.integers(0, self.H)))

    def _jitter(self, scale: float) -> int:
        """Spostamento casuale simmetrico in [-scale, +scale]. Sicuro anche con scale = 0."""
        s = int(scale)
        if s <= 0:
            return 0
        return int(self.rng.integers(-s, s + 1))

    def _pick_color(self, mel: np.ndarray, idx: int, volume_factor: float) -> Tuple[Tuple[int, int, int], float]:
        """Sceglie una banda mel a caso e restituisce (colore BGR modulato dall'intensità, valore spettrale)."""
        spec_idx = int(self.rng.integers(0, mel.shape[0]))
        spec_val = float(mel[spec_idx, idx])
        base = self.freq_to_color(spec_idx)
        intensity = 0.5 + 0.5 * spec_val * volume_factor # Più volume e valore spettrale = più intensa
        return (int(base[0] * intensity), int(base[1] * intensity), int(base[2] * intensity)), spec_val

    def _blend_rect_alpha(self, frame: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int],
                           color_bgr: Tuple[int, int, int], alpha: float) -> None:
        """Blenda un rettangolo pieno nel frame con alpha compositing, operando SOLO sulla
        regione (ROI) del rettangolo invece che sull'intero frame. Riduce drasticamente il
        costo di frame.copy() quando si disegnano molte forme per frame (es. draw_squares,
        draw_rectangles, draw_random_shapes), mantenendo lo stesso risultato visivo."""
        H, W = frame.shape[:2]
        x1, y1 = top_left
        x2, y2 = bottom_right
        # cv2.rectangle usa coordinate inclusive (include il pixel bottom-right):
        # +1 per compensare lo slicing NumPy, che e' invece esclusivo sull'estremo superiore.
        x1c, y1c = max(x1, 0), max(y1, 0)
        x2c, y2c = min(x2 + 1, W), min(y2 + 1, H)
        if x2c <= x1c or y2c <= y1c:
            return
        roi = frame[y1c:y2c, x1c:x2c]
        overlay = roi.copy()
        overlay[:, :] = color_bgr
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

    def _blend_box(self, frame: np.ndarray, cx: int, cy: int, w: int, h: int,
                   jx: int, jy: int, color_bgr: Tuple[int, int, int], alpha: float) -> None:
        """Calcola gli angoli di un rettangolo centrato in (cx, cy) con jitter, li limita ai bordi
        del frame e lo blenda con alpha."""
        x1 = int(np.clip(int(cx - w / 2 + jx), 0, self.W))
        y1 = int(np.clip(int(cy - h / 2 + jy), 0, self.H))
        x2 = int(np.clip(int(cx + w / 2 + jx), 0, self.W))
        y2 = int(np.clip(int(cy + h / 2 + jy), 0, self.H))
        self._blend_rect_alpha(frame, (x1, y1), (x2, y2), color_bgr, alpha)

    def _axis_line(self, frame: np.ndarray, mel: np.ndarray, idx: int, i: int, vertical: bool,
                   base_pos: int, amp: float, volume_factor: float, tempo_factor: float, mood: float) -> None:
        """Disegna una linea a tutta larghezza (orizzontale) o altezza (verticale) con
        movimento ondulatorio basato sul tempo, colore e spessore basati su spettro e volume."""
        limit = (self.W if vertical else self.H) - 1
        offset = int(amp * tempo_factor * mood * np.sin(idx * 0.1 + i * 0.5))
        pos = int(np.clip(base_pos + offset, 0, limit)) # Assicura che la linea rimanga nel frame
        color_bgr, spec_val = self._pick_color(mel, idx, volume_factor)
        thickness = max(1, int(2 + spec_val * 5 * mood * volume_factor))
        if vertical:
            cv2.line(frame, (pos, 0), (pos, self.H), color_bgr, thickness)
        else:
            cv2.line(frame, (0, pos), (self.W, pos), color_bgr, thickness)

    # ------------------------------------------------------------------
    # Effetti
    # ------------------------------------------------------------------
    def draw_connected_lines(self, frame: np.ndarray, mel: np.ndarray, idx: int, volume_factor: float, tempo_factor: float):
        """Disegna un effetto di linee connesse basato sull'audio."""
        mood = self.get_mood_factor()
        
        # Numero di punti base, influenzato dal mood e altamente dal volume
        base_pts = int(8 * mood) 
        volume_pts = max(3, int(base_pts * (0.4 + 1.2 * volume_factor))) # Maggiore sensibilità al volume
        
        # Probabilità di connessione, influenzata dal tempo musicale
        connection_probability = 0.15 + (0.7 * tempo_factor * mood) 
        
        pts = [self._rand_pt() for _ in range(volume_pts)]
        
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if self.rng.random() < connection_probability:
                    # Preleva un valore dallo spettrogramma (random per varietà)
                    val = mel[int(self.rng.integers(0, mel.shape[0])), idx]
                    
                    # Soglia dinamica: più alta di base, ma abbassata da volume e tempo elevati
                    threshold = 0.25 * mood * (1.2 - volume_factor * 0.6 - tempo_factor * 0.2) 
                    
                    if val > threshold:
                        color = self.freq_to_color(int(self.rng.integers(0, mel.shape[0])))
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
        jitter_scale = (margin_x // 4) * mood * (0.3 + 0.7 * tempo_factor) # Più tempo = più jitter
        
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                x = c * margin_x + self._jitter(jitter_scale)
                y = r * margin_y + self._jitter(jitter_scale)
                pts.append((x, y))

        max_dist_threshold = max(margin_x, margin_y) * 1.8 # Distanza massima per connettere i punti
        # Soglia dinamica: rende le connessioni più difficili a basso volume/tempo
        threshold = 0.15 * mood * (1.3 - volume_factor * 0.5 - tempo_factor * 0.3)

        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dist = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                
                if dist < max_dist_threshold:
                    val = mel[int(self.rng.integers(0, mel.shape[0])), idx]
                    
                    if val > threshold:
                        color = self.freq_to_color(int(self.rng.integers(0, mel.shape[0])))
                        # Spessore basato su volume, tempo e intensità spettrale
                        thickness = max(1, int(val * 4 * mood * (0.3 + 0.4 * volume_factor + 0.3 * tempo_factor)))
                        cv2.line(frame, pts[i], pts[j], color, thickness)

    def draw_complex_geometric_network(self, frame: np.ndarray, mel: np.ndarray, idx: int, volume_factor: float, tempo_factor: float):
        """Disegna un effetto di rete geometrica complessa basata sull'audio."""
        mood = self.get_mood_factor()
        
        # Numero di punti basato sul volume, con una base più conservativa
        base_pts = int(20 * mood) 
        volume_pts = max(8, int(base_pts * (0.3 + 0.8 * volume_factor))) # Forte influenza del volume
        
        pts = [self._rand_pt() for _ in range(volume_pts)]
        
        # Probabilità di connessione basata sul tempo
        connection_prob = 0.18 + (0.4 * tempo_factor * mood) # La probabilità aumenta con il tempo
        
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if self.rng.random() < connection_prob:
                    val = mel[int(self.rng.integers(0, mel.shape[0])), idx]
                    # Soglia adattiva: influenza maggiore di volume e tempo
                    threshold = 0.22 * mood * (1.15 - volume_factor * 0.4 - tempo_factor * 0.25)
                    
                    if val > threshold:
                        color = self.freq_to_color(int(self.rng.integers(0, mel.shape[0])))
                        # Spessore che combina tutti i fattori, con forte peso sul volume e spettro
                        thickness = max(1, int(val * 5 * mood * (0.2 + 0.5 * volume_factor + 0.3 * tempo_factor)))
                        cv2.line(frame, pts[i], pts[j], color, thickness)

    def draw_horizontal_lines(self, frame: np.ndarray, mel: np.ndarray, idx: int, volume_factor: float, tempo_factor: float):
        """Disegna un effetto di linee orizzontali pulsanti basato sull'audio."""
        mood = self.get_mood_factor()
        
        # Numero di linee proporzionale al volume e alla difficoltà
        num_lines = int(5 + 20 * volume_factor * mood) 
        
        for i in range(num_lines):
            y_base = int((i / num_lines) * self.H)
            self._axis_line(frame, mel, idx, i, False, y_base, 20, volume_factor, tempo_factor, mood)

    def draw_vertical_lines(self, frame: np.ndarray, mel: np.ndarray, idx: int, volume_factor: float, tempo_factor: float):
        """Disegna un effetto di linee verticali pulsanti basato sull'audio."""
        mood = self.get_mood_factor()
        
        # Numero di linee proporzionale al volume e alla difficoltà
        num_lines = int(5 + 20 * volume_factor * mood) 
        
        for i in range(num_lines):
            x_base = int((i / num_lines) * self.W)
            self._axis_line(frame, mel, idx, i, True, x_base, 20, volume_factor, tempo_factor, mood)

    def draw_random_vertical_horizontal_lines(self, frame: np.ndarray, mel: np.ndarray, idx: int, volume_factor: float, tempo_factor: float):
        """Disegna un effetto che combina linee orizzontali e verticali casuali basate sull'audio."""
        mood = self.get_mood_factor()
        
        # Numero totale di linee (circa metà orizzontali, metà verticali)
        num_total_lines = int(10 + 30 * volume_factor * mood) 
        
        for i in range(num_total_lines):
            if self.rng.random() < 0.5: # 50% di probabilità per orizzontale
                y_base = int(self.rng.integers(0, self.H)) # Posizione Y casuale
                self._axis_line(frame, mel, idx, i, False, y_base, 15, volume_factor, tempo_factor, mood)
            else: # 50% di probabilità per verticale
                x_base = int(self.rng.integers(0, self.W)) # Posizione X casuale
                self._axis_line(frame, mel, idx, i, True, x_base, 15, volume_factor, tempo_factor, mood)

    def draw_squares(self, frame: np.ndarray, mel: np.ndarray, idx: int, volume_factor: float, tempo_factor: float):
        """Disegna quadrati di varie dimensioni basati sull'audio."""
        mood = self.get_mood_factor()
        
        # Numero di quadrati influenzato dal volume e dal mood
        num_squares = int(20 + 50 * volume_factor * mood)

        # Dimensione del quadrato influenzata dal volume (uguale per tutti i quadrati del frame)
        size = int(np.exp(np.log(5) + (np.log(80) - np.log(5)) * (0.3 + 0.7 * volume_factor * mood)))
        size = max(5, min(size, min(self.W, self.H) // 5)) # Limita la dimensione massima

        # Opacità simulata, più alta con volume/tempo
        alpha = max(0.2, min(0.7, 0.3 + 0.6 * volume_factor * tempo_factor * mood))
        
        for _ in range(num_squares):
            cx, cy = self._rand_pt()
            
            # Jitter basato sul tempo
            jx = int(10 * tempo_factor * mood * (self.rng.random() - 0.5))
            jy = int(10 * tempo_factor * mood * (self.rng.random() - 0.5))

            color_bgr, _ = self._pick_color(mel, idx, volume_factor)
            self._blend_box(frame, cx, cy, size, size, jx, jy, color_bgr, alpha)

    def draw_rectangles(self, frame: np.ndarray, mel: np.ndarray, idx: int, volume_factor: float, tempo_factor: float):
        """Disegna rettangoli di varie dimensioni e proporzioni basati sull'audio."""
        mood = self.get_mood_factor()
        
        # Numero di rettangoli
        num_rects = int(20 + 50 * volume_factor * mood)

        # Dimensione base influenzata dal volume (uguale per tutti i rettangoli del frame)
        base_dim = int(np.exp(np.log(5) + (np.log(100) - np.log(5)) * (0.3 + 0.7 * volume_factor * mood)))
        base_dim = max(5, min(base_dim, min(self.W, self.H) // 4))

        alpha = max(0.2, min(0.7, 0.3 + 0.6 * volume_factor * tempo_factor * mood))
        
        for _ in range(num_rects):
            cx, cy = self._rand_pt()
            
            # Proporzione casuale influenzata dal tempo (più tempo = più variazione)
            aspect_ratio = 0.5 + 1.5 * self.rng.random() * (0.5 + 0.5 * tempo_factor * mood)
            width = int(base_dim * aspect_ratio)
            height = int(base_dim / aspect_ratio)
            
            jx = int(10 * tempo_factor * mood * (self.rng.random() - 0.5))
            jy = int(10 * tempo_factor * mood * (self.rng.random() - 0.5))

            color_bgr, _ = self._pick_color(mel, idx, volume_factor)
            self._blend_box(frame, cx, cy, width, height, jx, jy, color_bgr, alpha)

    def draw_random_shapes(self, frame: np.ndarray, mel: np.ndarray, idx: int, volume_factor: float, tempo_factor: float):
        """Disegna casualmente quadrati o rettangoli basati sull'audio."""
        mood = self.get_mood_factor()
        
        # Numero totale di forme
        num_shapes = int(30 + 70 * volume_factor * mood)

        base_dim = int(np.exp(np.log(5) + (np.log(100) - np.log(5)) * (0.3 + 0.7 * volume_factor * mood)))
        base_dim = max(5, min(base_dim, min(self.W, self.H) // 4))

        alpha = max(0.2, min(0.7, 0.3 + 0.6 * volume_factor * tempo_factor * mood))
        
        for _ in range(num_shapes):
            cx, cy = self._rand_pt()
            
            jx = int(10 * tempo_factor * mood * (self.rng.random() - 0.5))
            jy = int(10 * tempo_factor * mood * (self.rng.random() - 0.5))

            color_bgr, _ = self._pick_color(mel, idx, volume_factor)

            if self.rng.random() < 0.5: # 50% di probabilità per quadrato
                width = height = base_dim
            else: # 50% di probabilità per rettangolo
                aspect_ratio = 0.5 + 1.5 * self.rng.random() * (0.5 + 0.5 * tempo_factor * mood)
                width = int(base_dim * aspect_ratio)
                height = int(base_dim / aspect_ratio)

            self._blend_box(frame, cx, cy, width, height, jx, jy, color_bgr, alpha)

    def generate_video(self, mel: np.ndarray, rms_frames: np.ndarray, onset_frames: np.ndarray, duration: float, sync_audio: bool = True) -> bool:
        """
        Genera il video frame per frame, applicando gli effetti visivi sincronizzati
        e opzionalmente unendo l'audio alla fine.
        """
        # Pulisce i file temporanei precedenti
        for f in [self.TEMP, self.FINAL]:
            if os.path.exists(f): 
                os.remove(f)

        self.muxed = False
        total_frames = int(duration * self.FPS)
        # Inizializza il VideoWriter di OpenCV
        writer = cv2.VideoWriter(self.TEMP, cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (self.W, self.H))
        
        if not writer.isOpened():
            st.error("❌ Impossibile inizializzare il writer video.")
            return False
            
        progress = st.progress(0) # Barra di progresso per Streamlit
        status = st.empty() # Placeholder per messaggi di stato

        draw_fn = {
            "connessioni": self.draw_connected_lines,
            "rettangoli_griglia": self.draw_rectangular_grid,
            "geometriche": self.draw_complex_geometric_network,
            "linee_orizzontali": self.draw_horizontal_lines,
            "linee_verticali": self.draw_vertical_lines,
            "linee_casuali_verticali_orizzontali": self.draw_random_vertical_horizontal_lines,
            "quadrati": self.draw_squares,
            "rettangoli": self.draw_rectangles,
            "forme_casuali_quadrati_rettangoli": self.draw_random_shapes,
        }.get(self.effect_mode)

        bg = np.array(self.bg_color, dtype=np.uint8)
        last_frame = max(total_frames - 1, 1)
        err_count = 0
        
        # Loop per generare ogni frame del video
        try:
            for i in range(total_frames):
                # Inizializza il frame con il colore di sfondo (SEMPRE, anche se l'effetto fallisce sotto,
                # cosi' writer.write viene chiamato ad ogni iterazione: numero di frame = total_frames
                # garantito, evitando disallineamenti audio/video nel merge FFmpeg con -shortest)
                frame = np.empty((self.H, self.W, 3), dtype=np.uint8)
                frame[:] = bg
                try:
                    # Stesso mapping lineare usato per RMS/onset (np.linspace sui frame video):
                    # il primo frame cade sul primo frame audio, l'ultimo sull'ultimo.
                    t_idx = int(round(i * (mel.shape[1] - 1) / last_frame))
                    t_idx = min(max(t_idx, 0), mel.shape[1] - 1)
                    
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
                    if draw_fn is not None:
                        draw_fn(frame, mel, t_idx, volume_factor, tempo_factor)

                    # Aggiorna la barra di progresso e lo stato in Streamlit
                    if i % 10 == 0 or i == total_frames - 1: # Aggiorna ogni 10 frame o all'ultimo
                        progress.progress((i + 1) / total_frames)
                        status.text(f"🎬 Generazione Frame {i + 1}/{total_frames} (Vol: {volume_factor:.2f}, Tempo: {tempo_factor:.2f})")

                except Exception as e:
                    err_count += 1
                    if err_count <= 3: # Evita di inondare la pagina di warning se l'errore si ripete su ogni frame
                        st.warning(f"⚠️ Errore al frame {i}: {str(e)}")

                writer.write(frame) # Scrive comunque il frame (anche solo sfondo) per mantenere il conteggio frame corretto
        finally:
            writer.release() # Rilascia sempre il writer video, anche in caso di errore

        if err_count > 3:
            st.warning(f"⚠️ Errori in totale su {err_count} frame (mostrati solo i primi 3).")
        
        # Sincronizzazione audio con FFmpeg
        if sync_audio and check_ffmpeg() and os.path.exists(self.audio_path):
            try:
                status.text("🎶 Sincronizzazione audio in corso (potrebbe richiedere qualche istante)...")
                subprocess.run([
                    "ffmpeg", "-y", # Sovrascrivi il file di output se esiste
                    "-i", self.TEMP, # Input video (senza audio)
                    "-i", self.audio_path, # Input audio
                    "-map", "0:v:0", "-map", "1:a:0", # Solo video del primo input e audio del secondo (ignora eventuali cover art)
                    "-c:v", "libx264", "-crf", "28", "-preset", "veryfast", # Codec video H.264, qualità, preset di velocità
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", # Compatibilità con player/social e avvio rapido
                    "-c:a", "aac", "-b:a", "192k", # Codec audio AAC
                    "-t", f"{total_frames / self.FPS:.3f}", "-shortest", # Taglia l'audio alla durata del video (anche dopo il troncamento a MAX_DURATION)
                    self.FINAL
                ], capture_output=True, check=True) # Cattura output e solleva errore se il comando fallisce
                os.remove(self.TEMP) # Rimuovi il video temporaneo senza audio
                self.muxed = True
            except subprocess.CalledProcessError as e:
                err_txt = (e.stderr or b"").decode(errors="replace")[-500:]
                st.error(f"❌ Errore FFmpeg durante la sincronizzazione audio: {err_txt}")
                os.replace(self.TEMP, self.FINAL) # Se FFmpeg fallisce, salva comunque il video senza audio
            except Exception as e:
                st.error(f"❌ Errore generico durante la sincronizzazione audio: {str(e)}")
                os.replace(self.TEMP, self.FINAL) # Se fallisce, salva comunque il video senza audio
        else:
            # Se FFmpeg non c'è o la sincronizzazione è disabilitata, rinomina il file temporaneo
            os.replace(self.TEMP, self.FINAL)
            
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

    # Inizializza session state per sopravvivere ai rerun da download_button
    defaults = {
        "video_path": None,
        "report_text": None,
        "video_filename": None,
        "report_filename": None,
        "file_size_mb": None,
        "workdir": None,       # cartella temporanea unica per sessione (niente conflitti tra utenti)
        "upload_key": None,    # identifica l'audio caricato
        "audio_meta": None,    # audio decodificato (y, sr, durata) in cache
        "analysis": None,      # risultato dell'analisi in cache
        "analysis_key": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Ogni sessione lavora in una cartella temporanea propria
    if not st.session_state.workdir or not os.path.isdir(st.session_state.workdir):
        st.session_state.workdir = tempfile.mkdtemp(prefix="audiolinee_")
    workdir = st.session_state.workdir

    uploaded_file = st.file_uploader("🎧 Carica un file audio (.wav o .mp3)", type=["wav", "mp3"])
    if uploaded_file is not None:
        if not validate_audio_file(uploaded_file):
            return

        # Mantiene l'estensione originale (mp3 resta mp3)
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext not in (".wav", ".mp3"):
            ext = ".wav"
        audio_path = os.path.join(workdir, f"input_audio{ext}")
        upload_id = (getattr(uploaded_file, "file_id", None) or uploaded_file.name, uploaded_file.size)

        # Nuovo audio: azzera risultati, cache e file del caricamento precedente
        if st.session_state.upload_key != upload_id:
            for old in glob.glob(os.path.join(workdir, "input_audio*")):
                try:
                    os.remove(old)
                except OSError:
                    pass
            st.session_state.upload_key = upload_id
            st.session_state.audio_meta = None
            st.session_state.analysis = None
            st.session_state.analysis_key = None
            st.session_state.video_path = None
            st.session_state.report_text = None

        # Salva il file audio caricato localmente per librosa e FFmpeg (solo se manca o è cambiato)
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) != uploaded_file.size:
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("🔊 Audio caricato correttamente!")

        # Carica e processa l'audio (una sola volta per file caricato)
        meta = st.session_state.audio_meta
        if meta is None or meta["key"] != upload_id:
            y, sr, audio_duration = load_and_process_audio(audio_path)
            if y is None: # Se il caricamento o la validazione falliscono
                return
            meta = {"key": upload_id, "y": y, "sr": sr, "duration": audio_duration}
            st.session_state.audio_meta = meta
        elif meta["duration"] >= MAX_DURATION:
            st.warning(f"⚠️ Audio troncato a {MAX_DURATION}s.")
        y, sr, audio_duration = meta["y"], meta["sr"], meta["duration"]
        st.info(f"🔊 Durata audio: {audio_duration:.2f} secondi")

        # Selezione parametri video prima dell'analisi per influenzare l'interpolazione
        col1, col2, col3 = st.columns(3)
        with col1:
            video_format = st.selectbox("📐 Formato video", list(FORMAT_RESOLUTIONS.keys()))
        with col2:
            effect_level = st.selectbox("🎨 Livello effetti", ["soft", "medium", "hard"])
        with col3:
            fps_choice = st.selectbox("🎞️ Fotogrammi al secondo (FPS)", [5, 10, 15, 24, 30], index=4) # Default 30 FPS

        # Analisi in cache: si rifà solo se cambia l'audio o gli FPS
        analysis_key = (upload_id, fps_choice)
        if st.session_state.analysis_key != analysis_key:
            with st.spinner("📊 Analisi audio avanzata in corso (volume, tempo, spettrogramma)..."):
                result = analyze_audio_features(y, sr, fps_choice, audio_duration)
            if result[0] is None or result[1] is None or result[2] is None:
                return # Se l'analisi fallisce, interrompi
            st.session_state.analysis = result
            st.session_state.analysis_key = analysis_key
        mel_spec_norm, rms_frames, onset_frames, bpm_detected = st.session_state.analysis
        
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
        effect_mode = st.selectbox("✨ Effetto artistico", [
            "connessioni", 
            "rettangoli_griglia", 
            "geometriche", 
            "linee_orizzontali", 
            "linee_verticali", 
            "linee_casuali_verticali_orizzontali",
            "quadrati",
            "rettangoli",
            "forme_casuali_quadrati_rettangoli"
        ])

        seed = st.number_input("🎲 Seed (stesso audio + stesso seed = stesso video)", min_value=0,
                               max_value=2**31 - 1, value=DEFAULT_SEED, step=1)

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
                effect_mode=effect_mode,
                seed=int(seed),
                workdir=workdir,
                audio_path=audio_path
            )
            
            # Genera il video
            success = generator.generate_video(mel_spec_norm, rms_frames, onset_frames, audio_duration, sync_audio)
            
            if success and os.path.exists(generator.FINAL):
                file_size = os.path.getsize(generator.FINAL)

                # Costruisci report
                effect_style_map = {
                    "connessioni": "Connected Lines / Node Graph Synthesis",
                    "rettangoli_griglia": "Rectangular Grid / Lattice Topology",
                    "geometriche": "Complex Geometric Network / Poly-Node Web",
                    "linee_orizzontali": "Horizontal Line Pulse / Waveform Scan",
                    "linee_verticali": "Vertical Line Pulse / Spectral Column",
                    "linee_casuali_verticali_orizzontali": "Stochastic Line Grid / Random Axis Emission",
                    "quadrati": "Square Geometry / Tessellation Field",
                    "rettangoli": "Rectangular Mass / Aspect-Ratio Drift",
                    "forme_casuali_quadrati_rettangoli": "Random Shape Collapse / Mixed Geometry Field"
                }
                effect_label = effect_style_map.get(effect_mode, effect_mode.replace("_", " ").title())
                mins = int(audio_duration) // 60
                secs = int(audio_duration) % 60
                duration_str = f"{mins:02d}:{secs:02d}"
                vol_mean = float(np.mean(rms_frames))
                vol_std = float(np.std(rms_frames))
                onset_mean = float(np.mean(onset_frames))
                peak_dbfs, clipping_label = measure_signal_level(y) # picco reale del segnale, non RMS normalizzato
                sr_khz = f"{sr // 1000}kHz"
                bpm_str = f"{bpm_detected:.1f} BPM" if bpm_detected and bpm_detected > 0 else "N/A BPM"
                res_w, res_h = FORMAT_RESOLUTIONS[video_format]
                # Codec reali: se FFmpeg ha fallito il file è mp4v senza audio, e il report lo dice
                video_codec = "H.264" if generator.muxed else "mp4v (RAW)"
                audio_codec = "AAC" if generator.muxed else "No Audio"
                level_map = {"soft": "Soft_Dynamics", "medium": "Medium_Impact", "hard": "Hard_Saturation"}
                level_label = level_map.get(effect_level, effect_level.upper())
                total_frames_count = int(audio_duration * fps_choice)

                report_text = f"""[ALINE_ARCHIVE] // PATTERN_{effect_mode.upper()} // {video_codec} // {audio_codec}
:: STYLE: {effect_label}
:: ENGINE: audio_line_generator [03.00]
:: SEED: {int(seed)} / PCG64 / reproducible
:: AUDIO: {sr_khz} / 32-bit Float / {bpm_str} / {clipping_label} / Peak={peak_dbfs:+.1f}dBFS
:: PATTERN: {effect_mode.replace("_", " ").title()}
:: LEVEL: {level_label} / Vol_μ={vol_mean:.2f} / Vol_σ={vol_std:.2f} / Onset_μ={onset_mean:.2f}
:: FORMAT: {video_format} / {res_w}×{res_h}px / {fps_choice} FPS / {total_frames_count} Frames
:: PROCESS: Mel-Spectrogram [{sr // 2 // 1000}kHz max] → RMS+Onset Sync → Frame Synthesis → {video_codec} Encode
:: DURATION: {duration_str}
"It's not a video. It's the music rendered as geometry."
> Direction & Algorithm: Loop507
#AudioVisual #GenerativeArt #AlgorithmicVideo #SoundDesign #NewMediaArt #VisualMusic #SpectralArt #MelSpectrogram #SignalProcessing #AudioSync #ComputationalArt #FrequencyMapping #MotionDesign #DataArt #OpenCVArt"""

                # Salva tutto in session_state → sopravvive ai rerun dei download_button
                st.session_state.video_path = generator.FINAL
                st.session_state.video_filename = f"audio_linee_sync_{video_format}_{effect_level}_{effect_mode}.mp4"
                st.session_state.report_text = report_text
                st.session_state.report_filename = f"report_audio_linee_{effect_mode}_{video_format}.txt"
                st.session_state.file_size_mb = file_size / 1024 / 1024

                # Anteprima (solo al momento della generazione, non persiste — ma non serve)
                PREVIEW_DURATION = 5
                preview_file = os.path.join(workdir, "preview_output.mp4")
                st.subheader("Guarda un'Anteprima del Video")
                if check_ffmpeg():
                    try:
                        st.info(f"Generazione anteprima di {PREVIEW_DURATION} secondi con audio...")
                        subprocess.run([
                            "ffmpeg", "-y",
                            "-i", generator.FINAL,
                            "-t", str(PREVIEW_DURATION),
                            "-c:v", "libx264", "-crf", "30", "-preset", "ultrafast",
                            "-pix_fmt", "yuv420p",
                            "-c:a", "copy",
                            preview_file
                        ], capture_output=True, check=True)
                        if os.path.exists(preview_file):
                            with open(preview_file, "rb") as pf:
                                st.video(pf.read())
                            os.remove(preview_file)
                        else:
                            st.warning("⚠️ Impossibile creare il file di anteprima.")
                    except subprocess.CalledProcessError as e:
                        st.error(f"❌ Errore FFmpeg anteprima: {(e.stderr or b'').decode(errors='replace')[-500:]}")
                    except Exception as e:
                        st.error(f"❌ Errore anteprima: {str(e)}")
                else:
                    st.warning("⚠️ FFmpeg non disponibile. Impossibile generare l'anteprima.")

            else:
                st.error("❌ Errore nella generazione del video.")

        # --- Download video + report: FUORI dal blocco genera, leggono da session_state ---
        # Sopravvivono ai rerun causati dai download_button stessi
        if st.session_state.video_path and os.path.exists(st.session_state.video_path):
            st.markdown("---")
            st.info(f"📁 Dimensione file: {st.session_state.file_size_mb:.1f} MB")

            with open(st.session_state.video_path, "rb") as f:
                st.download_button(
                    "⬇️ Scarica il video",
                    f,
                    file_name=st.session_state.video_filename,
                    mime="video/mp4",
                    key="dl_video"
                )

        if st.session_state.report_text:
            st.markdown("---")
            st.subheader("📋 Report per Social / YouTube")
            st.code(st.session_state.report_text, language=None)
            st.download_button(
                label="⬇️ Scarica Report (.txt)",
                data=st.session_state.report_text,
                file_name=st.session_state.report_filename,
                mime="text/plain",
                key="dl_report"
            )

        # Pulsante per pulire i file temporanei
        if st.button("🧹 Pulisci file temporanei"):
            temp_files = ["temp_output.mp4", "final_output.mp4", "preview_output.mp4"]
            for name in temp_files:
                path = os.path.join(workdir, name)
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        st.error(f"❌ Impossibile eliminare il file {name}: {str(e)}")
            # L'audio caricato viene riscritto automaticamente al prossimo rerun se serve
            for old in glob.glob(os.path.join(workdir, "input_audio*")):
                try:
                    os.remove(old)
                except OSError:
                    pass
            st.session_state.video_path = None
            st.session_state.report_text = None
            st.success("✅ File temporanei eliminati!")

if __name__ == "__main__":
    main()
