# 🎵 AudioLinee.py - Versione stabile aggiornata con BPM, FPS, colori e 3 effetti

import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import librosa
from PIL import Image

st.set_page_config(page_title="🎞️ AudioLinee", layout="wide")

# --- Funzione per convertire colore esadecimale a BGR OpenCV ---
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

# --- Funzione stima BPM ---
def estimate_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo

# --- Effetto Linee ---
def draw_lines(frequencies, volume, width, height, line_color):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    center = height // 2
    max_freq = len(frequencies)
    for i, energy in enumerate(frequencies):
        if energy < volume * 0.05:
            continue
        thickness = int(np.interp(energy, [0, np.max(frequencies)], [1, 10]))
        x = int(np.interp(i, [0, max_freq], [0, width]))
        cv2.line(img, (x, center - 50), (x, center + 50), line_color, thickness)
    return img

# --- Effetto Spectrum (linee verticali, orizzontali o entrambe) ---
def draw_spectrum(frequencies, volume, width, height, line_color, mode):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    max_freq = len(frequencies)
    center_y = height // 2
    center_x = width // 2

    for i, energy in enumerate(frequencies):
        if energy < volume * 0.05:
            continue
        thickness = int(np.interp(energy, [0, np.max(frequencies)], [1, 8]))
        x = int(np.interp(i, [0, max_freq], [0, width]))
        y = int(np.interp(i, [0, max_freq], [0, height]))
        if mode == 'soft':  # solo verticali
            cv2.line(img, (x, 0), (x, height), line_color, thickness)
        elif mode == 'medio':  # solo orizzontali
            cv2.line(img, (0, y), (width, y), line_color, thickness)
        elif mode == 'hard':  # entrambi
            cv2.line(img, (x, 0), (x, height), line_color, thickness)
            cv2.line(img, (0, y), (width, y), line_color, thickness)
    return img

# --- Effetto Barcode ---
def draw_barcode(frequencies, volume, width, height, bpm, frame_idx, fps, line_color):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    bar_spacing = 5
    max_bars = width // bar_spacing
    energies = np.interp(frequencies[:max_bars], [0, np.max(frequencies)], [0, 1])
    threshold = volume * 0.2
    # Calcola intervallo frame per seguire bpm
    beat_interval = int(fps / (bpm / 60)) if bpm > 0 else fps

    for i, e in enumerate(energies):
        if e < threshold:
            continue
        # spessore linee: alte freq linee sottili, basse più larghe
        bar_thickness = int(np.interp(i, [0, max_bars], [10, 1]))
        # fai "pulsare" linee larghe sul beat
        if frame_idx % beat_interval == 0 and bar_thickness > 5:
            bar_thickness = int(bar_thickness * 1.5)
        x = i * bar_spacing
        cv2.line(img, (x, 0), (x, height), line_color, bar_thickness)
    return img

def main():
    st.title("🎵 AudioLinee")

    # Scelta colori
    line_hex = st.color_picker("🎨 Colore linee", "#000000")
    bg_hex = st.color_picker("🖼️ Colore sfondo", "#FFFFFF")
    line_color = hex_to_bgr(line_hex)
    bg_color = tuple(int(bg_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    # Scelta FPS e effetto
    fps_option = st.selectbox("🎥 FPS video", [5, 10, 15, 24, 30, 60, 72], index=3)
    effect_type = st.selectbox("✨ Tipo di visualizzazione", ["Linee", "Spectrum", "Barcode"])
    spectrum_mode = None
    if effect_type == "Spectrum":
        spectrum_mode = st.selectbox("Modalità Spectrum", ["soft", "medio", "hard"])

    audio_file = st.file_uploader("🎧 Carica un file audio", type=["mp3", "wav"])

    if audio_file is not None:
        st.audio(audio_file)

        # Salva temporaneamente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=None)
        bpm = estimate_bpm(y, sr)
        try:
            bpm_display = f"{float(bpm):.1f}"
        except Exception:
            bpm_display = "Non stimato"

        st.info(f"🎵 BPM stimati: {bpm_display}, FPS impostati: {fps_option}")

        hop_length = 512
        S = np.abs(librosa.stft(y, hop_length=hop_length))
        volume = np.mean(S, axis=0)

        width, height = 1280, 720
        frames = []
        total_frames = min(len(volume), S.shape[1])

        for i in range(total_frames):
            freqs = S[:, i]
            vol = volume[i]
            if effect_type == "Linee":
                frame = draw_lines(freqs, vol, width, height, line_color)
            elif effect_type == "Spectrum":
                frame = draw_spectrum(freqs, vol, width, height, line_color, spectrum_mode)
            elif effect_type == "Barcode":
                frame = draw_barcode(freqs, vol, width, height, float(bpm) if bpm else 0, i, fps_option, line_color)
            else:
                frame = np.ones((height, width, 3), dtype=np.uint8) * 255  # sfondo bianco fallback
            frames.append(frame)

        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "output.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps_option, (width, height))

        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()

        st.video(video_path)
        st.success("✅ Video generato con successo!")

if __name__ == "__main__":
    main()
