# 🎵 AudioLinee.py (by Loop507) - Versione aggiornata con Barcode e FPS personalizzabili

import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import librosa
from PIL import Image
from scipy.signal import find_peaks

st.set_page_config(page_title="🎞️ AudioLinee", layout="wide")

# --- Impostazioni colore ---
line_color = st.color_picker("🎨 Colore delle linee", "#000000")
background_color = st.color_picker("🎼 Colore dello sfondo", "#FFFFFF")

# --- Impostazioni FPS ---
fps_option = st.selectbox("🎥 FPS del video", [5, 10, 15, 24, 30, 60, 72], index=3)

# --- Modalità di visualizzazione ---
effect_type = st.selectbox("✨ Tipo di visualizzazione", ["Linee", "Barcode"])

# --- Funzione per calcolo BPM ---
def estimate_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo

# --- Funzione per disegnare linee ---
def draw_lines(frequencies, volume, width, height):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    center = height // 2
    max_freq = len(frequencies)
    for i, energy in enumerate(frequencies):
        if energy < volume * 0.05:
            continue
        thickness = int(np.interp(energy, [0, np.max(frequencies)], [1, 10]))
        x = int(np.interp(i, [0, max_freq], [0, width]))
        cv2.line(img, (x, center - 50), (x, center + 50), hex_to_bgr(line_color), thickness)
    return img

# --- Funzione per disegnare barcode ---
def draw_barcode(frequencies, volume, width, height, bpm, frame_idx, fps):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    bar_spacing = 10
    max_bars = width // bar_spacing
    energies = np.interp(frequencies[:max_bars], [0, np.max(frequencies)], [0, 1])
    threshold = volume * 0.2
    beat_interval = int(fps / (bpm / 60)) if bpm > 0 else fps
    if frame_idx % beat_interval != 0:
        return img
    for i, e in enumerate(energies):
        if e < threshold:
            continue
        bar_thickness = int(np.interp(i, [0, max_bars], [1, 10]))
        x = i * bar_spacing
        cv2.line(img, (x, 0), (x, height), hex_to_bgr(line_color), bar_thickness)
    return img

# --- Utilità colore ---
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

# --- MAIN ---
def main():
    st.title("🎵 AudioLinee")
    audio_file = st.file_uploader("🏋️ Carica un file audio", type=[".mp3", ".wav"])

    if audio_file:
        st.audio(audio_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path)
        bpm = estimate_bpm(y, sr)
        st.info(f"🎵 BPM stimati: {bpm:.1f}, FPS impostati: {fps_option}")

        hop_length = 512
        S = np.abs(librosa.stft(y, hop_length=hop_length))
        volume = np.mean(S, axis=0)

        frames = []
        total_frames = len(volume)
        width, height = 1280, 720

        for i in range(total_frames):
            freqs = S[:, i]
            vol = volume[i]
            if effect_type == "Linee":
                frame = draw_lines(freqs, vol, width, height)
            elif effect_type == "Barcode":
                frame = draw_barcode(freqs, vol, width, height, bpm, i, fps_option)
            frames.append(frame)

        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "output.mp4")

        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_option, (width, height))
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()

        st.video(video_path)
        st.success("✅ Video generato con successo!")

if __name__ == "__main__":
    main()
