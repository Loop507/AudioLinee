import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import librosa
from PIL import Image

st.set_page_config(page_title="🎞️ AudioLinee", layout="wide")

# --- Impostazioni colore ---
line_color = st.sidebar.color_picker("🎨 Colore delle linee", "#000000")
background_color = st.sidebar.color_picker("🖼️ Colore dello sfondo", "#FFFFFF")

# --- Impostazioni FPS ---
fps_option = st.sidebar.selectbox("🎥 FPS del video", [5, 10, 15, 24, 30, 60, 72], index=3)

# --- Modalità di visualizzazione ---
effect_type = st.sidebar.selectbox("✨ Tipo di visualizzazione", ["Linee", "Spectrum", "Barcode"])

# --- Funzione per calcolo BPM ---
def estimate_bpm(y, sr):
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return tempo
    except:
        return None

# --- Funzioni di disegno ---

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

def draw_lines(frequencies, volume, width, height):
    img = np.ones((height, width, 3), dtype=np.uint8)
    img[:] = np.array(hex_to_bgr(background_color), dtype=np.uint8)
    center = height // 2
    max_freq = len(frequencies)
    for i, energy in enumerate(frequencies):
        if energy < volume * 0.05:
            continue
        thickness = int(np.interp(energy, [0, np.max(frequencies)], [1, 10]))
        x = int(np.interp(i, [0, max_freq], [0, width]))
        cv2.line(img, (x, center - 50), (x, center + 50), hex_to_bgr(line_color), thickness)
    return img

def draw_spectrum(frequencies, volume, width, height):
    img = np.ones((height, width, 3), dtype=np.uint8)
    img[:] = np.array(hex_to_bgr(background_color), dtype=np.uint8)
    max_freq = len(frequencies)
    for i, energy in enumerate(frequencies):
        if energy < volume * 0.05:
            continue
        thickness = int(np.interp(energy, [0, np.max(frequencies)], [1, 5]))
        # Vertical lines
        x = int(np.interp(i, [0, max_freq], [0, width]))
        cv2.line(img, (x, height), (x, height - int(energy * height * 2)), hex_to_bgr(line_color), thickness)
        # Horizontal lines
        y = int(np.interp(i, [0, max_freq], [0, height]))
        cv2.line(img, (0, y), (int(energy * width * 2), y), hex_to_bgr(line_color), thickness)
    return img

def draw_barcode(frequencies, volume, width, height, bpm, frame_idx, fps):
    img = np.ones((height, width, 3), dtype=np.uint8)
    img[:] = np.array(hex_to_bgr(background_color), dtype=np.uint8)
    bar_spacing = 5
    max_bars = width // bar_spacing
    energies = np.interp(frequencies[:max_bars], [0, np.max(frequencies)], [0, 1])
    threshold = volume * 0.2
    beat_interval = int(fps / (bpm / 60)) if bpm and bpm > 0 else fps
    beat_active = (frame_idx % beat_interval) == 0

    for i, e in enumerate(energies):
        if e < threshold:
            continue
        thickness = int(np.interp(i, [0, max_bars], [1, 15]))
        if beat_active:
            thickness = int(thickness * 1.5)  # ingrandisci con il bpm
        x = i * bar_spacing
        cv2.line(img, (x, 0), (x, height), hex_to_bgr(line_color), thickness)
    return img

def generate_video(frames, fps, width, height):
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    return video_path

def main():
    st.title("🎵 AudioLinee")

    audio_file = st.file_uploader("🎧 Carica un file audio (.wav o .mp3)", type=["wav", "mp3"])
    if not audio_file:
        st.info("Carica un file audio per iniziare.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=
