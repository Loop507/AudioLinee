# 🎵 AudioLinee.py (by Loop507) - Versione aggiornata e corretta

import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import librosa
from PIL import Image

st.set_page_config(page_title="🎞️ AudioLinee", layout="wide")

# --- Funzione per stimare BPM ---
def estimate_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo

# --- Funzione per convertire colore HEX a BGR OpenCV ---
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    # OpenCV usa BGR
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

# --- Funzione per disegnare le linee (effetto Linee) ---
def draw_lines(frequencies, volume, width, height, line_color_bgr):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # sfondo bianco
    center = height // 2
    max_freq = len(frequencies)
    max_energy = np.max(frequencies) if np.max(frequencies) > 0 else 1
    for i, energy in enumerate(frequencies):
        if energy < volume * 0.05:
            continue
        thickness = int(np.interp(energy, [0, max_energy], [1, 10]))
        x = int(np.interp(i, [0, max_freq], [0, width]))
        cv2.line(img, (x, center - 50), (x, center + 50), line_color_bgr, thickness)
    return img

# --- Funzione per disegnare lo spectrum (linee verticali/orizzontali o entrambe) ---
def draw_spectrum(frequencies, volume, width, height, line_color_bgr, mode="soft"):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    max_freq = len(frequencies)
    max_energy = np.max(frequencies) if np.max(frequencies) > 0 else 1

    if mode == "soft":
        # Solo linee verticali
        for i, energy in enumerate(frequencies):
            if energy < volume * 0.05:
                continue
            thickness = int(np.interp(energy, [0, max_energy], [1, 5]))
            x = int(np.interp(i, [0, max_freq], [0, width]))
            line_length = int(np.interp(energy, [0, max_energy], [10, height//2]))
            cv2.line(img, (x, height//2 - line_length), (x, height//2 + line_length), line_color_bgr, thickness)
    elif mode == "medium":
        # Solo linee orizzontali
        for i, energy in enumerate(frequencies):
            if energy < volume * 0.05:
                continue
            thickness = int(np.interp(energy, [0, max_energy], [1, 5]))
            y = int(np.interp(i, [0, max_freq], [0, height]))
            line_length = int(np.interp(energy, [0, max_energy], [10, width//2]))
            cv2.line(img, (width//2 - line_length, y), (width//2 + line_length, y), line_color_bgr, thickness)
    else:  # hard
        # linee verticali + orizzontali
        for i, energy in enumerate(frequencies):
            if energy < volume * 0.05:
                continue
            thickness = int(np.interp(energy, [0, max_energy], [1, 5]))
            x = int(np.interp(i, [0, max_freq], [0, width]))
            y = int(np.interp(i, [0, max_freq], [0, height]))
            line_length_v = int(np.interp(energy, [0, max_energy], [10, height//4]))
            line_length_h = int(np.interp(energy, [0, max_energy], [10, width//4]))
            # verticale
            cv2.line(img, (x, height//2 - line_length_v), (x, height//2 + line_length_v), line_color_bgr, thickness)
            # orizzontale
            cv2.line(img, (width//2 - line_length_h, y), (width//2 + line_length_h, y), line_color_bgr, thickness)
    return img

# --- Funzione per disegnare barcode ---
def draw_barcode(frequencies, volume, width, height, bpm, frame_idx, fps, line_color_bgr):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    bar_spacing = 5
    max_bars = width // bar_spacing
    energies = np.interp(frequencies[:max_bars], [0, np.max(frequencies)], [0, 1])
    threshold = volume * 0.2
    beat_interval = int(fps / (bpm / 60)) if bpm > 0 else fps
    # Per far "battere" le barre con i bpm, alterniamo visibilità ogni beat_interval frame
    visible = (frame_idx % beat_interval) < (beat_interval // 2) if beat_interval > 0 else True
    if not visible:
        return img
    for i, e in enumerate(energies):
        if e < threshold:
            continue
        # Spessore linee: alte freq linee sottili, medie medie, basse spesse
        if i < max_bars // 3:
            thickness = int(np.interp(e, [0, 1], [7, 15]))  # basse frequenze, linee larghe
        elif i < 2 * max_bars // 3:
            thickness = int(np.interp(e, [0, 1], [3, 7]))  # medie frequenze, linee medie
        else:
            thickness = int(np.interp(e, [0, 1], [1, 3]))  # alte frequenze, linee sottili
        x = i * bar_spacing
        cv2.line(img, (x, 0), (x, height), line_color_bgr, thickness)
    return img

def main():
    st.title("🎵 AudioLinee")

    # Selettori colori e impostazioni
    line_color = st.color_picker("🎨 Colore linee", "#000000")
    background_color = st.color_picker("🖼️ Colore sfondo", "#FFFFFF")
    fps_option = st.selectbox("🎥 FPS video", [5, 10, 15, 24, 30, 60, 72], index=3)
    effect_type = st.selectbox("✨ Tipo effetto", ["Linee", "Spectrum", "Barcode"])
    spectrum_mode = None
    if effect_type == "Spectrum":
        spectrum_mode = st.selectbox("Modalità Spectrum", ["soft", "medium", "hard"])

    audio_file = st.file_uploader("🎧 Carica un file audio (.wav, .mp3)", type=["wav", "mp3"])
    if not audio_file:
        st.info("Carica un file audio per iniziare.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path)
    bpm = estimate_bpm(y, sr)
    st.info(f"🎵 BPM stimati: {bpm:.1f}, FPS impostati: {fps_option}")

    # Calcolo spettrogramma e volume
    hop_length = 512
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    volume = np.mean(S, axis=0)

    frames = []
    total_frames = min(len(volume), 300)  # Limito a 300 frame per durata max, puoi aumentare o rimuovere
    width, height = 1280, 720
    line_color_bgr = hex_to_bgr(line_color)
    background_color_bgr = hex_to_bgr(background_color)

    for i in range(total_frames):
        freqs = S[:, i]
        vol = volume[i]
        if effect_type == "Linee":
            frame = draw_lines(freqs, vol, width, height, line_color_bgr)
        elif effect_type == "Spectrum":
            frame = draw_spectrum(freqs, vol, width, height, line_color_bgr, spectrum_mode)
        else:  # Barcode
            frame = draw_barcode(freqs, vol, width, height, bpm, i, fps_option, line_color_bgr)

        # Applico sfondo colore
        bg_img = np.ones_like(frame, dtype=np.uint8) * np.array(background_color_bgr, dtype=np.uint8)
        frame = cv2.addWeighted(bg_img, 1, frame, 1, 0)
        frames.append(frame)

    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "output.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps_option, (width, height))
    for frame in frames:
        # OpenCV usa BGR, frame è già BGR
        out.write(frame)
    out.release()

    st.video(video_path)
    st.success("✅ Video generato con successo!")

if __name__ == "__main__":
    main()
