import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import librosa

st.set_page_config(page_title="🎞️ AudioLinee", layout="wide")

# Colori
line_color = st.color_picker("🎨 Colore linee", "#000000")
background_color = st.color_picker("🖼️ Colore sfondo", "#FFFFFF")

# FPS
fps_option = st.selectbox("🎥 FPS del video", [5, 10, 15, 24, 30, 60, 72], index=3)

# Effetti
effect_type = st.selectbox("✨ Tipo visualizzazione", ["Linee", "Barcode", "Spectrum"])

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

def estimate_bpm(y, sr):
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return tempo
    except:
        return None

def draw_lines(freqs, volume, width, height):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    center = height // 2
    max_freq = len(freqs)
    for i, energy in enumerate(freqs):
        if energy < volume * 0.05:
            continue
        thickness = int(np.interp(energy, [0, np.max(freqs)], [1, 10]))
        x = int(np.interp(i, [0, max_freq], [0, width]))
        cv2.line(img, (x, center - 50), (x, center + 50), hex_to_bgr(line_color), thickness)
    return img

def draw_barcode(freqs, volume, width, height, bpm, frame_idx, fps):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    bar_spacing = 10
    max_bars = width // bar_spacing
    energies = np.interp(freqs[:max_bars], [0, np.max(freqs)], [0, 1])
    threshold = volume * 0.2
    beat_interval = int(fps / (bpm / 60)) if bpm and bpm > 0 else fps
    if frame_idx % max(1, beat_interval) != 0:
        return img
    for i, e in enumerate(energies):
        if e < threshold:
            continue
        bar_thickness = int(np.interp(i, [0, max_bars], [1, 10]))
        x = i * bar_spacing
        cv2.line(img, (x, 0), (x, height), hex_to_bgr(line_color), bar_thickness)
    return img

def draw_spectrum(freqs, volume, width, height):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    max_freq = len(freqs)
    for i, energy in enumerate(freqs):
        if energy < volume * 0.05:
            continue
        thickness = 2
        x = int(np.interp(i, [0, max_freq], [0, width]))
        y_top = height
        y_bottom = height - int(energy * height)
        cv2.line(img, (x, y_top), (x, y_bottom), hex_to_bgr(line_color), thickness)
    return img

def main():
    st.title("🎵 AudioLinee")

    uploaded_file = st.file_uploader("🎧 Carica un file audio (.wav o .mp3)", type=["wav", "mp3"])
    if not uploaded_file:
        st.info("Carica un file audio per iniziare.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path)
    bpm = estimate_bpm(y, sr)
    if bpm is None:
        bpm_display = "Non stimato"
    else:
        bpm_display = f"{bpm:.1f}"
    st.info(f"🎵 BPM stimati: {bpm_display}, FPS impostati: {fps_option}")

    hop_length = 512
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    volume = np.mean(S, axis=0)

    width, height = 1280, 720
    total_frames = len(volume)
    frames = []

    for i in range(total_frames):
        freqs = S[:, i]
        vol = volume[i]

        if effect_type == "Linee":
            frame = draw_lines(freqs, vol, width, height)
        elif effect_type == "Barcode":
            frame = draw_barcode(freqs, vol, width, height, bpm, i, fps_option)
        elif effect_type == "Spectrum":
            frame = draw_spectrum(freqs, vol, width, height)
        else:
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255

        frames.append(frame)

    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "output.mp4")

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_option, (width, height))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

    st.success("✅ Video generato con successo!")
    st.video(video_path)

if __name__ == "__main__":
    main()
