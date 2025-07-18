# 🎵 AudioLinee.py (by Loop507)
# Generatore di video visivi sincronizzati con l'audio
# Realizzato con Streamlit - Funziona online su Streamlit Cloud

import streamlit as st
import numpy as np
import cv2
import librosa
import matplotlib.cm as cm
import matplotlib.colors as colors
import os
import subprocess

# Configurazione pagina
st.set_page_config(page_title="🎵 AudioLinee - by Loop507", layout="centered")
st.title("🎨 AudioLinee")
st.markdown("### by Loop507")
st.markdown("Carica un file audio e genera un video visivo sincronizzato.")

def generate_video(format="1:1", level="medium", sync_audio=False):
    FORMAT_RESOLUTIONS = {
        "16:9": (1920, 1080),
        "9:16": (1080, 1920),
        "1:1": (1080, 1080),
        "4:3": (800, 600)
    }

    FRAME_WIDTH, FRAME_HEIGHT = FORMAT_RESOLUTIONS[format]
    FPS = 30
    DURATION_SECONDS = 10
    LEVEL = level
    TEMP_VIDEO = "temp_output.mp4"
    FINAL_VIDEO = "final_output.mp4"

    COLOR_GRADIENT = LEVEL in ["medium", "hard"]
    ROTATE_LINES = LEVEL == "hard"
    MULTI_BAND_FREQUENCIES = LEVEL == "hard"

    # Rimuovi file esistenti
    for f in [TEMP_VIDEO, FINAL_VIDEO]:
        if os.path.exists(f):
            os.remove(f)

    # Carica l'audio
    y, sr = librosa.load("input_audio.wav", sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(S, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

    # Inizializza il writer del video
    video_writer = cv2.VideoWriter(
        TEMP_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )

    # Crea una mappatura del colore (aggiornata)
    cmap = cm.get_cmap("plasma")
    norm = colors.Normalize(vmin=0, vmax=1)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    def energy_to_color(energy):
        rgba = scalar_map.to_rgba(energy)
        return tuple(int(x * 255) for x in rgba[:3])  # Solo RGB

    def generate_distorted_line(frame, x1, y1, x2, y2, distortion, energy=1.0, angle=0):
        num_points = 50
        x = np.linspace(x1, x2, num_points)
        y = np.linspace(y1, y2, num_points)

        # Distorsione sinusoidale
        y += distortion * np.sin(np.linspace(0, 2 * np.pi, num_points))

        # Rotazione
        if ROTATE_LINES:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            theta = np.radians(angle)
            x -= cx
            y -= cy
            xr = x * np.cos(theta) - y * np.sin(theta)
            yr = x * np.sin(theta) + y * np.cos(theta)
            x, y = xr + cx, yr + cy

        for i in range(num_points - 1):
            color = energy_to_color(energy) if COLOR_GRADIENT else (255, 255, 255)
            cv2.line(frame, (int(x[i]), int(y[i])), (int(x[i+1]), int(y[i+1])), color, 1)

    LINE_DENSITY = 50

    if MULTI_BAND_FREQUENCIES:
        bands = np.array_split(mel_spec_norm, 4)

    for frame_idx in range(FPS * DURATION_SECONDS):
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        time_index = min(int(frame_idx / (FPS * DURATION_SECONDS) * mel_spec_norm.shape[1]), mel_spec_norm.shape[1] - 1)

        # Linee verticali
        for i in range(LINE_DENSITY):
            y1 = i * (FRAME_HEIGHT / LINE_DENSITY)
            y2 = y1 + (FRAME_HEIGHT / LINE_DENSITY)
            x1, x2 = 0, FRAME_WIDTH

            if MULTI_BAND_FREQUENCIES:
                band_index = i % len(bands)
                col = bands[band_index][:, time_index]
                energy = np.mean(col)
                distortion = np.mean(col) * 50
            else:
                column = mel_spec_norm[:, time_index]
                energy = column[i % len(column)]
                distortion = energy * 50

            angle = int(energy * 30) if ROTATE_LINES else 0
            generate_distorted_line(frame, x1, y1, x2, y2, distortion, energy, angle)

        # Linee orizzontali
        for j in range(LINE_DENSITY):
            x1 = j * (FRAME_WIDTH / LINE_DENSITY)
            x2 = x1 + (FRAME_WIDTH / LINE_DENSITY)
            y1, y2 = 0, FRAME_HEIGHT

            if MULTI_BAND_FREQUENCIES:
                band_index = j % len(bands)
                r = bands[band_index][time_index, :]
                energy = np.mean(r)
                distortion = np.mean(r) * 50
            else:
                row = mel_spec_norm[time_index, :]
                energy = row[j % len(row)]
                distortion = energy * 50

            angle = int(energy * 30) if ROTATE_LINES else 0
            generate_distorted_line(frame, x1, y1, x2, y2, distortion, energy, angle)

        # Effetto crescita/diminuzione
        energy_frame = np.mean(mel_spec_norm[:, time_index])
        scale_factor = 1 + 0.1 * (energy_frame - 0.5)
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # Centra il frame ridimensionato
        new_h, new_w = frame.shape[:2]
        top = max(0, (FRAME_HEIGHT - new_h) // 2)
        left = max(0, (FRAME_WIDTH - new_w) // 2)
        frame = cv2.copyMakeBorder(
            frame,
            top=top,
            bottom=FRAME_HEIGHT - new_h - top,
            left=left,
            right=FRAME_WIDTH - new_w - left,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        video_writer.write(frame)

    video_writer.release()

    # Sincronizza l'audio al video se richiesto
    if sync_audio:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", TEMP_VIDEO,
            "-i", "input_audio.wav",
            "-c:v", "copy",
            "-c:a", "aac",
            "-strict", "experimental",
            FINAL_VIDEO
        ]
        subprocess.run(cmd)
        os.remove(TEMP_VIDEO)
    else:
        os.rename(TEMP_VIDEO, FINAL_VIDEO)

    print("✅ Video generato con successo!")

# Interfaccia utente
uploaded_file = st.file_uploader("🎧 Carica un file audio (.wav o .mp3)", type=["wav", "mp3"])
if uploaded_file is not None:
    with open("input_audio.wav", "wb") as f:
        f.write(uploaded_file.read())
    st.success("🔊 Audio caricato correttamente!")

    col1, col2 = st.columns(2)
    with col1:
        video_format = st.selectbox("📐 Formato video", ["16:9", "9:16", "1:1", "4:3"])
    with col2:
        effect_level = st.selectbox("🎨 Livello effetti", ["soft", "medium", "hard"])

    sync_audio = st.checkbox("🔊 Sincronizza l'audio nel video")

    if st.button("🎬 Genera Video"):
        with st.spinner("⏳ Sto generando il video... Questo potrebbe richiedere alcuni minuti."):
            generate_video(format=video_format, level=effect_level, sync_audio=sync_audio)

        if os.path.exists("final_output.mp4"):
            with open("final_output.mp4", "rb") as f:
                st.download_button("⬇️ Scarica il video", f, file_name="output_video.mp4")
        else:
            st.error("❌ Si è verificato un errore nella generazione del video.")
