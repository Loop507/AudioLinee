# 🎵 AudioLinee - by Loop507 - Versione Estesa con Effetti e Colori per Frequenze

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

def generate_melspectrogram(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
        if S.size == 0:
            st.error("❌ Impossibile generare lo spettrogramma: l'audio è troppo breve o non è valido.")
            return None
        mel_spec_db = librosa.power_to_db(S, ref=np.max)
        min_val = mel_spec_db.min()
        max_val = mel_spec_db.max()
        if max_val == min_val:
            st.error("❌ L'audio non contiene variazioni sufficienti per generare il video.")
            return None
        mel_spec_norm = (mel_spec_db - min_val) / (max_val - min_val)
        if mel_spec_norm.shape[1] == 0:
            st.error("❌ Lo spettrogramma è vuoto: l'audio è troppo breve o non contiene dati validi.")
            return None
        return mel_spec_norm
    except Exception as e:
        st.error(f"❌ Errore nella generazione dello spettrogramma: {str(e)}")
        return None

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    rgb = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (rgb[2], rgb[1], rgb[0])

# Interfaccia principale

def main():
    from Audiolinee_Colorfx import VideoGenerator

    FORMAT_RESOLUTIONS = {
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "1:1": (720, 720),
        "4:3": (800, 600)
    }

    st.set_page_config(page_title="🎵 AudioLinee - by Loop507", layout="centered")
    st.title("🎨 AudioLinee")
    st.markdown("### by Loop507")
    st.markdown("Carica un file audio e genera un video visivo sincronizzato.")

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

        with st.spinner("📊 Analisi audio in corso..."):
            mel_spec_norm = generate_melspectrogram(y, sr)
        if mel_spec_norm is None:
            return
        st.success("✅ Analisi audio completata!")

        col1, col2, col3 = st.columns(3)
        with col1:
            video_format = st.selectbox("📐 Formato video", list(FORMAT_RESOLUTIONS.keys()))
        with col2:
            effect_level = st.selectbox("🎨 Livello effetti", ["soft", "medium", "hard"])
        with col3:
            fps_choice = st.selectbox("🎞️ Fotogrammi al secondo (FPS)", [10, 15, 24, 30], index=3)

        bg_color_hex = st.color_picker("🎨 Colore sfondo", "#FFFFFF")
        line_color_hex = st.color_picker("🎨 Colore linee generiche", "#000000")

        col_low, col_mid, col_high = st.columns(3)
        with col_low:
            low_color_hex = st.color_picker("⬇️ Basse Frequenze", "#000000")
        with col_mid:
            mid_color_hex = st.color_picker("↔️ Medie Frequenze", "#FF0000")
        with col_high:
            high_color_hex = st.color_picker("⬆️ Alte Frequenze", "#0000FF")

        effect_type = st.selectbox("✨ Tipo di effetto visivo", ["Rete di Linee", "Linee Spezzate", "Linee Esplosive"])

        sync_audio = st.checkbox("🔊 Sincronizza l'audio nel video")
        if not check_ffmpeg():
            st.warning("⚠️ FFmpeg non disponibile - La sincronizzazione audio è disabilitata")
            sync_audio = False

        if st.button("🎬 Genera Video"):
            bg_color_bgr = hex_to_bgr(bg_color_hex)
            line_color_bgr = hex_to_bgr(line_color_hex)
            low_color_bgr = hex_to_bgr(low_color_hex)
            mid_color_bgr = hex_to_bgr(mid_color_hex)
            high_color_bgr = hex_to_bgr(high_color_hex)

            generator = VideoGenerator(
                format_res=FORMAT_RESOLUTIONS[video_format],
                level=effect_level,
                fps=fps_choice,
                bg_color=bg_color_bgr,
                line_color=line_color_bgr,
                low_color=low_color_bgr,
                mid_color=mid_color_bgr,
                high_color=high_color_bgr
            )

            success = generator.generate_video(mel_spec_norm, audio_duration, sync_audio, effect_type=effect_type)
            if success and os.path.exists("final_output.mp4"):
                with open("final_output.mp4", "rb") as f:
                    st.download_button("⬇️ Scarica il video", f, file_name=f"audiolinee_{video_format}_{effect_level}.mp4", mime="video/mp4")
                file_size = os.path.getsize("final_output.mp4")
                st.info(f"📁 Dimensione file: {file_size / 1024 / 1024:.1f} MB")
            else:
                st.error("❌ Si è verificato un errore nella generazione del video.")

        if st.button("🧹 Pulisci file temporanei"):
            temp_files = ["input_audio.wav", "temp_output.mp4", "final_output.mp4"]
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
            st.success("✅ File temporanei eliminati!")

if __name__ == "__main__":
    main()
