"""
Page Streamlit pour le scénario 1 : détection parallèle et fusion pondérée.

Cette page combine la détection d'expressions faciales et l'analyse
textuelle en parallèle. L'utilisateur peut ajuster le poids accordé à
chaque modalité avant de fusionner les deux. La playlist générée
reflète la distribution fusionnée.
"""

import os
import json
import time
import streamlit as st

from emotion_detection.facial_utils import detect_facial_emotion
from emotion_detection.text_utils import emotion_distribution, detect_emotion_from_text
from emotion_detection.multimodal_fusion import fuse_distributions, dominant_emotion
from recommender.playlist_generator import generate_playlist

# Préparer les fichiers de log
LOG_DIR = "logs"
PLAYLIST_LOG = os.path.join(LOG_DIR, "playlist_log.json")
FEEDBACK_LOG = os.path.join(LOG_DIR, "feedback_log.json")
os.makedirs(LOG_DIR, exist_ok=True)


def append_json_line(path: str, obj: dict) -> None:
    """Append a JSON object as a single line to the specified file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


st.title("🔀 Scénario 1 : Détection Parallèle avec Fusion Pondérée")
st.write(
    "Détection simultanée des émotions faciales et textuelles.\n"
    "Ajustez les poids pour chaque modalité et fusionnez les résultats pour obtenir l'émotion finale."
)

with st.expander("⚙️ Paramètres de détection", expanded=False):
    duration = st.slider("Durée d'analyse (sec)", 5, 15, 10, 1)
    analyze_every = st.slider("Intervalle d'analyse (sec)", 1, 5, 2, 1)
    camera_index = st.number_input("Index caméra", min_value=0, value=0, step=1)

placeholder = st.empty()

if "face_label" not in st.session_state:
    st.session_state.face_label = None
if "final_emotion" not in st.session_state:
    st.session_state.final_emotion = None
if "spotify_url" not in st.session_state:
    st.session_state.spotify_url = None

colA, colB = st.columns(2)
with colA:
    if st.button("🎥 Lancer la détection faciale", type="primary"):
        with st.spinner(f"Détection en cours ({duration}s)…"):
            label, probs, extras = detect_facial_emotion(
                duration=duration,
                st_placeholder=placeholder,
                analyze_every=analyze_every,
                record_video=False,
                camera_index=camera_index,
            )
        st.session_state.face_label = label
        st.success(f"Émotion faciale détectée : **{label}**")
with colB:
    if st.button("♻️ Réinitialiser"):
        for k in ["face_label", "final_emotion", "spotify_url"]:
            st.session_state[k] = None
        st.rerun()

if st.session_state.face_label:
    st.subheader("2) Entrée textuelle et pondération")
    st.caption(
        "Décris ton humeur en quelques mots et ajuste les poids entre la détection faciale et la détection textuelle."
    )
    # Poids pour la fusion : slider de 0 à 100
    face_weight_percent = st.slider(
        "Poids de l'émotion faciale (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        help="Répartissez l'importance entre l'émotion faciale et l'émotion textuelle",
    )
    face_weight = face_weight_percent / 100.0
    text_weight = 1.0 - face_weight
    user_text = st.text_area(
        "💬 Saisie libre (facultative)",
        placeholder="Ex: je me sens triste mais motivé…",
    )
    if st.button("✅ Fusionner et valider l'émotion finale"):
        face_probs = {st.session_state.face_label: 1.0}
        if user_text and user_text.strip():
            txt_dist = emotion_distribution(user_text)
            fused_dist = fuse_distributions(
                face_probs,
                txt_dist,
                face_weight=face_weight,
                text_weight=text_weight,
            )
            final, prob = dominant_emotion(fused_dist)
        else:
            # Pas de texte fourni → conserver l'émotion faciale
            final = st.session_state.face_label
        st.session_state.final_emotion = final
        st.success(f"Émotion finale : **{final}**")

    if st.session_state.final_emotion:
        st.subheader("3) Génération de la playlist Spotify")
        with st.spinner("🎧 Création de la playlist…"):
            url = generate_playlist(st.session_state.final_emotion)
        st.session_state.spotify_url = url
        if url:
            st.link_button("▶️ Écouter la playlist", url)
            append_json_line(
                PLAYLIST_LOG,
                {
                    "ts": int(time.time()),
                    "emotion": st.session_state.final_emotion,
                    "source": "multimodal_weighted",
                    "face_weight": face_weight,
                    "text_weight": text_weight,
                    "url": url,
                },
            )
            st.subheader("4) Feedback")
            fb_cols = st.columns(2)
            with fb_cols[0]:
                if st.button("👍 J'aime"):
                    append_json_line(
                        FEEDBACK_LOG,
                        {
                            "ts": int(time.time()),
                            "emotion": st.session_state.final_emotion,
                            "playlist_url": st.session_state.spotify_url,
                            "feedback": "like",
                            "scenario": "multimodal_weighted",
                        },
                    )
                    st.toast("Merci pour le feedback 👍")
            with fb_cols[1]:
                if st.button("👎 J'aime pas"):
                    append_json_line(
                        FEEDBACK_LOG,
                        {
                            "ts": int(time.time()),
                            "emotion": st.session_state.final_emotion,
                            "playlist_url": st.session_state.spotify_url,
                            "feedback": "dislike",
                            "scenario": "multimodal_weighted",
                        },
                    )
                    st.toast("Merci pour le feedback 👎")
        else:
            st.info("Impossible de générer une playlist pour le moment.")
