# pages/1_Détection_Faciale.py
"""
Scénario 2 : Détection faciale avec validation/correction (Streamlit)

Flux :
1) Détection via webcam (DeepFace côté facial_utils)
2) Validation ou correction textuelle
3) Reco Spotify à partir de l'émotion finale (embeds)

Logs JSONL : logs/playlist_log.json, logs/feedback_log.json
"""

import os
import json
import time
import streamlit as st

# ⚠️ On s'appuie sur ces modules EXISTANTS dans ton repo
# - emotion_detection/facial_utils.py  -> detect_facial_emotion(...)
# - emotion_detection/text_utils.py    -> detect_emotion_from_text(...)
# - recommender/playlist_generator.py  -> generate_playlist(emotion, n_tracks)
import emotion_detection.facial_utils as facial_utils
from emotion_detection.text_utils import detect_emotion_from_text
from recommender.playlist_generator import generate_playlist

# -------------------------------
# Préparation logs
# -------------------------------
LOG_DIR = "logs"
PLAYLIST_LOG = os.path.join(LOG_DIR, "playlist_log.json")
FEEDBACK_LOG = os.path.join(LOG_DIR, "feedback_log.json")
os.makedirs(LOG_DIR, exist_ok=True)


def append_json_line(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -------------------------------
# UI
# -------------------------------
st.title("😊 Scénario 2 : Détection Faciale avec Validation")
st.write(
    "La caméra détecte une émotion ; tu peux la valider ou la corriger en texte. "
    "La recommandation Spotify est générée à partir de l’émotion finale."
)

with st.expander("⚙️ Paramètres de détection", expanded=False):
    duration = st.slider("Durée d'analyse (sec)", 5, 15, 10, 1)
    analyze_every = st.slider("Intervalle d’analyse (sec)", 1, 5, 2, 1)
    camera_index = st.number_input("Index caméra", min_value=0, value=0, step=1)

preview_placeholder = st.empty()

# -------------------------------
# Session state
# -------------------------------
st.session_state.setdefault("face_label", None)
st.session_state.setdefault("final_emotion", None)
st.session_state.setdefault("spotify_uris", [])

colA, colB = st.columns(2)

with colA:
    if st.button("🎥 Lancer la détection", type="primary"):
        with st.spinner(f"Détection en cours (~{duration}s)…"):
            # facial_utils.detect_facial_emotion doit renvoyer (label, grouped, extras)
            label, grouped, extras = facial_utils.detect_facial_emotion(
                duration=duration,
                st_placeholder=preview_placeholder,
                analyze_every=analyze_every,
                record_video=False,
                camera_index=camera_index,
            )
        st.session_state["face_label"] = label
        if label:
            st.success(f"Émotion détectée : **{label}**")
        else:
            st.warning(
                "Aucune émotion fiable détectée. Essaie à nouveau (luminosité, cadrage)."
            )

with colB:
    if st.button("♻️ Réinitialiser"):
        for k in ("face_label", "final_emotion", "spotify_uris"):
            st.session_state[k] = None if k != "spotify_uris" else []
        st.rerun()

# -------------------------------
# Étape 2 : Validation / correction
# -------------------------------
if st.session_state["face_label"] and not st.session_state["final_emotion"]:
    st.subheader("2) Valider ou corriger l'émotion détectée")
    st.write(f"Émotion détectée : **{st.session_state['face_label']}**")

    user_text = st.text_input(
        "Si l'émotion est incorrecte, écris une courte phrase (ex. « je ne vais pas bien »)",
        placeholder="Ex: je me sens triste…",
    )

    if st.button("✅ Valider l’émotion finale"):
        final = st.session_state["face_label"]
        if user_text and user_text.strip():
            # Détection NLP pour correction
            final = detect_emotion_from_text(user_text.strip()) or final

        st.session_state["final_emotion"] = final
        st.success(f"Émotion finale : **{final}**")

# -------------------------------
# Étape 3 : Recommandation Spotify
# -------------------------------
if st.session_state.final_emotion:
    st.subheader("3) Génération de la playlist Spotify")
    with st.spinner("🎧 Recherche de morceaux…"):
        tracks = generate_playlist(st.session_state.final_emotion, n_tracks=12)

    if not tracks:
        st.error("Aucune recommandation Spotify n’a été trouvée (même après fallback).")
    else:
        # Extraire des URIs, qu'elles viennent de recommendations ou de playlist_items
        uris = [t.get("uri") or t.get("track", {}).get("uri") for t in tracks if t]
        uris = [u for u in uris if u]
        st.write("**Aperçu :**")
        for uri in uris[:5]:
            st.components.v1.iframe(
                f"https://open.spotify.com/embed/track/{uri.split(':')[-1]}", height=80
            )

        # Log minimal (tu peux garder ton append_json_line)
        append_json_line(
            PLAYLIST_LOG,
            {
                "ts": int(time.time()),
                "emotion": st.session_state.final_emotion,
                "source": "validation_correction",
                "uris": uris,
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
                        "uris": uris,
                        "feedback": "like",
                        "scenario": "validation_correction",
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
                        "uris": uris,
                        "feedback": "dislike",
                        "scenario": "validation_correction",
                    },
                )
                st.toast("Merci pour le feedback 👎")
