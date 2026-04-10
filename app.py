# --- Réduction logs bas niveau ---
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("GLOG_minloglevel", "4")
os.environ.setdefault("GLOG_logtostderr", "1")
os.environ.setdefault("ABSL_LOG_SEVERITY", "error")

# --- Arrêt propre de la boucle asyncio ---
import atexit, asyncio
def _close_asyncio_loop():
    try:
        loop = asyncio.get_event_loop()
    except Exception:
        return
    try:
        if loop.is_running():
            loop.call_soon_threadsafe(lambda: None)
            loop.stop()
        if not loop.is_closed():
            loop.close()
    except Exception:
        pass
atexit.register(_close_asyncio_loop)

import keras  # noqa: F401 — doit précéder tout import DeepFace/mtcnn pour enregistrer tensorflow.keras
import time, logging, contextlib, sys, threading, urllib.parse
from typing import Dict, List, Optional
from collections import deque

import streamlit as st
import random

# IMPORTANT : set_page_config doit être le tout premier appel Streamlit
st.set_page_config(
    page_title="Face2Melody V3",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

from spotipy import Spotify, SpotifyException
logger = logging.getLogger(__name__)
from feedback_learning import load_feedback_stats, rerank_tracks_with_feedback

def get_log_path(pid: str) -> str:
    return f"logs/experiment_{pid}.jsonl"

from recommender.emotion_detector import get_status, get_backend, is_ready
from emotion_detection.text_utils import detect_emotion_from_text, emotion_distribution

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    if not os.getenv("SPOTIPY_CLIENT_ID") or not os.getenv("SPOTIPY_REDIRECT_URI"):
        raise RuntimeError("Variables SPOTIPY_CLIENT_ID / SPOTIPY_REDIRECT_URI manquantes.")
except Exception:
    pass

logging.getLogger("spotipy.client").setLevel(logging.ERROR)
logging.getLogger("spotipy.oauth2").setLevel(logging.INFO)
logging.getLogger("spotipy.client").propagate = False
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)

import cv2, numpy as np
try:
    cv2.setNumThreads(1)
except Exception:
    pass

from recommender.emotion_detector import analyze_emotion, is_ready, get_status, reload_deepface
from PIL import Image
from recommender.spotify_interface import SpotifyInterface
from recommender.lastfm_interface import LastFMInterface, lastfm_available
from experiment_utils import (
    fuse, build_user_profile, recommend_for_emotion, append_log_line,
    audio_features_by_uri, now_iso, set_sp_global, set_cache_salt,
)

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity("error")
except Exception:
    pass

SHOW_DEEPFACE_UI = False
try:
    from analytics_gen import render as analytics_render
except Exception:
    analytics_render = None

# ──────────────────────────────────────────────────────────────────────────────
# Constantes & détection
# ──────────────────────────────────────────────────────────────────────────────

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
FRAME_INTERVAL_DEEPFACE = 1

DEFAULT_TARGETS = {
    "happy":   {"val": 0.85, "eng": 0.75},
    "sad":     {"val": 0.20, "eng": 0.35},
    "angry":   {"val": 0.30, "eng": 0.90},
    "neutral": {"val": 0.55, "eng": 0.50},
}
LABEL_SAFE_GENRES = {
    "happy":   ["dance", "pop", "electronic"],
    "sad":     ["indie", "chill", "pop"],
    "angry":   ["rock", "electronic", "indie"],
    "neutral": ["pop", "indie", "dance"],
}
REQUIRED_SCOPES = [
    "user-top-read",
    "playlist-modify-private",
    "playlist-modify-public",
    "user-library-read",
]
SCOPE = " ".join(REQUIRED_SCOPES)

# ──────────────────────────────────────────────────────────────────────────────
# Injection CSS — thème sombre professionnel
# ──────────────────────────────────────────────────────────────────────────────

_FONTS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
"""

_CSS = """
<style>
/* ══ Cyber Violet — Face2Melody V3 Design System ══════════════════════════ */

/* Variables */
:root {
    --bg-primary:    #07071a;
    --bg-card:       rgba(139, 92, 246, 0.06);
    --accent-purple: #8b5cf6;
    --accent-cyan:   #06b6d4;
    --accent-gold:   #f59e0b;
    --glow-purple:   rgba(139, 92, 246, 0.35);
    --glow-cyan:     rgba(6, 182, 212, 0.35);
    --text-muted:    #94a3b8;
    --emotion-happy:   #f59e0b;
    --emotion-sad:     #3b82f6;
    --emotion-angry:   #ef4444;
    --emotion-neutral: #10b981;
}

/* Animations */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 8px var(--glow-purple), 0 0 16px var(--glow-purple); }
    50%       { box-shadow: 0 0 20px var(--glow-cyan),  0 0 40px var(--glow-cyan); }
}
@keyframes waveform {
    0%, 100% { transform: scaleY(0.4); opacity: 0.6; }
    50%       { transform: scaleY(1.0); opacity: 1.0; }
}
@keyframes heartbeat {
    0%, 100% { transform: scale(1.0); }
    14%       { transform: scale(1.15); }
    28%       { transform: scale(1.0); }
    42%       { transform: scale(1.12); }
}
@keyframes scan-line {
    0%   { top: 0%; opacity: 0.4; }
    100% { top: 100%; opacity: 0; }
}

/* ── Titre principal ──────────────────────────────────────────────────────── */
.f2m-title {
    font-family: 'Orbitron', 'Arial Black', sans-serif;
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #a78bfa 0%, #06b6d4 60%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.15rem;
    letter-spacing: 0.06em;
    text-shadow: none;
    filter: drop-shadow(0 0 18px rgba(139,92,246,0.5));
}
.f2m-subtitle {
    font-family: 'Inter', sans-serif;
    color: #64748b;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

/* ── Cards chat messages ─────────────────────────────────────────────────── */
.chat-card-assistant {
    background: var(--bg-card);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), 0 0 0 1px rgba(139,92,246,0.1);
}

/* ── Badge émotion ───────────────────────────────────────────────────────── */
.emotion-badge {
    display: inline-block;
    font-family: 'Orbitron', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 4px 14px;
    border-radius: 20px;
    border: 2px solid var(--accent-purple);
    color: var(--accent-purple);
    background: rgba(139,92,246,0.1);
    animation: pulse-glow 2.5s ease-in-out infinite;
    margin-top: 6px;
}
.emotion-badge-happy   { border-color: var(--emotion-happy);   color: var(--emotion-happy);   background: rgba(245,158,11,0.1); }
.emotion-badge-sad     { border-color: var(--emotion-sad);     color: var(--emotion-sad);     background: rgba(59,130,246,0.1); }
.emotion-badge-angry   { border-color: var(--emotion-angry);   color: var(--emotion-angry);   background: rgba(239,68,68,0.1); }
.emotion-badge-neutral { border-color: var(--emotion-neutral); color: var(--emotion-neutral); background: rgba(16,185,129,0.1); }

/* ── Waveform BPM ────────────────────────────────────────────────────────── */
.bpm-waveform {
    display: flex;
    align-items: center;
    gap: 3px;
    height: 24px;
    margin: 4px 0;
}
.bpm-bar {
    width: 3px;
    border-radius: 2px;
    background: var(--accent-cyan);
    animation: waveform 0.8s ease-in-out infinite;
}
.bpm-bar:nth-child(1) { animation-delay: 0.0s; height: 60%; }
.bpm-bar:nth-child(2) { animation-delay: 0.1s; height: 100%; }
.bpm-bar:nth-child(3) { animation-delay: 0.2s; height: 45%; }
.bpm-bar:nth-child(4) { animation-delay: 0.3s; height: 80%; }
.bpm-bar:nth-child(5) { animation-delay: 0.4s; height: 35%; }
.bpm-bar:nth-child(6) { animation-delay: 0.15s; height: 90%; }
.bpm-bar:nth-child(7) { animation-delay: 0.25s; height: 55%; }

/* ── Carte XAI ───────────────────────────────────────────────────────────── */
.xai-card {
    background: rgba(6, 182, 212, 0.04);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-top: 0.5rem;
    font-size: 0.86rem;
    color: #cbd5e1;
    font-family: 'Inter', sans-serif;
    position: relative;
    overflow: hidden;
}
.xai-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, var(--accent-cyan), transparent);
}

/* ── Badge plateforme ────────────────────────────────────────────────────── */
.platform-badge {
    display: inline-block;
    font-family: 'Orbitron', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 3px 14px;
    border-radius: 20px;
    border: 1px solid rgba(139,92,246,0.5);
    background: rgba(139,92,246,0.12);
    color: #a78bfa;
    text-transform: uppercase;
}
.platform-badge-lastfm  { border-color: #d32f2f; background: rgba(211,47,47,0.12); color: #ef5350; }
.platform-badge-youtube { border-color: #c62828; background: rgba(198,40,40,0.10); color: #ef5350; }
.platform-badge-apple   { border-color: #ad1457; background: rgba(173,20,87,0.10); color: #f48fb1; }

/* ── Carte track Last.fm ─────────────────────────────────────────────────── */
.lastfm-track-card {
    background: rgba(139,92,246,0.05);
    border: 1px solid rgba(139,92,246,0.18);
    border-radius: 10px;
    padding: 0.65rem 0.9rem;
    margin: 0.3rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
    transition: border-color 0.2s;
}
.lastfm-track-card:hover {
    border-color: rgba(139,92,246,0.5);
}

/* ── Camera container ────────────────────────────────────────────────────── */
.camera-container {
    position: relative;
    border-radius: 10px;
    overflow: hidden;
    border: 2px solid rgba(139,92,246,0.4);
    box-shadow: 0 0 20px rgba(139,92,246,0.2);
}
.camera-scan {
    position: absolute;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    animation: scan-line 2.5s linear infinite;
    pointer-events: none;
}

/* ── Sidebar header ──────────────────────────────────────────────────────── */
.sidebar-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #a78bfa, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.06em;
    filter: drop-shadow(0 0 10px rgba(139,92,246,0.6));
}
.sidebar-subtitle {
    color: #475569;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Métriques header ────────────────────────────────────────────────────── */
.metric-cyber {
    background: rgba(139,92,246,0.06);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    text-align: center;
}
.metric-cyber .value {
    font-family: 'Orbitron', monospace;
    font-size: 1.2rem;
    color: var(--accent-cyan);
    font-weight: 700;
}
.metric-cyber .label {
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
/* ── Player card (embedded players) ─────────────────────────────────────── */
.player-card {
    background: rgba(139, 92, 246, 0.05);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 12px;
    padding: 10px;
    margin: 6px 0;
    backdrop-filter: blur(8px);
    overflow: hidden;
}
</style>
"""

# ──────────────────────────────────────────────────────────────────────────────
# Remapping 7 émotions DeepFace → 4 classes projet
# ──────────────────────────────────────────────────────────────────────────────

def _map7to4(em: dict) -> Dict[str, float]:
    a = float(em.get("angry", 0.0))
    d = float(em.get("disgust", 0.0))
    f = float(em.get("fear", 0.0))
    h = float(em.get("happy", 0.0))
    s = float(em.get("sad", 0.0))
    u = float(em.get("surprise", 0.0))
    n = float(em.get("neutral", 0.0))
    mapped = {"angry": a + d + f, "happy": h, "sad": s, "neutral": n + 0.5 * u}
    tot = sum(mapped.values()) or 0.0
    if tot > 0:
        mapped = {k: v / tot for k, v in mapped.items()}
    return mapped

# ──────────────────────────────────────────────────────────────────────────────
# Fil caméra — module-level (non bloquant pour le chat)
# ──────────────────────────────────────────────────────────────────────────────

_EMA_EMOTIONS = ("neutral", "happy", "sad", "angry")

_CAM_STATE: Dict = {
    "running":    False,
    "thread":     None,
    "frame_rgb":  None,
    "probs":      {"neutral": 1.0},
    "probs_ema":  {"neutral": 1.0, "happy": 0.0, "sad": 0.0, "angry": 0.0},
    "label":      "neutral",
    "faces_seen": False,
    "frame_count": 0,
}
_CAM_LOCK = threading.Lock()
_cam_stop_event: threading.Event = threading.Event()


def _camera_worker(stop_event: threading.Event) -> None:
    """Fil caméra : capture en continu, analyse DeepFace, stocke dans _CAM_STATE."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with _CAM_LOCK:
            _CAM_STATE["running"] = False
        return

    frame_count = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1

            faces = _FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(70, 70))
            probs: Dict[str, float] = {}
            label = "neutral"

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_bgr = frame[y:y+h, x:x+w]

                # Pré-traitement image
                try:
                    if w >= 80 and h >= 80:
                        face_bgr = cv2.resize(face_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
                    gmean = float(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).mean())
                    gamma = 1.0 if 90 <= gmean <= 160 else (1.15 if gmean < 90 else 0.9)
                    table = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255.0).astype("uint8")
                    face_bgr = cv2.LUT(face_bgr, table)
                except Exception:
                    pass

                # Analyse DeepFace (si disponible)
                if frame_count % FRAME_INTERVAL_DEEPFACE == 0 and is_ready():
                    try:
                        res = analyze_emotion(face_bgr, detector_backend="skip", enforce_detection=False)
                        if isinstance(res, list) and res:
                            res = res[0]
                        em = (res or {}).get("emotion") or {}
                        s = float(sum(em.values()) or 0.0)
                        if s > 0:
                            probs = _map7to4(em)
                    except Exception:
                        pass

                # Fallback heuristique si DeepFace n'a rien retourné
                if not probs:
                    roi_gray = gray[y:y+h, x:x+w]
                    mean = float(roi_gray.mean())
                    std  = float(roi_gray.std())
                    val  = max(0.0, min(1.0, (mean - 60.0) / 120.0))
                    eng  = max(0.0, min(1.0, (std  - 20.0) / 80.0))
                    if val > 0.65 and eng > 0.35:
                        probs = {"happy": 0.7, "neutral": 0.3}
                    elif val < 0.35 and eng < 0.40:
                        probs = {"sad": 0.6, "neutral": 0.4}
                    else:
                        probs = {"neutral": 1.0}

                label = max(probs, key=probs.get)

                # Overlay visuel
                cv2.rectangle(rgb, (x, y), (x + w, y + h), (100, 255, 150), 2)
                cv2.putText(rgb, label, (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 150), 2)
                try:
                    tops = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:2]
                    cv2.putText(rgb, " | ".join(f"{k}:{v:.2f}" for k, v in tops),
                                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                except Exception:
                    pass

            with _CAM_LOCK:
                _CAM_STATE["frame_rgb"]   = rgb
                _CAM_STATE["frame_count"] = frame_count
                ema = _CAM_STATE["probs_ema"]
                if probs:
                    # EMA — α=0.4 vers la nouvelle observation
                    for k in _EMA_EMOTIONS:
                        ema[k] = 0.4 * probs.get(k, 0.0) + 0.6 * ema[k]
                    _CAM_STATE["probs"]      = probs
                    _CAM_STATE["faces_seen"] = True
                else:
                    # Pas de visage → déclin progressif vers neutre
                    for k in _EMA_EMOTIONS:
                        target = 1.0 if k == "neutral" else 0.0
                        ema[k] = 0.15 * target + 0.85 * ema[k]
                # Normalisation (évite la dérive flottante sur sessions longues)
                ema_sum = sum(ema.values()) or 1.0
                for k in _EMA_EMOTIONS:
                    ema[k] /= ema_sum
                # Label dérivé de l'EMA (transitions douces)
                _CAM_STATE["label"] = max(ema, key=ema.get)

            time.sleep(0.04)  # ~25 fps
    finally:
        cap.release()
        with _CAM_LOCK:
            _CAM_STATE["running"] = False


def start_camera() -> None:
    """Démarre le fil caméra en arrière-plan."""
    global _cam_stop_event
    with _CAM_LOCK:
        if _CAM_STATE["running"]:
            return
    _cam_stop_event = threading.Event()
    t = threading.Thread(target=_camera_worker, args=(_cam_stop_event,), daemon=True)
    t.start()
    with _CAM_LOCK:
        _CAM_STATE["running"] = True
        _CAM_STATE["thread"]  = t


def stop_camera() -> None:
    """Arrête le fil caméra."""
    _cam_stop_event.set()
    with _CAM_LOCK:
        _CAM_STATE["running"] = False
        _CAM_STATE["thread"]  = None
        _CAM_STATE["frame_rgb"] = None


def get_face_probs() -> Dict[str, float]:
    """Retourne la dernière distribution d'émotions faciales capturée."""
    with _CAM_LOCK:
        return dict(_CAM_STATE["probs"])


def get_camera_status_v3() -> str:
    """
    Dérive le statut qualitatif de la caméra pour le poids dynamique V3.
    Utilisé par process_multimodal_emotions(camera_status=...).
    """
    with _CAM_LOCK:
        if not _CAM_STATE["running"] or not _CAM_STATE["faces_seen"]:
            return "unavailable"
        max_conf = max(_CAM_STATE["probs"].values(), default=0.0)
        if max_conf < 0.35:
            return "low_light"
        return "ok"


# ──────────────────────────────────────────────────────────────────────────────
# Gestion session state
# ──────────────────────────────────────────────────────────────────────────────

for _k, _v in {
    "features_cache": {},
    "af_fail": set(),
    "spotify_ready": False,
    "af_cache_by_id": {},
    "af_fail_ids": set(),
    "emotion_history": {},
    "disliked_uris": set(),
    "api_err": set(),
    "block_spotify_features": False,
}.items():
    st.session_state.setdefault(_k, _v)


# ──────────────────────────────────────────────────────────────────────────────
# Spotify helpers
# ──────────────────────────────────────────────────────────────────────────────

def normalize_redirect_uri():
    cur = os.getenv("SPOTIPY_REDIRECT_URI") or ""
    if ":8503/callback" in cur or "//localhost:8503/callback" in cur:
        fixed = "http://127.0.0.1:8888/callback"
        os.environ["SPOTIPY_REDIRECT_URI"] = fixed

def get_sp_client(pid: str) -> SpotifyInterface:
    return SpotifyInterface(
        scope=SCOPE,
        market=st.session_state.get("market", "FR"),
        cache_path=f".cache-{pid}",
        show_dialog=True,
    )

def _fb_reload(pid: str):
    if pid and st.session_state.get("_fb_pid") != pid:
        st.session_state["fb_stats"] = load_feedback_stats(f"logs/experiment_{pid}.jsonl")
        st.session_state["_fb_pid"] = pid
    st.session_state.setdefault(
        "fb_stats",
        {"disliked": set(), "liked": set(), "rating_avg": {}, "emo_stats": {}},
    )

def get_audio_features_for_uri(sp: Spotify, uri: str) -> dict:
    if not sp or not uri:
        return {}
    try:
        track_id = uri.split(":")[-1]
        feats_list = sp.audio_features([track_id])
        if not feats_list or feats_list[0] is None:
            return {}
        f = feats_list[0]
        return {"valence": f.get("valence"), "energy": f.get("energy"), "tempo": f.get("tempo")}
    except Exception:
        return {}

def fetch_audio_features_with_cache(sp, track_ids: List[str],
                                     sleep_between: float = 0.25,
                                     max_batch: int = 50) -> Dict[str, Dict]:
    out = {}
    if st.session_state.get("block_spotify_features"):
        return out
    max_batch = min(max_batch, 20)
    cache = st.session_state["af_cache_by_id"]
    fail  = st.session_state["af_fail_ids"]
    pending = [tid for tid in track_ids if tid and tid not in cache and tid not in fail]
    for tid in track_ids:
        if tid in cache:
            out[tid] = cache[tid]
    for i in range(0, len(pending), max_batch):
        chunk = pending[i: i + max_batch]
        try:
            af_list = sp.audio_features(chunk) or []
            for f in af_list or []:
                if f and f.get("id"):
                    cache[f["id"]] = f
                    out[f["id"]] = f
        except SpotifyException as e:
            status = getattr(e, "http_status", None)
            if status in (401, 403):
                st.session_state["block_spotify_features"] = True
                break
            for tid in chunk:
                fail.add(tid)
        except Exception:
            for tid in chunk:
                fail.add(tid)
        finally:
            if st.session_state.get("block_spotify_features"):
                break
            time.sleep(sleep_between)
    return out

def _safe_audio_features(sp, uri: str) -> Dict:
    try:
        if not st.session_state.get("spotify_ready") or not uri:
            return {}
        tid = (uri or "").split(":")[-1]
        if not tid:
            return {}
        id_cache  = st.session_state.get("af_cache_by_id", {})
        fail_ids  = st.session_state.get("af_fail_ids", set())
        uri_cache = st.session_state.get("features_cache", {})
        if tid in id_cache:
            return id_cache.get(tid) or {}
        if tid in fail_ids:
            return {}
        if uri in uri_cache:
            feat = uri_cache.get(uri) or {}
            if feat.get("id"):
                id_cache[feat["id"]] = feat
            return feat
        af_map = fetch_audio_features_with_cache(sp, [tid], sleep_between=0.0, max_batch=1) or {}
        feat = af_map.get(tid) or {}
        uri_cache[uri] = feat
        return feat
    except SpotifyException:
        st.session_state["block_spotify_features"] = True
        return {}
    except Exception:
        return {}

def clean_profile(profile: Dict, label: str) -> Dict:
    label = (label or "neutral").lower()
    prof = dict(profile or {})
    for k in list(prof.keys()):
        if prof[k] is None:
            prof.pop(k, None)
    if "mean_valence" not in prof or "mean_energy" not in prof:
        d = DEFAULT_TARGETS.get(label, DEFAULT_TARGETS["neutral"])
        prof.setdefault("mean_valence", d["val"])
        prof.setdefault("mean_energy",  d["eng"])
    return prof

def _sanitize_and_shorten_seeds(seeds: Dict, label: str) -> Dict:
    s = dict(seeds or {})
    label_l = (label or "neutral").lower()
    rot_map = st.session_state.setdefault("_seed_rot", {})
    rot = int(rot_map.get(label_l, 0))
    base_artists = list(s.get("artists") or [])
    base_tracks  = list(s.get("tracks")  or [])
    def _pick_rot(lst, k):
        if not lst: return []
        n = min(k, len(lst))
        start = rot % len(lst)
        return [lst[(start + i) % len(lst)] for i in range(n)]
    safe_genres = LABEL_SAFE_GENRES.get(label_l, LABEL_SAFE_GENRES["neutral"])
    genres_in   = list(dict.fromkeys(s.get("genres") or []))
    base        = [g for g in genres_in if g and g.lower() not in ("hip-hop", "hiphop", "rap")]
    if not base: base = safe_genres
    mixed_all   = list(dict.fromkeys(safe_genres + base + genres_in))
    mixed_all   = [g for g in mixed_all if g and g.lower() not in ("hip-hop", "hiphop", "rap")]
    if not mixed_all: mixed_all = safe_genres
    s["artists"] = _pick_rot(base_artists, 2)
    s["tracks"]  = _pick_rot(base_tracks, 2)
    s["genres"]  = _pick_rot(mixed_all, 3)
    s["profile"] = clean_profile(s.get("profile"), label_l)
    rot_map[label_l] = rot + 1
    return s

def cold_start_preferences():
    st.info("Personnalisation initiale.")
    genres = st.multiselect(
        "Genres favoris",
        ["pop","dance","rock","hip-hop","indie","electronic","chill","latin","afropop"],
        default=["pop","dance"],
    )
    energy = st.slider("Énergie cible", 0.0, 1.0, 0.6, 0.05)
    mood   = st.slider("Valence", 0.0, 1.0, 0.6, 0.05)
    if st.button("Enregistrer préférences"):
        st.session_state["seeds"] = {
            "artists": [], "tracks": [],
            "genres":  genres[:3] or ["pop","dance"],
            "profile": {"mean_valence": mood, "mean_energy": energy},
        }
        st.success("Préférences sauvegardées.")

def ensure_seeds(spif: SpotifyInterface):
    if "seeds" in st.session_state:
        st.session_state["seeds"]["profile"] = clean_profile(
            st.session_state["seeds"].get("profile"), "neutral"
        )
        safe = ["pop","indie","dance"]
        g = list(dict.fromkeys((st.session_state["seeds"].get("genres") or []) + safe))
        st.session_state["seeds"]["genres"] = g[:5]
        return
    prof = build_user_profile(spif.sp)
    if not prof["artists"] and not prof["tracks"] and not prof["genres"]:
        cold_start_preferences()
        if "seeds" not in st.session_state:
            st.session_state["seeds"] = {
                "artists": [], "tracks": [], "genres": ["pop","dance"], "profile": {}
            }
    else:
        st.session_state["seeds"] = prof
    st.session_state["seeds"]["profile"] = clean_profile(
        st.session_state["seeds"].get("profile"), "neutral"
    )
    safe = ["pop","indie","dance"]
    g = list(dict.fromkeys((st.session_state["seeds"].get("genres") or []) + safe))
    st.session_state["seeds"]["genres"] = g[:5]

def manual_auth_ui(spif: SpotifyInterface, key_prefix: str = "auth"):
    with st.expander("Autorisation manuelle Spotify"):
        if not spif:
            st.info("Client Spotify non initialisé.")
            return
        url = None
        try:
            am = getattr(spif, "auth_manager", None)
            if am and hasattr(am, "get_authorize_url"):
                url = am.get_authorize_url()
        except Exception:
            pass
        if url:
            st.link_button("Ouvrir la page d'autorisation Spotify", url, key=f"{key_prefix}_link")
        col1, _ = st.columns(2)
        with col1:
            if st.button("Vider cache OAuth", key=f"{key_prefix}_clear"):
                try: spif.clear_cache()
                except Exception: pass
                st.success("Cache vidé.")

def _collect_user_signals(sp) -> Dict[str, set]:
    st.session_state.setdefault("_user_top_artist_ids", None)
    st.session_state.setdefault("_user_liked_track_ids", None)
    if st.session_state["_user_top_artist_ids"] is None:
        top_artist_ids = set()
        try:
            ta = sp.current_user_top_artists(limit=20, time_range="medium_term") or {}
            for a in ta.get("items") or []:
                if a.get("id"): top_artist_ids.add(a["id"])
        except Exception:
            pass
        st.session_state["_user_top_artist_ids"] = top_artist_ids
    if st.session_state["_user_liked_track_ids"] is None:
        liked_ids = set()
        try:
            offs = 0
            while offs < 100:
                batch = sp.current_user_saved_tracks(limit=50, offset=offs) or {}
                items = batch.get("items") or []
                if not items: break
                for it in items:
                    t = it.get("track") or {}
                    if t.get("id"): liked_ids.add(t["id"])
                offs += 50
        except Exception:
            pass
        st.session_state["_user_liked_track_ids"] = liked_ids
    return {
        "top_artist_ids": st.session_state["_user_top_artist_ids"] or set(),
        "liked_track_ids": st.session_state["_user_liked_track_ids"] or set(),
    }

def _emotion_distance_score(track: Dict, label: str, sp) -> float:
    d = DEFAULT_TARGETS.get((label or "neutral").lower(), DEFAULT_TARGETS["neutral"])
    tgt_v, tgt_e = float(d["val"]), float(d["eng"])
    try:
        tid = track.get("id")
        if not tid: return 0.0
        f = st.session_state.get("af_cache_by_id", {}).get(tid)
        if not f:
            f = _safe_audio_features(sp, track.get("uri") or f"spotify:track:{tid}")
        v = f.get("valence"); e = f.get("energy")
        if v is None or e is None: return 0.0
        dist = (float(v) - tgt_v) ** 2 + (float(e) - tgt_e) ** 2
        return max(0.0, 1.0 - min(1.0, dist * 2.0))
    except Exception:
        return 0.0

def _rerank_tracks_personalized(tracks: List[Dict], label: str, sp) -> List[Dict]:
    sig = _collect_user_signals(sp)
    ta = sig["top_artist_ids"]; liked = sig["liked_track_ids"]
    scored = []
    for t in tracks or []:
        tid = t.get("id")
        aid = ((t.get("artists") or [{}])[0] or {}).get("id")
        aff = 0.0
        if tid in liked: aff += 1.5
        if aid in ta:    aff += 1.0
        emo   = _emotion_distance_score(t, label, sp)
        score = 0.65 * aff + 0.35 * emo
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored]

def recommend_from_library(sp, label: str, limit: int = 6,
                            exclude_uris=None, profile: Dict = None,
                            max_fetch: int = 600) -> List[Dict]:
    label = (label or "neutral").lower().strip()
    profile = profile or {}
    d = DEFAULT_TARGETS.get(label, DEFAULT_TARGETS["neutral"])
    tgt_val = float(profile.get("mean_valence", d["val"]))
    tgt_eng = float(profile.get("mean_energy",  d["eng"]))
    ex = set(exclude_uris or [])
    items = []; candidates = []
    try:
        offset = 0
        while offset < max_fetch:
            batch = sp.current_user_saved_tracks(limit=50, offset=offset) or {}
            page  = batch.get("items") or []
            if not page: break
            items.extend(page)
            for it in page:
                t = it.get("track") or {}
                uri = t.get("uri")
                if uri and uri not in ex: candidates.append(t)
            offset += 50
            if len(candidates) >= max(limit * 10, 60): break
    except Exception:
        pass
    if not candidates: return []
    uniq = []; seen = set()
    for t in candidates:
        uri = t.get("uri")
        if uri and uri not in seen:
            uniq.append(t); seen.add(uri)
    candidates = uniq
    if len(candidates) <= limit:
        return candidates[: max(1, limit)]
    pool = candidates[:]; random.shuffle(pool); pool = pool[:120]
    ids = [t.get("id") for t in pool if t.get("id")]
    feats_map = {}
    try:
        feats_map = fetch_audio_features_with_cache(sp, ids, sleep_between=0.0, max_batch=20) or {}
    except Exception:
        pass
    if feats_map:
        scored = []
        for t in pool:
            tid = t.get("id")
            if not tid: continue
            f = feats_map.get(tid)
            if not f: continue
            v = f.get("valence"); e = f.get("energy")
            if v is None or e is None: continue
            dist = (float(v) - tgt_val) ** 2 + (float(e) - tgt_eng) ** 2
            scored.append((dist, t))
        scored.sort(key=lambda x: x[0])
        ranked = [t for _, t in scored]
        if len(ranked) < limit:
            ranked_uris = {t.get("uri") for t in ranked if t.get("uri")}
            for t in candidates:
                uri = t.get("uri")
                if uri and uri not in ranked_uris:
                    ranked.append(t); ranked_uris.add(uri)
                if len(ranked) >= limit: break
        return ranked[: max(1, limit)]
    random.shuffle(candidates)
    return candidates[: max(1, limit)]

def _fallback_tracks(spif, market, base_label, seeds_mod, exclude):
    try:
        rec = (spif.sp.recommendations(
            seed_genres=",".join((seeds_mod.get("genres") or [])[:5] or ["pop","indie","dance"]),
            limit=10, market=market) or {})
        tracks = [t for t in rec.get("tracks") or [] if t.get("uri")]
        return tracks[:6]
    except Exception:
        return []

def _robust_recommend(spif: SpotifyInterface, label: str,
                       seeds_mod: Dict, limit: int, exclude_uris) -> List[Dict]:
    if not st.session_state.get("spotify_ready"):
        return []
    label = (label or "neutral").lower()
    seeds_local = dict(seeds_mod or {})
    def _call(seeds, rerank=True):
        return (recommend_for_emotion(
            spif.sp, spif.market, label, seeds,
            limit=limit, prefer_preview=True,
            rerank=rerank and (not st.session_state.get("block_spotify_features")),
            exclude_uris=exclude_uris,
        ) or [])
    tracks = []
    try:
        tracks = _call(seeds_local, rerank=True)
    except SpotifyException as e:
        http_status = getattr(e, "http_status", None)
        if http_status in (401, 403, 429):
            st.session_state["block_spotify_features"] = True
    except Exception as e:
        st.session_state["api_err"].add(str(e))
    if not tracks:
        try:
            tracks = _call(seeds_local, rerank=False)
        except Exception:
            pass
    if not tracks:
        safe_only = dict(seeds_local)
        safe_only["artists"] = []; safe_only["tracks"] = []
        safe_only["genres"]  = LABEL_SAFE_GENRES.get(label, LABEL_SAFE_GENRES["neutral"])
        safe_only["profile"] = {}
        try:
            tracks = _call(safe_only, rerank=False)
        except Exception:
            pass
    if not tracks:
        try:
            tracks = _fallback_tracks(spif, spif.market, label, seeds_local, set()) or []
        except Exception:
            pass
    if tracks:
        try:
            tracks = _rerank_tracks_personalized(tracks, label, spif.sp)[:limit]
        except Exception:
            pass
    return tracks or []

def reset_session():
    pid = st.session_state.get("pid")
    if pid:
        p = f".cache-{pid}"
        if os.path.exists(p):
            try: os.remove(p)
            except: pass
    st.session_state.clear()
    st.rerun()

def purge_all_cache():
    import glob, shutil
    removed = 0
    for p in glob.glob(".cache-*") + [".cache-spotipy"]:
        try:
            if os.path.isdir(p):   shutil.rmtree(p, ignore_errors=True); removed += 1
            elif os.path.isfile(p): os.remove(p); removed += 1
        except Exception:
            pass
    try:
        import streamlit as _st; _st.cache_data.clear()
    except Exception: pass
    try:
        import streamlit as _st; _st.cache_resource.clear()
    except Exception: pass
    return removed

def get_lastfm_client() -> "LastFMInterface | None":
    """
    Retourne un singleton LastFMInterface depuis la session_state.
    Retourne None si la clé API n'est pas configurée ou si pylast manque.
    """
    if not lastfm_available():
        return None
    api_key    = os.getenv("LASTFM_API_KEY", "").strip()
    api_secret = os.getenv("LASTFM_API_SECRET", "").strip()
    if not api_key or not api_secret:
        return None
    username = st.session_state.get("lastfm_username", "").strip() or None
    # Recréer si le username a changé
    cached = st.session_state.get("_lastfm_client")
    cached_user = st.session_state.get("_lastfm_client_user")
    if cached and cached_user == username:
        return cached
    try:
        client = LastFMInterface(api_key, api_secret, username=username)
        st.session_state["_lastfm_client"]      = client
        st.session_state["_lastfm_client_user"] = username
        return client
    except Exception:
        return None


def update_fb_stats(track_uri: str, label: str, rating: int, like: bool, dislike: bool):
    if not track_uri: return
    label = (label or "neutral").lower()
    st.session_state.setdefault("fb_stats", {})
    stats = st.session_state["fb_stats"]
    stats.setdefault(label, {})
    stats[label].setdefault(track_uri, {"likes": 0, "dislikes": 0, "ratings": []})
    if like:    stats[label][track_uri]["likes"]   += 1
    if dislike: stats[label][track_uri]["dislikes"] += 1
    if isinstance(rating, int):
        stats[label][track_uri]["ratings"].append(rating)
        stats[label][track_uri]["ratings"] = stats[label][track_uri]["ratings"][-20:]


# ──────────────────────────────────────────────────────────────────────────────
# Fragment caméra — auto-refresh toutes les 0.5s (Streamlit ≥ 1.33)
# ──────────────────────────────────────────────────────────────────────────────

_EMOTION_COLORS = {
    "happy":   "#f59e0b",
    "sad":     "#3b82f6",
    "angry":   "#ef4444",
    "neutral": "#10b981",
}


@st.fragment(run_every=0.25)
def _camera_live_fragment() -> None:
    """Affiche frame + badge EMA + barres de confiance. Rafraîchi 4×/s."""
    with _CAM_LOCK:
        frame     = _CAM_STATE.get("frame_rgb")
        label     = _CAM_STATE.get("label", "neutral")
        probs_ema = dict(_CAM_STATE.get("probs_ema", {"neutral": 1.0}))
        running   = _CAM_STATE.get("running", False)

    color = _EMOTION_COLORS.get(label, "#8b5cf6")

    if running and frame is not None:
        st.markdown(
            '<div class="camera-container">'
            '<div class="camera-scan"></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.image(frame, channels="RGB", use_container_width=True)
        st.markdown(
            f'<div class="emotion-badge emotion-badge-{label}" '
            f'style="border-color:{color}; color:{color}; '
            f'background:rgba({_hex_to_rgb(color)},0.12);">'
            f'◉ {label.upper()}</div>',
            unsafe_allow_html=True,
        )
        # Barres de confiance EMA
        for emotion, prob in sorted(probs_ema.items(), key=lambda kv: kv[1], reverse=True):
            st.progress(float(prob), text=f"{emotion} {prob:.0%}")
    elif running:
        st.info("Initialisation caméra…")
    else:
        st.caption("Caméra inactive — cliquez sur ▶ Démarrer")


def _hex_to_rgb(hex_color: str) -> str:
    """Convertit #RRGGBB en 'R,G,B' pour CSS rgba()."""
    h = hex_color.lstrip("#")
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"
    except Exception:
        return "139,92,246"


# ──────────────────────────────────────────────────────────────────────────────
# Génération URL plateforme (YouTube / Apple Music)
# ──────────────────────────────────────────────────────────────────────────────

def _build_platform_url(platform: str, artists: List[str], genres: List[str]) -> str:
    """
    Construit une URL de recherche pour YouTube ou Apple Music à partir des
    artistes et genres suggérés par l'agent V3.
    """
    parts = (artists[:1] or []) + (genres[:2] or [])
    query = " ".join(parts).strip()
    if not query:
        query = "musique"
    q_enc = urllib.parse.quote(query)
    if platform == "YouTube":
        return f"https://www.youtube.com/results?search_query={q_enc}"
    if platform == "Apple Music":
        return f"https://music.apple.com/search?term={q_enc}"
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Rendu XAI — expander analyse cognitive
# ──────────────────────────────────────────────────────────────────────────────

def _render_xai_expander(xai: Dict) -> None:
    """
    Affiche l'expander XAI sous la bulle de l'agent.
    Contient : bpm_analysis, analyse_cognitive_interne, confidence_score, métriques.
    """
    with st.expander("🔬 Voir l'analyse cognitive (XAI — Chapitre 6)"):
        # Métriques clés
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎵 Valence",   f"{xai.get('valence', 0):.2f}")
        col2.metric("⚡ Énergie",    f"{xai.get('energy',  0):.2f}")
        col3.metric("🎯 Confiance", f"{xai.get('confidence_score', 0):.0%}")
        col4.metric("😊 Émotion",   xai.get("emotion", "—").capitalize())

        st.divider()

        # Analyse BPM
        bpm_txt = xai.get("bpm_analysis", "")
        if bpm_txt:
            st.markdown("**Analyse physiologique (BPM)**")
            st.info(bpm_txt)

        # Raisonnement technique
        cog = xai.get("analyse_cognitive_interne", "")
        if cog:
            st.markdown("**Raisonnement interne (XAI)**")
            st.markdown(f"```\n{cog}\n```")

        if xai.get("from_fallback"):
            st.warning("⚠️ Mode compatibilité V1 — LLM indisponible lors de cette analyse.")


# ──────────────────────────────────────────────────────────────────────────────
# Rendu des recommandations musicales (multi-plateforme)
# ──────────────────────────────────────────────────────────────────────────────

def _render_music_block(music_data: Dict, spif, pid: str) -> None:
    """
    Affiche les recommandations musicales selon la plateforme choisie.
    - Spotify → liste de tracks avec preview / embed + feedback
    - YouTube / Apple Music → st.link_button stylé
    """
    if not music_data:
        return

    platform = music_data.get("platform", "Spotify")
    label    = music_data.get("label", "neutral")
    log_path = f"logs/experiment_{pid}.jsonl" if pid else "logs/experiment_demo.jsonl"

    st.markdown("---")

    if platform == "Last.fm":
        # ── Last.fm tracks ───────────────────────────────────────────────────
        tracks_lfm = music_data.get("tracks_lastfm", [])
        badge_cls  = "platform-badge-lastfm"
        st.markdown(
            f'<span class="platform-badge {badge_cls}">🎵 Last.fm</span>',
            unsafe_allow_html=True,
        )
        if not tracks_lfm:
            lfm_key = os.getenv("LASTFM_API_KEY", "").strip()
            if not lfm_key:
                st.warning(
                    "Clé Last.fm manquante. Ajoutez `LASTFM_API_KEY` dans `.env` "
                    "et relancez l'app."
                )
            else:
                st.info("Aucune recommandation Last.fm disponible pour cette émotion.")
            return
        st.markdown("**🎧 Recommandations Last.fm**")
        for i, track in enumerate(tracks_lfm, 1):
            title      = track.get("title", "—")
            artist     = track.get("artist", "—")
            url        = track.get("url", "")
            image_url  = track.get("image_url", "")
            col_img, col_info = st.columns([1, 5])
            with col_img:
                if image_url:
                    st.image(image_url, width=52)
                else:
                    st.markdown(
                        '<div style="width:52px;height:52px;background:rgba(139,92,246,0.15);'
                        'border-radius:6px;display:flex;align-items:center;justify-content:center;'
                        'font-size:1.4rem;">🎵</div>',
                        unsafe_allow_html=True,
                    )
            with col_info:
                st.markdown(f"**{i}. {title}**")
                st.caption(f"🎤 {artist}")
                embed_url = track.get("embed_url", "")
                if embed_url:
                    st.components.v1.html(
                        f'<div style="overflow:hidden;border-radius:8px;">'
                        f'<iframe src="{embed_url}" height="175" '
                        f'allow="autoplay *; encrypted-media *; fullscreen *;" '
                        f'sandbox="allow-forms allow-popups allow-same-origin allow-scripts '
                        f'allow-storage-access-by-user-activation allow-top-navigation-by-user-activation" '
                        f'frameborder="0" width="100%" style="border-radius:8px;"></iframe></div>',
                        height=195,
                    )
                elif url:
                    st.markdown(
                        f'<a href="{url}" target="_blank" style="color:#8b5cf6;font-size:0.8rem;">🎵 Écouter sur Last.fm</a>',
                        unsafe_allow_html=True,
                    )
        return

    if platform == "YouTube":
        url       = music_data.get("url", "")
        artists   = music_data.get("artists", [])
        genres    = music_data.get("genres",  [])
        yt_videos = music_data.get("youtube_videos", [])
        st.markdown(
            '<span class="platform-badge platform-badge-youtube">▶️ YouTube</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Artistes suggérés :** {' · '.join(artists) if artists else '—'}")
        st.markdown(f"**Genres :** {' · '.join(genres) if genres else '—'}")
        if yt_videos:
            for vid in yt_videos:
                video_id = vid.get("video_id", "")
                vtitle   = vid.get("title", "")
                if not video_id:
                    continue
                if vtitle:
                    st.caption(vtitle)
                st.components.v1.iframe(
                    f"https://www.youtube.com/embed/{video_id}",
                    height=200,
                )
        elif url:
            st.link_button("🎵 Écouter sur YouTube", url, use_container_width=True)
        else:
            st.warning("Impossible de générer le lien de recherche.")
        return

    if platform == "Apple Music":
        url     = music_data.get("url", "")
        artists = music_data.get("artists", [])
        genres  = music_data.get("genres",  [])
        st.markdown(
            '<span class="platform-badge platform-badge-apple">🍎 Apple Music</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Artistes suggérés :** {' · '.join(artists) if artists else '—'}")
        st.markdown(f"**Genres :** {' · '.join(genres) if genres else '—'}")
        if url:
            st.link_button("🍎 Écouter sur Apple Music", url, use_container_width=True)
        else:
            st.warning("Impossible de générer le lien de recherche.")
        return

    # ── Spotify tracks ─────────────────────────────────────────────────────
    tracks = music_data.get("tracks", [])
    if not tracks:
        st.info("Aucun titre trouvé. Vérifiez votre connexion Spotify.")
        return

    st.markdown("**🎧 Recommandations Spotify**")
    for i, t in enumerate(tracks, 1):
        name     = t.get("name", "?")
        artists  = ", ".join(a.get("name", "") for a in t.get("artists", []))
        uri      = t.get("uri")
        prev     = t.get("preview_url")
        external = t.get("external_urls", {}).get("spotify")

        c1, c2, c3 = st.columns([4, 4, 2])
        with c1:
            st.write(f"{i}. **{name}** — {artists}")
            if prev:
                st.audio(prev)
            elif external:
                track_id = t.get("id", "")
                st.components.v1.html(
                    f'<div style="background:rgba(139,92,246,0.05);border:1px solid '
                    f'rgba(139,92,246,0.25);border-radius:12px;padding:10px;overflow:hidden;">'
                    f'<iframe src="https://open.spotify.com/embed/track/{track_id}'
                    f'?utm_source=generator" width="100%" height="152" frameborder="0" '
                    f'allowtransparency="true" allow="encrypted-media"></iframe></div>',
                    height=172,
                )
            else:
                st.caption("Pas d'extrait disponible")

        with c2:
            rating  = st.slider(f"Compatibilité #{i}", 1, 5, 3, key=f"chat_rate_{music_data.get('turn_id',0)}_{i}")
            like    = st.toggle(f"👍 #{i}", key=f"chat_like_{music_data.get('turn_id',0)}_{i}")
            played  = st.toggle(f"Écouté #{i}", key=f"chat_play_{music_data.get('turn_id',0)}_{i}")
            dislike = st.toggle(f"👎 #{i}", key=f"chat_dis_{music_data.get('turn_id',0)}_{i}")
            if dislike and uri:
                st.session_state.setdefault("disliked_uris", set()).add(uri)

            if st.button(f"Feedback #{i}", key=f"chat_fb_{music_data.get('turn_id',0)}_{i}"):
                feats = get_audio_features_for_uri(spif.sp, uri) if uri and spif else {}
                xai   = music_data.get("xai", {})
                append_log_line(log_path, {
                    "timestamp":    int(time.time() * 1000),
                    "scenario":     "CHAT_V3",
                    "pid":          pid,
                    "track_uri":    uri,
                    "track_name":   name,
                    "track_artists": artists,
                    "fused_label":  label,
                    "rating":       int(rating),
                    "like":         bool(like),
                    "played":       bool(played),
                    "dislike":      bool(dislike),
                    "audio_features": feats,
                    "platform":     platform,
                    # Champs XAI V3
                    "agent_message_utilisateur":    xai.get("message_utilisateur", ""),
                    "agent_bpm_analysis":           xai.get("bpm_analysis", ""),
                    "agent_analyse_cognitive":      xai.get("analyse_cognitive_interne", ""),
                    "agent_valence":                xai.get("valence"),
                    "agent_energy":                 xai.get("energy"),
                    "agent_confidence":             xai.get("confidence_score"),
                    "agent_suggested_genres":       xai.get("suggested_genres", []),
                    "agent_suggested_artists":      xai.get("suggested_artists", []),
                    "agent_bpm":                    xai.get("user_bpm"),
                    "agent_from_fallback":          xai.get("from_fallback", False),
                    "fusion_version": "V1" if xai.get("from_fallback") else "V3",
                })
                update_fb_stats(uri, label, int(rating), bool(like), bool(dislike))
                st.success("✅ Feedback enregistré.")

        with c3:
            if st.button(f"Features #{i}", key=f"chat_feat_{music_data.get('turn_id',0)}_{i}"):
                if uri and spif:
                    feats = get_audio_features_for_uri(spif.sp, uri)
                    if feats:
                        st.json(feats)
                    else:
                        st.warning("Features indisponibles (API 403).")


# ──────────────────────────────────────────────────────────────────────────────
# Génération musicale trimodale → données pour le chat
# ──────────────────────────────────────────────────────────────────────────────

def _generate_music_data(agent_result: Dict, spif, pid: str,
                          platform: str, turn_id: int) -> Dict:
    """
    Prépare les données musicales à afficher dans le chat selon la plateforme.
    Pour Spotify : appelle _robust_recommend avec les genres de l'agent.
    Pour YouTube / Apple Music : construit une URL de recherche.
    """
    emotion   = agent_result.get("emotion_unifiee", "neutral")
    mp        = agent_result.get("music_params", {})
    genres    = mp.get("suggested_genres",  [])
    artists   = mp.get("suggested_artists", [])
    valence   = mp.get("target_valence",    0.55)
    energy    = mp.get("target_energy",     0.50)
    from_fallback = agent_result.get("from_fallback", False)

    base = {
        "platform":  platform,
        "label":     emotion,
        "genres":    genres,
        "artists":   artists,
        "turn_id":   turn_id,
        "xai": {
            "message_utilisateur":    agent_result.get("message_utilisateur", ""),
            "bpm_analysis":           agent_result.get("bpm_analysis", ""),
            "analyse_cognitive_interne": agent_result.get("analyse_cognitive_interne", ""),
            "valence":                valence,
            "energy":                 energy,
            "confidence_score":       agent_result.get("confidence_score", 0.0),
            "suggested_genres":       genres,
            "suggested_artists":      artists,
            "user_bpm":               st.session_state.get("user_bpm", 75),
            "from_fallback":          from_fallback,
            "emotion":                emotion,
        },
    }

    if platform == "YouTube":
        base["url"] = _build_platform_url("YouTube", artists, genres)
        yt_key = os.getenv("YOUTUBE_API_KEY", "").strip()
        if yt_key:
            try:
                from recommender.youtube_search import search_youtube
                query = " ".join((artists[:1] or []) + (genres[:2] or [])).strip() or "musique"
                base["youtube_videos"] = search_youtube(query, yt_key, max_results=3)
            except Exception:
                base["youtube_videos"] = []
        else:
            base["youtube_videos"] = []
        return base

    if platform == "Apple Music":
        base["url"] = _build_platform_url("Apple Music", artists, genres)
        return base

    # ── Last.fm ───────────────────────────────────────────────────────────────
    if platform == "Last.fm":
        lfm = get_lastfm_client()
        tracks_lfm: list = []
        if lfm:
            try:
                tracks_lfm = lfm.get_recommendations(emotion, limit=5)
            except Exception:
                tracks_lfm = []
        if tracks_lfm:
            try:
                from recommender.itunes_search import get_apple_embed_url_batch
                get_apple_embed_url_batch(tracks_lfm)
            except Exception:
                for t in tracks_lfm:
                    t.setdefault("embed_url", "")
        base["tracks_lastfm"] = tracks_lfm
        return base

    # ── Spotify ──────────────────────────────────────────────────────────────
    if not st.session_state.get("spotify_ready") or not spif:
        base["tracks"] = []
        return base

    try:
        ensure_seeds(spif)
    except Exception:
        pass

    seeds = dict(st.session_state.get("seeds", {}))

    # Injection des genres suggérés par l'agent (priorité haute)
    if genres and not from_fallback:
        existing = list(seeds.get("genres") or [])
        for g in genres:
            if g not in existing:
                existing.insert(0, g)
        seeds["genres"] = existing[:5]

    seeds["profile"] = clean_profile(
        {"mean_valence": valence, "mean_energy": energy}, emotion
    )
    seeds = _sanitize_and_shorten_seeds(seeds, emotion)
    prof  = dict(seeds.get("profile") or {})
    prof.pop("mean_tempo", None)
    seeds["profile"] = {
        "mean_valence": prof.get("mean_valence"),
        "mean_energy":  prof.get("mean_energy"),
    }

    exclude = (
        st.session_state.get("disliked_uris", set())
        | set(st.session_state.get("chat_track_uris", []))
    )
    limit  = 6
    tracks = _robust_recommend(spif, emotion, seeds, limit, exclude)

    # Apprentissage feedback
    stats  = st.session_state.get("fb_stats") or {}
    tracks = rerank_tracks_with_feedback(tracks, emotion, stats, top_k=limit)

    # Mémorisation des URIs pour éviter les répétitions
    new_uris = [t.get("uri") for t in tracks if t.get("uri")]
    st.session_state.setdefault("chat_track_uris", [])
    st.session_state["chat_track_uris"].extend(new_uris)

    base["tracks"] = tracks
    return base


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar V3 — caméra + signaux + connexion
# ──────────────────────────────────────────────────────────────────────────────

def _sidebar_ui():
    with st.sidebar:
        # ── En-tête Cyber Violet ──────────────────────────────────────────────
        st.markdown(
            "<div style='text-align:center; padding:10px 0;'>"
            "<span class='sidebar-logo'>FACE²MELODY</span><br>"
            "<span class='sidebar-subtitle'>V3 · Compagnon Émotionnel IA · UQAM</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Vue ───────────────────────────────────────────────────────────────
        view = st.radio("Vue", ["Expérience", "Analytics"], index=0, key="view_mode",
                        horizontal=True)

        st.divider()

        if st.session_state.get("view_mode") != "Expérience":
            return

        # ── Plateforme musicale ───────────────────────────────────────────────
        st.markdown("### 🎵 Plateforme")
        st.selectbox(
            "Cible de recommandation",
            ["Last.fm", "Spotify", "YouTube", "Apple Music"],
            key="platform",
            help=(
                "Last.fm : recommandations gratuites sans Premium. "
                "Spotify : API complète (Premium requis). "
                "YouTube / Apple Music : liens de recherche directs."
            ),
        )
        # Connexion Last.fm (si plateforme sélectionnée)
        if st.session_state.get("platform") == "Last.fm":
            lfm_key = os.getenv("LASTFM_API_KEY", "").strip()
            if not lfm_key:
                st.warning(
                    "Clé Last.fm manquante dans `.env`. "
                    "Créez une app sur last.fm/api/account/create"
                )
            else:
                st.text_input(
                    "Nom d'utilisateur Last.fm (optionnel)",
                    key="lastfm_username",
                    placeholder="ex: votre_pseudo",
                    help="Laissez vide pour des recommandations anonymes par tags émotionnels.",
                )
                lfm = get_lastfm_client()
                if lfm:
                    if st.session_state.get("lastfm_username"):
                        st.success(f"✓ Last.fm : {st.session_state['lastfm_username']}")
                    else:
                        st.caption("🎵 Mode anonyme — recommandations par émotion")

        st.divider()

        # ── BPM simulé ────────────────────────────────────────────────────────
        st.markdown("### 💓 Physiologie")
        bpm = st.slider(
            "Rythme cardiaque simulé (BPM)",
            min_value=40, max_value=160, value=75, step=1,
            key="user_bpm",
            help="Signal physiologique pour la fusion trimodale V3. BPM > 100 + visage neutre = anxiété détectée.",
        )
        # Waveform animée + feedback coloré
        if bpm > 120:
            bpm_color = "#ef4444"; bpm_label = "stress aigu possible"
        elif bpm > 100:
            bpm_color = "#f97316"; bpm_label = "anxiété latente suspectée"
        elif bpm >= 80:
            bpm_color = "#10b981"; bpm_label = "état équilibré"
        elif bpm >= 60:
            bpm_color = "#3b82f6"; bpm_label = "calme physiologique"
        else:
            bpm_color = "#8b5cf6"; bpm_label = "fatigue ou mélancolie"
        bars = "".join(
            f'<div class="bpm-bar" style="background:{bpm_color};'
            f'animation-duration:{max(0.3, 0.8 - (bpm-40)/200):.2f}s;"></div>'
            for _ in range(7)
        )
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<div class="bpm-waveform">{bars}</div>'
            f'<span style="color:{bpm_color};font-size:0.78rem;font-weight:600;">'
            f'{bpm_label}</span></div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Caméra temps réel ─────────────────────────────────────────────────
        st.markdown("### 📷 Flux Vidéo (Vision)")

        cam_running = _CAM_STATE["running"]
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if not cam_running:
                if st.button("▶ Démarrer", key="start_cam", use_container_width=True):
                    start_camera()
                    st.rerun()
            else:
                if st.button("⏹ Arrêter", key="stop_cam", use_container_width=True):
                    stop_camera()
                    st.rerun()
        with c_btn2:
            if cam_running and st.button("🔄 Rafraîchir", key="refresh_cam", use_container_width=True):
                st.rerun()

        # Affichage live (fragment auto-refresh 0.5s)
        _camera_live_fragment()

        if cam_running:
            cam_status = get_camera_status_v3()
            status_map = {
                "ok":          "🟢 Signal facial fiable",
                "low_light":   "🟡 Faible luminosité — poids texte augmenté",
                "unavailable": "🔴 Visage non détecté",
            }
            st.caption(status_map.get(cam_status, cam_status))

        st.divider()

        # ── Auth section — conditionnel selon la plateforme ──────────────────
        _platform = st.session_state.get("platform", "Spotify")

        if _platform == "Spotify":
            st.markdown("### 🎧 Connexion Spotify")

            pid = st.text_input(
                "Participant ID",
                value=st.session_state.get("pid", ""),
                key="pid_input_sidebar",
                placeholder="ex: P001",
            )
            if pid:
                st.session_state["pid"] = pid

            st.selectbox("Market", ["FR","US","GB","DE","ES","IT","CA"],
                         index=0, key="market_select_sidebar")

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                if st.button("Connexion", key="connect_btn_sidebar", use_container_width=True):
                    if st.session_state.get("pid"):
                        spif = get_sp_client(st.session_state["pid"])
                        st.session_state["spif"] = spif
                        set_sp_global(spif.sp)
                        set_cache_salt(st.session_state["pid"])
                        try:
                            me   = spif.sp.current_user()
                            miss = spif.missing_scopes(REQUIRED_SCOPES) or []
                            st.session_state["missing_scopes"]  = miss
                            st.session_state["spotify_ready"]   = True
                            st.success(f"✓ {me.get('display_name', 'Connecté')}")
                            if not miss:
                                ensure_seeds(spif)
                        except Exception as e:
                            st.session_state["spotify_ready"]  = False
                            st.error(f"OAuth : {e}")
                            manual_auth_ui(spif, key_prefix="sidebar_auth")
                    else:
                        st.warning("Renseignez un Participant ID.")
            with col_s2:
                if st.button("Reset", key="reset_sidebar", use_container_width=True):
                    reset_session()

            if st.session_state.get("spotify_ready"):
                st.success("Spotify connecté")
            else:
                st.caption("Non connecté à Spotify.")

        elif _platform == "YouTube":
            st.markdown("### ▶️ YouTube")
            yt_key = os.getenv("YOUTUBE_API_KEY", "").strip()
            if yt_key:
                st.caption("✓ YOUTUBE_API_KEY détectée — embed vidéo activé")
            else:
                st.caption("Sans clé API : liens de recherche uniquement.  \n"
                           "Ajoutez `YOUTUBE_API_KEY` dans `.env` pour les embeds.")

        elif _platform == "Apple Music":
            st.markdown("### 🍎 Apple Music")
            st.caption("Lecture via iTunes embed — aucune clé API requise.")

        # Last.fm : UI déjà gérée dans la section plateforme ci-dessus

        st.divider()

        # ── Diagnostics compact ───────────────────────────────────────────────
        with st.expander("Diagnostics"):
            st.caption(f"Détection: {get_status()} | Backend: {get_backend()}")
            st.caption(f"Cache feats: {len(st.session_state.get('af_cache_by_id', {}))}")
            if st.session_state.get("api_err"):
                for e in sorted(st.session_state["api_err"])[:3]:
                    st.caption(f"• {e}")
            if st.button("Purger caches", key="purge_sidebar"):
                n = purge_all_cache()
                st.info(f"{n} caches supprimés.")


# ──────────────────────────────────────────────────────────────────────────────
# Interface chat principale — V3 Affective Companion
# ──────────────────────────────────────────────────────────────────────────────

def _chat_ui():
    """Interface conversationnelle principale — Fusion Trimode V3."""
    from agent_logic import process_multimodal_emotions

    spif = st.session_state.get("spif")
    pid  = st.session_state.get("pid", "")
    _fb_reload(pid)

    # ── Injection CSS + Fonts (Cyber Violet) ────────────────────────────────
    st.markdown(_FONTS, unsafe_allow_html=True)
    st.markdown(_CSS,   unsafe_allow_html=True)

    # ── En-tête principal ────────────────────────────────────────────────────
    st.markdown(
        '<div class="f2m-title">FACE²MELODY</div>'
        '<div class="f2m-subtitle">V3 · Compagnon Émotionnel IA · M.Sc. UQAM · Chapitre 6</div>',
        unsafe_allow_html=True,
    )

    # ── Métriques live ───────────────────────────────────────────────────────
    platform = st.session_state.get("platform", "Last.fm")
    bpm_val  = st.session_state.get("user_bpm", 75)
    cam_ok   = _CAM_STATE["running"] and _CAM_STATE["faces_seen"]

    with _CAM_LOCK:
        detected_emotion = _CAM_STATE.get("label", "neutral")
    emo_color = _EMOTION_COLORS.get(detected_emotion, "#8b5cf6")

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown(
            f'<div class="metric-cyber">'
            f'<div class="value">{platform}</div>'
            f'<div class="label">Plateforme</div></div>',
            unsafe_allow_html=True,
        )
    with h2:
        bpm_icon = "🔴" if bpm_val > 100 else ("🟢" if bpm_val >= 60 else "🔵")
        st.markdown(
            f'<div class="metric-cyber">'
            f'<div class="value">{bpm_icon} {bpm_val}</div>'
            f'<div class="label">BPM</div></div>',
            unsafe_allow_html=True,
        )
    with h3:
        st.markdown(
            f'<div class="metric-cyber">'
            f'<div class="value">{"🟢 ON" if cam_ok else "⚫ OFF"}</div>'
            f'<div class="label">Caméra</div></div>',
            unsafe_allow_html=True,
        )
    with h4:
        st.markdown(
            f'<div class="metric-cyber">'
            f'<div class="value" style="color:{emo_color};">'
            f'{detected_emotion.upper()}</div>'
            f'<div class="label">Émotion</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Initialisation de l'historique chat ──────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "role":    "assistant",
                "content": "Bonjour Mohamed, comment vous sentez-vous aujourd'hui ?",
                "type":    "greeting",
            }
        ]
    if "chat_turn_id" not in st.session_state:
        st.session_state["chat_turn_id"] = 0

    # ── Affichage de l'historique ─────────────────────────────────────────────
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
            # Composant XAI (si disponible)
            if msg.get("xai_data"):
                _render_xai_expander(msg["xai_data"])
            # Recommandations musicales (si disponibles)
            if msg.get("music_data"):
                _render_music_block(msg["music_data"], spif, pid)

    # ── Entrée utilisateur ────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Comment vous sentez-vous ? Décrivez votre état en quelques mots…"
    )

    if not user_input:
        if not st.session_state.get("spotify_ready") and platform == "Spotify":
            st.info(
                "Connectez-vous à Spotify dans la barre latérale pour recevoir "
                "des recommandations personnalisées, ou choisissez YouTube / Apple Music."
            )
        return

    # ── Traitement du message ─────────────────────────────────────────────────
    # 1. Ajout du message utilisateur
    st.session_state["chat_history"].append({
        "role":    "user",
        "content": user_input,
    })

    # 2. Capture des signaux trimodaux
    face_probs     = get_face_probs()
    user_bpm       = st.session_state.get("user_bpm", 75)
    camera_status  = get_camera_status_v3()

    # 3. Appel agent V3 — Fusion Trimode (Vision + Texte + BPM)
    with st.spinner("🧠 Analyse trimodale en cours (Vision · Texte · BPM)…"):
        try:
            agent_result = process_multimodal_emotions(
                face_probs    = face_probs,
                user_text     = user_input,
                user_bpm      = user_bpm,
                camera_status = camera_status,
            )
        except Exception as exc:
            logger.error("process_multimodal_emotions a échoué : %s", exc)
            agent_result = {
                "emotion_unifiee":           "neutral",
                "bpm_analysis":              "Analyse indisponible.",
                "analyse_cognitive_interne": f"Erreur : {exc}",
                "message_utilisateur":       "Je rencontre une difficulté technique. Réessayez dans un instant.",
                "confidence_score":          0.0,
                "music_params": {
                    "target_valence":    0.55,
                    "target_energy":     0.50,
                    "suggested_artists": [],
                    "suggested_genres":  ["pop","indie","lo-fi"],
                },
                "from_fallback": True,
            }

    # 4. Génération musicale selon la plateforme
    turn_id = st.session_state["chat_turn_id"]
    st.session_state["chat_turn_id"] += 1

    with st.spinner(f"🎵 Génération des recommandations {platform}…"):
        music_data = _generate_music_data(agent_result, spif, pid, platform, turn_id)

    # 5. Construction du message assistant
    assistant_content = agent_result.get("message_utilisateur") or (
        f"J'ai détecté une émotion **{agent_result.get('emotion_unifiee','—')}**. "
        "Voici ce que je vous propose."
    )

    # Ajout de l'émotion en gras si pas déjà mentionnée
    emotion_cap = agent_result.get("emotion_unifiee", "neutral").capitalize()
    if emotion_cap.lower() not in assistant_content.lower():
        assistant_content = f"**{emotion_cap} détecté** · {assistant_content}"

    xai_data = {
        "bpm_analysis":            agent_result.get("bpm_analysis", ""),
        "analyse_cognitive_interne": agent_result.get("analyse_cognitive_interne", ""),
        "confidence_score":        agent_result.get("confidence_score", 0.0),
        "valence":                 agent_result.get("music_params", {}).get("target_valence", 0.55),
        "energy":                  agent_result.get("music_params", {}).get("target_energy",  0.50),
        "suggested_genres":        agent_result.get("music_params", {}).get("suggested_genres",  []),
        "suggested_artists":       agent_result.get("music_params", {}).get("suggested_artists", []),
        "emotion":                 agent_result.get("emotion_unifiee", "neutral"),
        "user_bpm":                user_bpm,
        "from_fallback":           agent_result.get("from_fallback", False),
    }

    st.session_state["chat_history"].append({
        "role":       "assistant",
        "content":    assistant_content,
        "type":       "response",
        "xai_data":   xai_data,
        "music_data": music_data,
    })

    # Mise à jour de l'historique émotionnel (pour déduplication Spotify)
    label_emo = agent_result.get("emotion_unifiee", "neutral")
    new_uris  = [t.get("uri") for t in music_data.get("tracks", []) if t.get("uri")]
    st.session_state.setdefault("emotion_history", {})
    st.session_state["emotion_history"].setdefault(label_emo, set()).update(new_uris)

    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ──────────────────────────────────────────────────────────────────────────────

def run_app():
    try:
        normalize_redirect_uri()
    except Exception:
        pass

    _sidebar_ui()

    # Vue Analytics
    if st.session_state.get("view_mode") == "Analytics":
        if analytics_render:
            try:
                analytics_render()
            except Exception as e:
                st.error("Erreur dans la page Analytics.")
                st.exception(e)
        else:
            st.error(
                "Module analytics_gen introuvable. "
                "Lancez `streamlit run analytics_gen.py` pour l'utiliser en autonome."
            )
        return

    # Vue Expérience — interface chat V3
    _chat_ui()


run_app()
