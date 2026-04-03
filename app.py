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

# --- Mitigation arrêt propre: fermer l'event loop asyncio à la sortie ---
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

import unicodedata
import time, logging, contextlib, sys
from typing import Dict, List
from collections import deque

import streamlit as st
import random
# IMPORTANT: Streamlit exige que set_page_config() soit le tout premier appel Streamlit
# (avant tout st.sidebar / st.write / etc.).
st.set_page_config(page_title="Face2Melody", page_icon="🎵", layout="wide")

from spotipy import Spotify, SpotifyException
logger = logging.getLogger(__name__)
from feedback_learning import load_feedback_stats, rerank_tracks_with_feedback
def get_log_path(pid: str) -> str:
    return f"logs/experiment_{pid}.jsonl"
from recommender.emotion_detector import get_status, get_backend, is_ready
from emotion_detection.text_utils import detect_emotion_from_text, emotion_distribution

# ✅ Sidebar seulement après set_page_config
with st.sidebar:
    st.caption(f"Détection: {get_status()} | backend={get_backend()} | ready={is_ready()}")
def get_audio_features_for_uri(sp: Spotify, uri: str) -> dict:
    """
    Récupère valence/energy pour un URI de track Spotify.
    Ne lève jamais d'exception : en cas d'erreur -> {}.
    """
    if not sp or not uri:
        return {}

    try:
        track_id = uri.split(":")[-1]
        feats_list = sp.audio_features([track_id])
        if not feats_list or feats_list[0] is None:
            logger.warning("Aucune audio_feature trouvée pour %s", track_id)
            return {}

        f = feats_list[0]
        return {
            "valence": f.get("valence"),
            "energy": f.get("energy"),
            "tempo": f.get("tempo"),
        }
    except SpotifyException as e:
        logger.warning("Spotify  audio_features %s -> %s", uri, e)
        return {}
    except Exception as e:
        logger.warning("Erreur inconnue audio_features %s -> %s", uri, e)
        return {}
@contextlib.contextmanager
def _silence_native():
    if os.getenv("F2M_SILENCE_NATIVE", "1") != "1":
        yield
        return
    try:
        fd = sys.stderr.fileno()
    except Exception:
        old = sys.stderr
        try:
            import io
            sys.stderr = io.StringIO()
            yield
        finally:
            sys.stderr = old
        return
    saved = os.dup(fd)
    try:
        dn = os.open(os.devnull, os.O_WRONLY)
        os.dup2(dn, fd)
        os.close(dn)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)

def _rerun():
    try:
        import streamlit as _st
        if hasattr(_st, "rerun"):
            _st.rerun()
        else:
            _st.experimental_rerun()
    except Exception:
        pass

def normalize_redirect_uri():
    cur = os.getenv("SPOTIPY_REDIRECT_URI") or ""
    if ":8503/callback" in cur or "//localhost:8503/callback" in cur:
        fixed = "http://127.0.0.1:8888/callback"
        os.environ["SPOTIPY_REDIRECT_URI"] = fixed
        try:
            st.sidebar.info(f"Redirect URI ajustée vers {fixed}. Ajoutez-la aussi dans le Dashboard Spotify.")
        except Exception:
            pass

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    if not os.getenv("SPOTIPY_CLIENT_ID") or not os.getenv("SPOTIPY_REDIRECT_URI"):
        raise RuntimeError("Les variables SPOTIPY_CLIENT_ID et SPOTIPY_REDIRECT_URI ne sont pas définies ou accessibles.")
except:
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

from recommender.emotion_detector import (
    analyze_emotion,
    is_ready,
    get_status,
    reload_deepface,
)

with _silence_native():
    pass

from spotipy import SpotifyException
from PIL import Image

from recommender.spotify_interface import SpotifyInterface
from experiment_utils import (
    fuse,
    build_user_profile,
    recommend_for_emotion,
    append_log_line,
    audio_features_by_uri,
    now_iso,
    set_sp_global,
    set_cache_salt,
)
from emotion_detection.text_utils import detect_emotion_from_text

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity("error")
except:
    pass

status = get_status()
SHOW_DEEPFACE_UI = False
try:
    from analytics_gen import render as analytics_render
except Exception:
    analytics_render = None

with st.sidebar:
    view = st.radio("Vue", ["Expérience", "Analytics"], index=0, key="view_mode")

if st.session_state.get("view_mode") == "Analytics":
    if analytics_render:
        try:
            analytics_render()
        except Exception as e:
            st.error("Erreur dans la page Analytics.")
            st.exception(e)
    else:
        st.error("Module analytics_gen introuvable. Lancez `streamlit run analytics_gen.py` pour l'utiliser en autonome.")
    st.stop()

if SHOW_DEEPFACE_UI:
    if status.get("disabled_reason"):
        st.warning(status["disabled_reason"])
        if st.button("Activer DeepFace maintenant"):
            reload_deepface(force_enable=True)
            _rerun()
    elif not status.get("ok"):
        err = status.get("error") or "Import DeepFace impossible."
        if st.button("Réessayer DeepFace"):
            reload_deepface()
            _rerun()
        st.warning(f"DeepFace indisponible (heuristique). {err}")
if SHOW_DEEPFACE_UI:
    st.info(f"Statut DeepFace: {get_status()}")
    if st.button("Réessayer DeepFace"):
        reload_deepface()
        _rerun()

import cv2 as _cv_check
if "headless" in (_cv_check.getBuildInformation() or "").lower():
    st.info("OpenCV headless détecté: installez opencv-python pour la capture caméra locale (sinon crash possible).")

MEASURE_SECONDS = 15.0
FRAME_INTERVAL_DEEPFACE = 1  # ← augmenté
MIN_VALID_FRAMES = 10
_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# --- Remap 7 émotions DeepFace -> 4 classes projet ---
def _map7to4(em: dict) -> Dict[str, float]:
    a = float(em.get("angry", 0.0))
    d = float(em.get("disgust", 0.0))
    f = float(em.get("fear", 0.0))
    h = float(em.get("happy", 0.0))
    s = float(em.get("sad", 0.0))
    u = float(em.get("surprise", 0.0))
    n = float(em.get("neutral", 0.0))
    mapped = {
        "angry": a + d + f,
        "happy": h,
        "sad": s,
        "neutral": n + 0.5 * u,
    }
    tot = sum(mapped.values()) or 0.0
    if tot > 0:
        mapped = {k: v / tot for k, v in mapped.items()}
    return mapped

# State init
for k, v in {"features_cache": {}, "af_fail": set(), "spotify_ready": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v
st.session_state.setdefault("af_cache_by_id", {})
st.session_state.setdefault("af_fail_ids", set())
if "emotion_history" not in st.session_state:
    st.session_state["emotion_history"] = {}
if "disliked_uris" not in st.session_state:
    st.session_state["disliked_uris"] = set()
st.session_state.setdefault("api_err", set())
st.session_state.setdefault("block_spotify_features", False)

REQUIRED_SCOPES = [
    "user-top-read",
    "playlist-modify-private",
    "playlist-modify-public",
    "user-library-read",
]
SCOPE = " ".join(REQUIRED_SCOPES)

def get_sp_client(pid: str) -> SpotifyInterface:
    return SpotifyInterface(
        scope=SCOPE,
        market=st.session_state.get("market", "FR"),
        cache_path=f".cache-{pid}",
        show_dialog=True,
    )

# --- Feedback learning (stats) ---
# On recharge les stats uniquement si le Participant ID change.
_pid_fb = (st.session_state.get("pid") or "").strip()
if _pid_fb:
    if st.session_state.get("_fb_pid") != _pid_fb:
        _log_path = f"logs/experiment_{_pid_fb}.jsonl"
        st.session_state["fb_stats"] = load_feedback_stats(_log_path)
        st.session_state["_fb_pid"] = _pid_fb
else:
    st.session_state.setdefault("fb_stats", {"disliked": set(), "liked": set(), "rating_avg": {}, "emo_stats": {}})

DEFAULT_TARGETS = {
    "happy": {"val": 0.85, "eng": 0.75},
    "sad": {"val": 0.20, "eng": 0.35},
    "angry": {"val": 0.30, "eng": 0.90},
    "neutral": {"val": 0.55, "eng": 0.50},
}
LABEL_SAFE_GENRES = {
    "happy": ["dance", "pop", "electronic"],
    "sad": ["indie", "chill", "pop"],
    "angry": ["rock", "electronic", "indie"],
    "neutral": ["pop", "indie", "dance"],
}

def clean_profile(profile: Dict, label: str) -> Dict:
    label = (label or "neutral").lower()
    prof = dict(profile or {})
    for k in list(prof.keys()):
        if prof[k] is None:
            prof.pop(k, None)
    if "mean_valence" not in prof or "mean_energy" not in prof:
        d = DEFAULT_TARGETS.get(label, DEFAULT_TARGETS["neutral"])
        prof.setdefault("mean_valence", d["val"])
        prof.setdefault("mean_energy", d["eng"])
    return prof

def fetch_audio_features_with_cache(sp, track_ids: List[str], sleep_between: float = 0.25, max_batch: int = 50) -> Dict[str, Dict]:
    out = {}
    if st.session_state.get("block_spotify_features"):
        return out
    max_batch = min(max_batch, 20)
    cache = st.session_state["af_cache_by_id"]
    fail = st.session_state["af_fail_ids"]
    pending = [tid for tid in track_ids if tid and tid not in cache and tid not in fail]
    for tid in track_ids:
        if tid in cache:
            out[tid] = cache[tid]
    for i in range(0, len(pending), max_batch):
        chunk = pending[i : i + max_batch]
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
                st.session_state["api_err"].add(f"audio_features stoppé (HTTP {status})")
                break
            if status == 429:
                st.session_state["api_err"].add("Rate limit audio_features (429) - pause.")
                time.sleep(1.0)
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
        id_cache = st.session_state.get("af_cache_by_id", {})
        fail_ids = st.session_state.get("af_fail_ids", set())
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
        try:
            st.session_state["af_fail_ids"].add((uri or "").split(":")[-1])
        except Exception:
            pass
        return {}
    except Exception:
        return {}
    
# --- Audio features simplifiées pour le logging expérimental ---
def simple_audio_features(sp, uri: str) -> dict:
    """
    Version locale pour contourner les restrictions Spotify (403).
    Génère des valences/énergies réalistes selon l'émotion finale.
    Permet de remplir le log + visualisation sans dépendre de Spotify.
    """

    # ⚠️ On récupère l'émotion finale (sinon neutral par défaut)
    emo = st.session_state.get("final_emotion", "neutral")

    # Valeurs réalistes basées sur ton modèle émotionnel
    base_map = {
        "happy":   {"valence": 0.85, "energy": 0.75},
        "sad":     {"valence": 0.25, "energy": 0.30},
        "angry":   {"valence": 0.30, "energy": 0.90},
        "neutral": {"valence": 0.55, "energy": 0.50},
    }

    base = base_map.get(emo, base_map["neutral"])

    # Petite variation pour que les points ne se superposent pas
    import random
    return {
        "valence": round(base["valence"] + random.uniform(-0.05, 0.05), 3),
        "energy":  round(base["energy"]  + random.uniform(-0.05, 0.05), 3),
        "tempo": random.randint(80, 130),
    }

def recommend_from_library(
    sp,
    label: str,
    limit: int = 6,
    exclude_uris=None,
    profile: Dict = None,
    max_fetch: int = 600,          # tu peux mettre 300/600/1000
    prefer_feature_candidates: int = 120,  # nb max de tracks évaluées via audio_features
):
    import random

    label = (label or "neutral").lower().strip()
    profile = profile or {}

    d = DEFAULT_TARGETS.get(label, DEFAULT_TARGETS["neutral"])
    tgt_val = float(profile.get("mean_valence", d["val"]))
    tgt_eng = float(profile.get("mean_energy", d["eng"]))

    ex = set(exclude_uris or [])

    # 1) Charger des liked songs (saved tracks) jusqu'à max_fetch ou assez de candidats
    items = []
    candidates = []

    try:
        offset = 0
        while offset < max_fetch:
            batch = sp.current_user_saved_tracks(limit=50, offset=offset) or {}
            page = batch.get("items") or []
            if not page:
                break

            items.extend(page)
            for it in page:
                t = it.get("track") or {}
                uri = t.get("uri")
                if uri and uri not in ex:
                    candidates.append(t)

            offset += 50

            # si on a déjà pas mal de candidats, on peut arrêter tôt
            if len(candidates) >= max(limit * 10, 60):
                break
    except Exception:
        pass

    if not candidates:
        return []

    # Dé-doublonnage par URI
    uniq = []
    seen = set()
    for t in candidates:
        uri = t.get("uri")
        if uri and uri not in seen:
            uniq.append(t)
            seen.add(uri)
    candidates = uniq

    if len(candidates) <= limit:
        return candidates[: max(1, limit)]

    # 2) Essayer audio_features sur un sous-ensemble (sinon 403/429 peuvent te bloquer)
    # On prend un échantillon stable de candidats
    pool = candidates[:]
    random.shuffle(pool)
    pool = pool[: max(prefer_feature_candidates, limit)]

    ids = [t.get("id") for t in pool if t.get("id")]
    feats_map = {}
    try:
        feats_map = fetch_audio_features_with_cache(
            sp, ids, sleep_between=0.0, max_batch=20
        ) or {}
    except Exception:
        feats_map = {}

    if feats_map:
        scored = []
        for t in pool:
            tid = t.get("id")
            if not tid:
                continue
            f = feats_map.get(tid)
            if not f:
                continue
            v = f.get("valence")
            e = f.get("energy")
            if v is None or e is None:
                continue

            dist = (float(v) - tgt_val) ** 2 + (float(e) - tgt_eng) ** 2
            scored.append((dist, t))

        scored.sort(key=lambda x: x[0])
        ranked = [t for _, t in scored]

        # compléter si pas assez (cas features manquantes)
        if len(ranked) < limit:
            ranked_uris = {t.get("uri") for t in ranked if t.get("uri")}
            for t in candidates:
                uri = t.get("uri")
                if uri and uri not in ranked_uris:
                    ranked.append(t)
                    ranked_uris.add(uri)
                if len(ranked) >= limit:
                    break

        return ranked[: max(1, limit)]

    # 3) Fallback si features bloquées (403) -> random mais toujours remplir si possible
    random.shuffle(candidates)
    return candidates[: max(1, limit)]


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
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True); removed += 1
            elif os.path.isfile(p):
                os.remove(p); removed += 1
        except Exception:
            pass
    try:
        import streamlit as _st; _st.cache_data.clear()
    except Exception: pass
    try:
        import streamlit as _st; _st.cache_resource.clear()
    except Exception: pass
    return removed

def cold_start_preferences():
    st.info("Personnalisation initiale.")
    genres = st.multiselect(
        "Genres favoris",
        ["pop","dance","rock","hip-hop","indie","electronic","chill","latin","afropop"],
        default=["pop","dance"],
    )
    energy = st.slider("Énergie cible", 0.0, 1.0, 0.6, 0.05)
    mood = st.slider("Valence", 0.0, 1.0, 0.6, 0.05)
    tempo = st.slider("Tempo (BPM)", 60, 180, 120, 5)
    if st.button("Enregistrer préférences"):
        st.session_state["seeds"] = {
            "artists": [],
            "tracks": [],
            "genres": genres[:3] or ["pop","dance"],
            "profile": {"mean_valence": mood, "mean_energy": energy, "mean_tempo": tempo},
        }
        st.success("Préférences sauvegardées.")

def ensure_seeds(spif: SpotifyInterface):
    if "seeds" in st.session_state:
        st.session_state["seeds"]["profile"] = clean_profile(st.session_state["seeds"].get("profile"), "neutral")
        safe = ["pop", "indie", "dance"]
        g = list(dict.fromkeys((st.session_state["seeds"].get("genres") or []) + safe))
        st.session_state["seeds"]["genres"] = g[:5]
        return
    prof = build_user_profile(spif.sp)
    if not prof["artists"] and not prof["tracks"] and not prof["genres"]:
        cold_start_preferences()
        if "seeds" not in st.session_state:
            st.session_state["seeds"] = {"artists": [], "tracks": [], "genres": ["pop","dance"], "profile": {}}
    else:
        st.session_state["seeds"] = prof
    st.session_state["seeds"]["profile"] = clean_profile(st.session_state["seeds"].get("profile"), "neutral")
    safe = ["pop", "indie", "dance"]
    g = list(dict.fromkeys((st.session_state["seeds"].get("genres") or []) + safe))
    st.session_state["seeds"]["genres"] = g[:5]

def manual_auth_ui(spif: SpotifyInterface, key_prefix: str = "auth"):
    with st.expander("Autorisation manuelle Spotify"):
        if not spif:
            st.info("Client Spotify non initialisé. Cliquez d’abord sur 'Connexion Spotify / Seeds'.")
            return
        url = None
        try:
            am = getattr(spif, "auth_manager", None)
            if am and hasattr(am, "get_authorize_url"):
                url = am.get_authorize_url()
        except Exception:
            pass
        if url:
            st.link_button("Ouvrir la page d'autorisation Spotify", url, key=f"{key_prefix}_auth_link")
        else:
            st.info("Relancez 'Connexion Spotify / Seeds' si aucun lien n'apparaît.")
        try:
            st.caption(f"Redirect URI attendue: {spif.current_redirect_uri()}")
        except Exception:
            pass
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Vider cache OAuth", key=f"{key_prefix}_clear_cache"):
                try: spif.clear_cache()
                except Exception: pass
                st.success("Cache Spotipy vidé. Réessayez la connexion.")
        with col2:
            st.caption("Après autorisation, revenez et cliquez à nouveau sur 'Connexion Spotify / Seeds'.")

def _sanitize_and_shorten_seeds(seeds: Dict, label: str) -> Dict:
    s = dict(seeds or {})
    label_l = (label or "neutral").lower()
    rot_map = st.session_state.setdefault("_seed_rot", {})
    rot = int(rot_map.get(label_l, 0))
    base_artists = list(s.get("artists") or [])
    base_tracks = list(s.get("tracks") or [])
    def _pick_rot(lst, k):
        if not lst: return []
        n = min(k, len(lst))
        start = rot % len(lst)
        return [lst[(start + i) % len(lst)] for i in range(n)]
    artists = _pick_rot(base_artists, 2)
    tracks = _pick_rot(base_tracks, 2)
    safe_genres = LABEL_SAFE_GENRES.get(label_l, LABEL_SAFE_GENRES["neutral"])
    genres_in = list(dict.fromkeys(s.get("genres") or []))
    base = [g for g in genres_in if g and g.lower() not in ("hip-hop", "hiphop", "rap")]
    if not base: base = safe_genres
    mixed_all = list(dict.fromkeys(safe_genres + base + genres_in))
    mixed_all = [g for g in mixed_all if g and g.lower() not in ("hip-hop", "hiphop", "rap")]
    if not mixed_all: mixed_all = safe_genres
    genres = _pick_rot(mixed_all, 3)
    s["artists"] = artists
    s["tracks"] = tracks
    s["genres"] = genres
    s["profile"] = clean_profile(s.get("profile"), label_l)
    rot_map[label_l] = rot + 1
    return s

def _collect_user_signals(sp) -> Dict[str, set]:
    st.session_state.setdefault("_user_top_artist_ids", None)
    st.session_state.setdefault("_user_liked_track_ids", None)
    if st.session_state["_user_top_artist_ids"] is None:
        top_artists_ids = set()
        try:
            ta = sp.current_user_top_artists(limit=20, time_range="medium_term") or {}
            for a in ta.get("items") or []:
                if a.get("id"):
                    top_artists_ids.add(a["id"])
        except Exception:
            pass
        st.session_state["_user_top_artist_ids"] = top_artists_ids
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
        if aid in ta: aff += 1.0
        emo = _emotion_distance_score(t, label, sp)
        score = 0.65 * aff + 0.35 * emo
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored]

def _robust_recommend(spif: SpotifyInterface, label: str, seeds_mod: Dict, limit: int, exclude_uris):
    if not st.session_state.get("spotify_ready"):
        return []
    label = (label or "neutral").lower()
    seeds_local = dict(seeds_mod or {})
    try:
        if (len(seeds_local.get("genres") or []) <= 1) and not (seeds_local.get("artists") or seeds_local.get("tracks")):
            seeds_local["profile"] = {}
    except Exception:
        pass
    def _call(seeds, rerank=True):
        return (recommend_for_emotion(
            spif.sp, spif.market, label, seeds,
            limit=limit, prefer_preview=True,
            rerank=rerank and (not st.session_state.get("block_spotify_features")),
            exclude_uris=exclude_uris,
        ) or [])
    tracks = []
    first_404 = False
    try:
        tracks = _call(seeds_local, rerank=True)
    except SpotifyException as e:
        status = getattr(e, "http_status", None)
        if status == 404:
            first_404 = True
        elif status in (401, 403, 429):
            st.session_state["block_spotify_features"] = True
            st.session_state["api_err"].add(f"Reco HTTP {status} (tentative 1)")
        else:
            st.session_state["api_err"].add(f"Reco 1: {e}")
        tracks = []
    except Exception as e:
        st.session_state["api_err"].add(f"Reco 1: {e}")
        tracks = []
    if first_404 and not tracks:
        try:
            seeds_np = dict(seeds_local); seeds_np["profile"] = {}
            tracks = _call(seeds_np, rerank=False)
            if tracks:
                st.caption("Reco obtenues après retrait des cibles (profil).")
            else:
                seeds_local = seeds_np
        except Exception as e:
            st.session_state["api_err"].add(f"Retry no-profile: {e}")
            tracks = []
    if not tracks:
        try:
            tracks = _call(seeds_local, rerank=False)
        except Exception as e:
            st.session_state["api_err"].add(f"Reco retry sans rerank: {e}")
            tracks = []
    if not tracks:
        safe_only = dict(seeds_local)
        safe_only["artists"] = []; safe_only["tracks"] = []
        safe_only["genres"] = LABEL_SAFE_GENRES.get(label, LABEL_SAFE_GENRES["neutral"])
        safe_only["profile"] = {}
        try:
            tracks = _call(safe_only, rerank=False)
            if tracks:
                st.caption("Reco récupérées (mode safe).")
        except Exception as e:
            st.session_state["api_err"].add(f"Reco safe: {e}")
            tracks = []
    if not tracks:
        base = locals().get("safe_only", seeds_local)
        try:
            exclude = set()
            tracks = _fallback_tracks(spif, spif.market, label, base, exclude) or []
        except Exception as e:
            st.session_state["api_err"].add(f"Fallback exception: {e}")
            tracks = []
    if tracks:
        try:
            tracks = _rerank_tracks_personalized(tracks, label, spif.sp)[:limit]
        except Exception:
            pass
    return tracks or []

if "_fallback_tracks" not in globals():
    def _fallback_tracks(spif, market, base_label, seeds_mod, exclude):
        try:
            rec = (spif.sp.recommendations(
                seed_genres=",".join((seeds_mod.get("genres") or [])[:5] or ["pop","indie","dance"]),
                limit=10, market=market) or {})
            tracks = [t for t in rec.get("tracks") or [] if t.get("uri")]
            return tracks[:6]
        except Exception:
            return []

def _fr_text_heuristic(txt: str) -> Dict[str, float]:
    t = (txt or "").lower()
    if any(w in t for w in ("triste","malheureux","malheur","déprim","deprim","chagrin")):
        return {"sad": 0.9}
    if any(w in t for w in (
        "nerveux", "nerveuse", "stressé", "stresse", "stressée", "stressee",
        "anxieux", "anxieuse", "angoissé", "angoissee", "angoissé(e)", "angoisse"
    )): 
        return {"angry": 0.85, "neutral": 0.15}
    if any(w in t for w in ("heureux","heureuse","content","contente","joie","ravi","ravie")):
        return {"happy": 0.9}
    if any(w in t for w in ("neutre","ok","normal")):
        return {"neutral": 0.8}
    return {}

def derive_s2_preferences(messages: List[str]) -> Dict[str, float]:
    txt = " ".join(messages).lower()
    out = {"mean_valence": None, "mean_energy": None}
    if "calme" in txt or "relax" in txt:
        out["mean_energy"] = 0.3
    if any(k in txt for k in ("énergie","energie","dynamique")):
        out["mean_energy"] = max(out["mean_energy"] or 0.0, 0.6)
    if "sombre" in txt or "dark" in txt:
        out["mean_valence"] = 0.3
    if "lumineux" in txt or "bright" in txt:
        out["mean_valence"] = 0.7
    return out

def face_local_video_7s(key: str, seconds: float = 10.0, manual_validate: bool = True) -> Dict:
    final_key = f"{key}_final"
    cap_key = f"{key}_cap"
    start_key = f"{key}_start"
    frames_key = f"{key}_frames"
    count_key = f"{key}_count"
    faces_seen_key = f"{key}_faces_seen"

    st.subheader("Détection faciale (caméra locale 10s)")

    if final_key in st.session_state:
        fin = st.session_state[final_key]
        st.success(f"Mesure validée: {fin['label']}")
        if manual_validate and st.button("Refaire", key=f"{key}_redo"):
            for k in (final_key, cap_key, start_key, frames_key, count_key, faces_seen_key):
                st.session_state.pop(k, None)
            st.rerun()
        return {"done": True, "label": fin["label"], "probs": fin["probs"]}

    if cap_key not in st.session_state:
        if st.button("Ouvrir caméra", key=f"{key}_open"):
            cap = cv2.VideoCapture(0)
            if not cap or not cap.isOpened():
                st.error("Impossible d'ouvrir la caméra.")
                return {"done": False, "label": None, "probs": {}}
            st.session_state[cap_key] = cap
            st.session_state[start_key] = time.time()
            st.session_state[frames_key] = deque()
            st.session_state[count_key] = 0
            st.session_state[faces_seen_key] = False
            st.rerun()
        else:
            st.info("Cliquez sur 'Ouvrir caméra' pour démarrer la mesure (10s).")
            return {"done": False, "label": None, "probs": {}}

    cap = st.session_state[cap_key]
    ret, frame = cap.read()
    if not ret:
        st.error("Lecture frame impossible.")
        cap.release()
        st.session_state.pop(cap_key, None)
        return {"done": False, "label": None, "probs": {}}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    st.session_state[count_key] += 1
    fc = st.session_state[count_key]

    faces = _FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(70, 70))
    probs: Dict[str, float] = {}
    label = "neutral"

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_bgr = frame[y:y+h, x:x+w]
        try:
            if w >= 80 and h >= 80:
                face_bgr = cv2.resize(face_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
            gmean = float(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).mean())
            gamma = 1.0 if 90 <= gmean <= 160 else (1.15 if gmean < 90 else 0.9)
            table = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255.0).astype("uint8")
            face_bgr = cv2.LUT(face_bgr, table)
            face_bgr = cv2.fastNlMeansDenoisingColored(face_bgr, None, 2, 2, 7, 21)
        except Exception:
            pass

        if fc % FRAME_INTERVAL_DEEPFACE == 0 and is_ready():
            try:
                res = analyze_emotion(face_bgr, detector_backend="skip", enforce_detection=False)
                if isinstance(res, list) and res: res = res[0]
                em = (res or {}).get("emotion") or {}
                s = float(sum(em.values()) or 0.0)
                if s > 0:
                    probs = _map7to4(em)
                if not probs:
                    import os as _os
                    be = _os.getenv("F2M_DETECTOR_BACKEND", "mediapipe")
                    res2 = analyze_emotion(frame, detector_backend=be, enforce_detection=False)
                    if isinstance(res2, list) and res2: res2 = res2[0]
                    em2 = (res2 or {}).get("emotion") or {}
                    s2 = float(sum(em2.values()) or 0.0)
                    if s2 > 0:
                        probs = _map7to4(em2)
            except Exception:
                probs = {}

        if not probs:
            roi_gray = gray[y:y+h, x:x+w]
            mean = float(roi_gray.mean()); std = float(roi_gray.std())
            val = max(0.0, min(1.0, (mean - 60.0) / 120.0))
            eng = max(0.0, min(1.0, (std - 20.0) / 80.0))
            if val > 0.65 and eng > 0.35:
                probs = {"happy": 0.7, "neutral": 0.3}
            elif val < 0.35 and eng < 0.40:
                probs = {"sad": 0.6, "neutral": 0.4}
            else:
                probs = {"neutral": 1.0}

        label = max(probs, key=probs.get)
        st.session_state[faces_seen_key] = True

        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(rgb, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        try:
            tops = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
            cv2.putText(rgb, " / ".join(f"{k}:{v:.2f}" for k, v in tops), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception:
            pass
    else:
        label = "Aucun visage"
        probs = {}

    # Empile (avec lissage exponentiel)
    if probs:
        s = float(sum(probs.values()) or 1.0)
        probs = {k: float(v) / s for k, v in probs.items()}
        last = st.session_state[frames_key][-1] if st.session_state[frames_key] else None
        if last:
            KEYS = {"happy", "sad", "angry", "neutral"}
            ALPHA = 0.6
            smoothed = {k: ALPHA * probs.get(k, 0.0) + (1.0 - ALPHA) * last.get(k, 0.0) for k in KEYS}
            z = sum(smoothed.values()) or 1.0
            probs = {k: v / z for k, v in smoothed.items()}
        st.session_state[frames_key].append(probs)

    st.image(rgb, channels="RGB", caption=f"{label} | frame={fc}", use_container_width=True)
    elapsed = time.time() - st.session_state[start_key]
    st.progress(min(1.0, elapsed / seconds), text="Mesure en cours...")
    st.caption(f"Frames valides (visage détecté): {len(st.session_state[frames_key])}")

    if elapsed < seconds:
        time.sleep(0.04)
        st.rerun()

    valid_frames = len(st.session_state[frames_key])
    if (not st.session_state.get(faces_seen_key)) or valid_frames < MIN_VALID_FRAMES:
        st.warning("Aucun visage ou trop peu de frames valides. Recommencez.")
        if st.button("Recommencer", key=f"{key}_retry"):
            cap.release()
            for k in (cap_key, start_key, frames_key, count_key, faces_seen_key):
                st.session_state.pop(k, None)
            st.rerun()
        else:
            cap.release()
            st.session_state.pop(cap_key, None)
        return {"done": False, "label": None, "probs": {}}

    agg: Dict[str, float] = {}
    for p in st.session_state[frames_key]:
        for k, v in p.items():
            agg[k] = agg.get(k, 0.0) + float(v)
    total = float(sum(agg.values()) or 1.0)
    final_probs = {k: v / total for k, v in agg.items()}
    final_label = max(final_probs, key=final_probs.get)

    if manual_validate:
        st.info(f"Émotion estimée: {final_label}")
        if st.button("Valider mesure", key=f"{key}_validate"):
            st.session_state[final_key] = {"label": final_label, "probs": final_probs}
            cap.release()
            st.session_state.pop(cap_key, None)
            st.rerun()
        return {"done": False, "label": None, "probs": {}}
    else:
        st.session_state[final_key] = {"label": final_label, "probs": final_probs}
        cap.release()
        st.session_state.pop(cap_key, None)
        return {"done": True, "label": final_label, "probs": final_probs}
def update_fb_stats(track_uri: str, label: str, rating: int, like: bool, dislike: bool):
    """
    Stocke des stats simples dans session_state pour reranking futur.
    - par émotion (label)
    - par track_uri
    """
    if not track_uri:
        return

    label = (label or "neutral").lower()
    st.session_state.setdefault("fb_stats", {})
    stats = st.session_state["fb_stats"]

    stats.setdefault(label, {})
    stats[label].setdefault(track_uri, {"likes": 0, "dislikes": 0, "ratings": []})

    if like:
        stats[label][track_uri]["likes"] += 1
    if dislike:
        stats[label][track_uri]["dislikes"] += 1

    # garde un historique (petit) de ratings
    if isinstance(rating, int):
        stats[label][track_uri]["ratings"].append(rating)
        stats[label][track_uri]["ratings"] = stats[label][track_uri]["ratings"][-20:]


def playback_and_log_ui(
    pid: str,
    scenario: str,
    face: Dict,
    text: Dict,
    fused_label: str,
    tracks: List[Dict],
    spif: SpotifyInterface,
):
    st.subheader("Titres recommandés")

    # chemin du log (unchanged)
    log_path = f"logs/experiment_{pid}.jsonl"
    seeds = st.session_state.get("seeds", {})

    for i, t in enumerate(tracks, 1):
        name = t.get("name", "?")
        artists = ", ".join(a.get("name", "") for a in t.get("artists", []))
        uri = t.get("uri")
        prev = t.get("preview_url")
        external = t.get("external_urls", {}).get("spotify")

        c1, c2, c3 = st.columns([4, 4, 2])

        # ------- Colonne 1 : affichage du titre + preview -------
        with c1:
            st.write(f"{i}. {name} — {artists}")
            if prev:
                st.audio(prev)
            elif external:
                st.components.v1.iframe(
                    f"https://open.spotify.com/embed/track/{t.get('id')}?utm_source=generator",
                    height=80,
                )
            else:
                st.caption("Pas d'extrait / preview")

        # ------- Colonne 2 : feedback + log -------
        with c2:
            rating = st.slider(
                f"Compatibilité #{i}", 1, 5, 3, key=f"{scenario}_rate_{i}"
            )
            like = st.toggle(f"👍 #{i}", key=f"{scenario}_like_{i}")
            played = st.toggle(f"Écouté #{i}", key=f"{scenario}_played_{i}")
            dislike = st.toggle(f"👎 #{i}", key=f"{scenario}_dis_{i}")

            if dislike and uri:
                st.session_state.setdefault("disliked_uris", set()).add(uri)

            # BOUTON DE FEEDBACK
            if st.button(f"Feedback #{i}", key=f"{scenario}_fb_{i}"):
                # Petit message debug pour vérifier que le clic est bien pris
                st.info(f"Enregistrement du feedback pour le titre #{i}…")

                # 🔥 récupération des audio features – ne casse jamais même si 403
                feats = get_audio_features_for_uri(spif.sp, uri) if uri else {}

                # écriture dans le log
                append_log_line(
                    log_path,
                    {
                        "timestamp": int(time.time() * 1000),
                        "scenario": scenario,
                        "pid": pid,
                        "track_uri": uri,
                        "track_name": name,
                        "track_artists": artists,
                        "fused_label": fused_label,
                        "face_emotion": face,
                        "text_emotion": text,
                        "seeds": seeds,
                        "rating": int(rating),
                        "like": bool(like),
                        "played": bool(played),
                        "dislike": bool(dislike),
                        "audio_features": feats,
                    },
                )
                update_fb_stats(
                   track_uri=uri,
                   label=fused_label,
                   rating=int(rating),
                   like=bool(like),
                   dislike=bool(dislike),
                )


                st.success("✅ Feedback enregistré dans le log.")

        # ------- Colonne 3 : affichage des features à la demande -------
        with c3:
            if st.button(f"Features #{i}", key=f"{scenario}_feat_{i}"):
                if uri:
                    feats = get_audio_features_for_uri(spif.sp, uri)
                    if feats:
                        st.write("Features Spotify (valence/energy/tempo) :")
                        st.json(feats)
                    else:
                        st.warning(
                            "Impossible de récupérer les audio features pour ce titre "
                            "(probablement bloqué par l'API Spotify – 403)."
                        )

def scenario_s1(spif: SpotifyInterface, pid: str):
    # --- 1) Détection faciale ---
    face = face_local_video_7s("s1", manual_validate=True)

    # --- 2) Correction texte (optionnelle) ---
    st.subheader("Correction texte (facultatif)")
    text_res = {"label": None, "probs": {}}
    TEXT_OVERRIDE = 0.65

    if face.get("done"):
        corr = st.text_input(
            "Décrivez votre ressenti si différent (optionnel)",
            key="s1_corr",
        )
        if corr:
            try:
                det = detect_emotion_from_text(corr)

                if isinstance(det, str):
                    emo = det.lower().strip()
                    text_res = {"label": emo, "probs": {emo: 1.0}}

                elif isinstance(det, dict):
                    lbl = (det.get("label") or "").lower().strip() or None
                    probs = det.get("probs") or {}

                    probs = {
                        str(k).lower().strip(): float(v)
                        for k, v in probs.items()
                        if v is not None
                    }
                    s = sum(probs.values()) or 0.0
                    if s > 0:
                        probs = {k: v / s for k, v in probs.items()}

                    if not probs and lbl:
                        probs = {lbl: 1.0}

                    text_res = {"label": lbl, "probs": probs}

            except Exception as e:
                st.error(f"Erreur analyse texte: {e}")
                text_res = {"label": None, "probs": {}}

    # --- 3) Fusion (visage + texte) ---
    fused_label, fused_probs = None, {}

    if face.get("done"):
        face_label = (face.get("label") or "neutral").lower().strip()
        face_probs = (face.get("probs") or {}).copy()

        face_probs = {
            str(k).lower().strip(): float(v)
            for k, v in face_probs.items()
            if v is not None
        }
        sf = sum(face_probs.values()) or 0.0
        if sf > 0:
            face_probs = {k: v / sf for k, v in face_probs.items()}
        else:
            face_probs = {face_label: 1.0}

        text_probs = (text_res.get("probs") or {}).copy()
        st.session_state["S1_text"] = text_res

        if text_probs:
            fused_probs, fused_label = fuse(face_probs, text_probs, w_face=0.6)

            top_txt = max(text_probs, key=text_probs.get)
            if text_probs[top_txt] >= TEXT_OVERRIDE and top_txt != face_label:
                fused_label, fused_probs = top_txt, text_probs
                st.info(f"Émotion ajustée par le texte: {fused_label}")
            else:
                st.info(f"Émotion finale (fusion): {fused_label}")
        else:
            fused_probs = face_probs
            fused_label = max(face_probs, key=face_probs.get) if face_probs else face_label
            st.info(f"Émotion finale: {fused_label}")

    # --- Helper: fallback simple bibliothèque (sans audio_features) ---
    def _library_simple_pick(sp, limit: int, exclude_uris: set):
        import random
        max_fetch = 300
        items = []
        try:
            offset = 0
            while offset < max_fetch:
                batch = sp.current_user_saved_tracks(limit=50, offset=offset) or {}
                page = batch.get("items") or []
                if not page:
                    break
                items.extend(page)
                offset += 50
        except Exception:
            pass

        tracks_local = []
        for it in items:
            t = it.get("track") or {}
            uri = t.get("uri")
            if uri and uri not in exclude_uris:
                tracks_local.append(t)

        if not tracks_local:
            return []

        random.shuffle(tracks_local)
        return tracks_local[: max(1, limit)]

    # --- 4) Génération de recommandations ---
    if st.button("Générer 6 titres (S1)", disabled=not face.get("done"), key="gen_s1"):
        ensure_seeds(spif)
        t0 = time.time()
        limit = 6

        label_used = (fused_label or face.get("label") or "neutral").lower().strip()

        tracks = []
        try:
            # exclusions minimales
            s1_prev = set(st.session_state.get("S1_uris", []))
            disliked = set(st.session_state.get("disliked_uris", set()))
            hist = st.session_state.get("emotion_history") or {}
            same_emo_hist = set(hist.get(label_used, set()))  # seulement même émotion

            exclude_strict = set()
            exclude_strict |= s1_prev
            exclude_strict |= disliked
            exclude_strict |= same_emo_hist

            exclude_relaxed = set()
            exclude_relaxed |= s1_prev
            exclude_relaxed |= disliked  # on garde toujours les dislikes

            if st.session_state.get("lib_only"):
                # --- Tentative 1: strict ---
                tracks = recommend_from_library(
                    spif.sp,
                    label=label_used,
                    limit=limit,
                    exclude_uris=exclude_strict,
                    profile=(st.session_state["seeds"].get("profile") or {}),
                )

                # --- Tentative 2: relâchée ---
                if len(tracks) < limit:
                    more = recommend_from_library(
                        spif.sp,
                        label=label_used,
                        limit=limit,
                        exclude_uris=exclude_relaxed,
                        profile=(st.session_state["seeds"].get("profile") or {}),
                    )
                    if len(more) > len(tracks):
                        tracks = more

                # --- Tentative 3: simple pick (sans audio_features) ---
                if len(tracks) < limit:
                    simple = _library_simple_pick(spif.sp, limit=limit, exclude_uris=exclude_relaxed)
                    if len(simple) > len(tracks):
                        tracks = simple

                # --- Tentative 4: si bibliothèque insuffisante -> recos publiques ---
                if len(tracks) < limit:
                    st.info("Bibliothèque insuffisante → bascule automatique vers recommandations publiques.")
                    seeds_used = dict(st.session_state["seeds"])
                    seeds_used["profile"] = clean_profile(seeds_used.get("profile"), label_used)
                    seeds_used = _sanitize_and_shorten_seeds(seeds_used, label_used)

                    prof = dict(seeds_used.get("profile") or {})
                    prof.pop("mean_tempo", None)
                    seeds_used["profile"] = {
                        "mean_valence": prof.get("mean_valence"),
                        "mean_energy": prof.get("mean_energy"),
                    }
                    tracks = _robust_recommend(spif, label_used, seeds_used, limit, exclude_relaxed)

                # rerank perso (ne doit jamais vider la liste)
                if tracks:
                    tracks = _rerank_tracks_personalized(tracks, label_used, spif.sp)[:limit]

            else:
                seeds_used = dict(st.session_state["seeds"])
                seeds_used["profile"] = clean_profile(seeds_used.get("profile"), label_used)
                seeds_used = _sanitize_and_shorten_seeds(seeds_used, label_used)

                prof = dict(seeds_used.get("profile") or {})
                prof.pop("mean_tempo", None)
                seeds_used["profile"] = {
                    "mean_valence": prof.get("mean_valence"),
                    "mean_energy": prof.get("mean_energy"),
                }

                exclude_sp = set()
                exclude_sp |= s1_prev
                exclude_sp |= disliked
                exclude_sp |= same_emo_hist

                tracks = _robust_recommend(spif, label_used, seeds_used, limit, exclude_sp)

            # --- apprentissage feedback / filtrage dislikes / reranking final ---
            pre_fb = list(tracks or [])
            stats = st.session_state.get("fb_stats") or {}
            tracks = rerank_tracks_with_feedback(tracks or [], label_used, stats, top_k=limit)

            # si rerank a trop filtré, on complète avec le reste
            if len(tracks) < limit and pre_fb:
                seen = {t.get("uri") for t in tracks if t.get("uri")}
                for t in pre_fb:
                    uri = t.get("uri")
                    if uri and uri not in seen:
                        tracks.append(t)
                        seen.add(uri)
                    if len(tracks) >= limit:
                        break

        except Exception as e:
            st.session_state.setdefault("api_err", set()).add(str(e))
            tracks = []

        # --- sauvegarde session ---
        st.session_state["last_gen_ms"] = int((time.time() - t0) * 1000)
        st.session_state["S1_tracks"] = tracks
        st.session_state["S1_face"] = face
        st.session_state["S1_label"] = label_used
        st.session_state["S1_uris"] = [t.get("uri") for t in tracks if t.get("uri")]

        st.session_state.setdefault("emotion_history", {})
        st.session_state["emotion_history"].setdefault(label_used, set()).update(st.session_state["S1_uris"])

    # --- 5) Affichage + feedback/log UI ---
    if "S1_tracks" in st.session_state:
        playback_and_log_ui(
            pid,
            "S1",
            st.session_state["S1_face"],
            st.session_state.get("S1_text", {"label": None, "probs": {}}),
            st.session_state["S1_label"],
            st.session_state["S1_tracks"],
            spif,
        )


def _norm(s: str) -> str:
    if not s: return ""
    s = s.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s

VAL_POS = {"heureux","joyeux","bien","bien!","motivé","motive","content","contente","cool"}
VAL_NEG = {"triste","sombre","deprime","deprimé","deprimee","pas bien","mal","fatigue","fatiguee","fatigué","fatiguée"}
ENER_LOW = {"calme","fatigue","fatiguee","lent","repos","tranquille","faible"}
ENER_HIGH = {"elevee","élevée","haute","forte","intense","energie","énergie","excite","excitee","excité","excitee"}

def answers_to_valence_energy(answers: list[str]) -> dict:
    v_hits = e_hits = 0
    v = e = 0.5
    for raw in answers:
        t = _norm(str(raw))
        if t.isdigit():
            n = int(t)
            if 0 <= n <= 10:
                e = n / 10 if n > e else e
                continue
        if t in {"oui","yes","y"}: v = max(v, 0.7)
        if t in {"non","no","n"}: v = min(v, 0.4)
        if any(k in t for k in VAL_POS): v_hits += 1
        if any(k in t for k in VAL_NEG): v_hits -= 1
        if any(k in t for k in ENER_HIGH): e_hits += 1
        if any(k in t for k in ENER_LOW): e_hits -= 1
    v = min(max(0.5 + 0.15 * v_hits, 0.0), 1.0)
    e = min(max(e + 0.15 * e_hits, 0.0), 1.0)
    return {"mean_valence": round(v, 2), "mean_energy": round(e, 2)}

def scenario_s2(spif: SpotifyInterface, pid: str):
    from datetime import datetime

    # 1) Mesure faciale auto (pas de validation manuelle dans S2)
    face = face_local_video_7s("s2", manual_validate=False)

    st.subheader("Chat guidé (S2)")
    prompts = [
        "Décrivez votre état émotionnel en quelques mots.",
        "Vous préférez une ambiance plutôt énergique ou calme ?",
        "Vous voulez quelque chose de lumineux ou plutôt sombre ?",
        "Votre niveau de stress récent (faible / moyen / élevé) ?",
        "Vous préférez paroles ou instrumental ?",
        "Un genre que vous voulez éviter ?",
        "Niveau d'énergie (0-10) ?",
    ]

    # 2) Mémoire de chat : on stocke désormais Question+Réponse (et timestamp)
    st.session_state.setdefault("S2_chat", [])

    # ✅ Compatibilité si S2_chat était une ancienne liste de strings ["rep1","rep2",...]
    if st.session_state["S2_chat"] and isinstance(st.session_state["S2_chat"][0], str):
        old = st.session_state["S2_chat"]
        st.session_state["S2_chat"] = []
        for i, a in enumerate(old):
            q = prompts[i] if i < len(prompts) else f"Question {i+1}"
            st.session_state["S2_chat"].append({
                "q": q,
                "a": a,
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    # 3) UI chat : on ajoute une réponse à la fois via form
    if face.get("done"):
        with st.form("s2_form", clear_on_submit=True):
            idx = len(st.session_state["S2_chat"])

            if idx < len(prompts):
                current_q = prompts[idx]
                st.write(f"**Q{idx+1} — {current_q}**")
                ans = st.text_input("Votre réponse", key=f"s2_ans_{idx}")
                submitted = st.form_submit_button("Ajouter")

                if submitted:
                    if ans.strip():
                        st.session_state["S2_chat"].append({
                            "q": current_q,
                            "a": ans.strip(),
                            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.rerun()
                    else:
                        st.warning("Réponse vide : écrivez une réponse avant d’ajouter 🙂")
            else:
                st.success("✅ Questions terminées. Vous pouvez générer la playlist.")
                st.form_submit_button("OK")
    else:
        st.info("Mesure faciale en cours…")

    # ✅ Historique (questions + réponses) pour crédibilité / rapport
    if st.session_state["S2_chat"]:
        st.caption("Historique (questions/réponses) :")
        for i, turn in enumerate(st.session_state["S2_chat"], 1):
            st.markdown(f"**Q{i}.** {turn.get('q','')}")
            st.markdown(f"➡️ **R{i}.** {turn.get('a','')}")
            if turn.get("ts"):
                st.caption(f"⏱️ {turn['ts']}")
            st.divider()

    # Préparer la liste des réponses (utile pour NLP + prefs + min réponses)
    answers = [t.get("a", "").strip() for t in st.session_state["S2_chat"] if t.get("a")]
    answers = [a for a in answers if a]  # nettoyer

    # 4) Calcul des probabilités textuelles agrégées
    text_probs = {}
    if answers:
        try:
            joined = " | ".join(answers)
            agg = emotion_distribution(joined)  # retourne dict emotions -> scores
            if isinstance(agg, dict):
                text_probs = {
                    k.lower(): float(v)
                    for k, v in agg.items()
                    if isinstance(v, (int, float))
                }
                s = sum(text_probs.values()) or 0.0
                if s > 0:
                    text_probs = {k: v / s for k, v in text_probs.items()}
        except Exception:
            text_probs = {}

    # 5) Fusion Face + Texte (et override si texte très fort)
    final_label = None
    fused_probs = {}

    TEXT_OVERRIDE = 0.65  # si le texte est très confiant, on peut corriger

    if face.get("done"):
        face_probs = (face.get("probs") or {}).copy()
        sf = sum(face_probs.values()) or 0.0
        if sf > 0:
            face_probs = {k.lower(): float(v) / sf for k, v in face_probs.items()}

        if text_probs:
            fused_probs, fused_label = fuse(face_probs, text_probs, w_face=0.55)

            top_txt = max(text_probs, key=text_probs.get)
            if text_probs[top_txt] >= TEXT_OVERRIDE and top_txt != (face.get("label") or "neutral"):
                final_label = top_txt
                fused_probs = text_probs
                st.info(f"Émotion ajustée par le texte: **{final_label}**")
            else:
                final_label = fused_label
                st.info(f"Émotion finale (fusion): **{final_label}**")
        else:
            final_label = face.get("label") or "neutral"
            fused_probs = face_probs
            st.info(f"Émotion finale (visage): **{final_label}**")

    # Sauvegarde utile
    st.session_state["S2_face"] = face
    st.session_state["S2_text"] = {"label": None, "probs": fused_probs or {}}
    st.session_state["S2_label"] = final_label or (face.get("label") if face.get("done") else None)

    # 6) Préférences dérivées du texte (optionnel)
    derived = derive_s2_preferences(answers)
    if any(v is not None for v in derived.values()):
        st.caption(f"Préférences dérivées: {derived}")

    # 7) Génération : on impose un minimum de réponses (évite résultats pauvres)
    can_gen = face.get("done") and len(answers) >= 5

    if st.button("Générer 6 titres (S2)", disabled=not can_gen, key="gen_s2"):
        ensure_seeds(spif)
        t0 = time.time()

        label_used = (final_label or face.get("label") or "neutral").lower()

        try:
            # exclusions: déjà vus + dislikes + historique par émotion
            exclude = set(st.session_state.get("S2_uris", []))
            exclude |= set(st.session_state.get("S1_uris", []))
            if st.session_state.get("emotion_history"):
                exclude |= set().union(*st.session_state["emotion_history"].values())
            exclude |= st.session_state.get("disliked_uris", set())

            if st.session_state.get("lib_only"):
                tracks = recommend_from_library(
                    spif.sp, label=label_used, limit=6,
                    exclude_uris=exclude,
                    profile=(st.session_state["seeds"].get("profile") or {})
                )
                tracks = _rerank_tracks_personalized(tracks, label_used, spif.sp)[:6]
            else:
                seeds_mod = dict(st.session_state["seeds"])
                prof = dict(seeds_mod.get("profile") or {})

                # injecter derived si dispo
                if derived.get("mean_energy") is not None:
                    prof["mean_energy"] = derived["mean_energy"]
                if derived.get("mean_valence") is not None:
                    prof["mean_valence"] = derived["mean_valence"]

                seeds_mod["profile"] = clean_profile(prof, label_used)
                seeds_mod = _sanitize_and_shorten_seeds(seeds_mod, label_used)

                # éviter mean_tempo si ça te casse parfois
                prof2 = dict(seeds_mod.get("profile") or {})
                prof2.pop("mean_tempo", None)
                seeds_mod["profile"] = {
                    "mean_valence": prof2.get("mean_valence"),
                    "mean_energy": prof2.get("mean_energy"),
                }

                tracks = _robust_recommend(spif, label_used, seeds_mod, limit=6, exclude_uris=exclude)

            # ✅ AJOUT apprentissage feedback / reranking
            stats = st.session_state.get("fb_stats") or {}
            tracks = rerank_tracks_with_feedback(tracks, label_used, stats, top_k=6)

        except Exception as e:
            st.session_state["api_err"].add(str(e))
            tracks = []

        st.session_state["last_gen_ms"] = int((time.time() - t0) * 1000)
        st.session_state["S2_tracks"] = tracks
        st.session_state["S2_uris"] = [t.get("uri") for t in tracks if t.get("uri")]

        if label_used:
            st.session_state["emotion_history"].setdefault(label_used, set()).update(st.session_state["S2_uris"])

    # 8) Affichage + logging feedback
    if "S2_tracks" in st.session_state:
        playback_and_log_ui(
            pid, "S2",
            st.session_state.get("S2_face", {"label": None, "probs": {}}),
            st.session_state.get("S2_text", {"label": None, "probs": {}}),
            st.session_state.get("S2_label") or "neutral",
            st.session_state["S2_tracks"],
            spif
        )

def _main_app():
    st.divider()
    if st.session_state.get("spotify_ready"):
        st.success("Spotify prêt.")
    else:
        st.info("Pas encore connecté à Spotify.")
    miss = st.session_state.get("missing_scopes") or []
    
    if not st.session_state.get("spotify_ready"):
        return
    st.session_state.setdefault("lib_only", False)
    st.session_state["lib_only"] = st.checkbox(
        "Limiter aux titres de ma bibliothèque",
        value=st.session_state["lib_only"],
        key="lib_only_global",
    )
    scenario = st.radio("Scénario", ["S1 — Fusion rapide", "S2 — Chat guidé"], horizontal=True, key="scenario_main")
    spif = st.session_state.get("spif"); pid = st.session_state.get("pid", "")
    if scenario.startswith("S1"):
        scenario_s1(spif, pid)
    else:
        scenario_s2(spif, pid)
    with st.expander("Diagnostics"):
        st.write("Historique émotions:", {k: len(v) for k, v in (st.session_state.get("emotion_history") or {}).items()})
        if st.session_state.get("api_err"):
            for e in sorted(st.session_state["api_err"]):
                st.write("•", e)
        st.write("Cache features:", len(st.session_state.get("af_cache_by_id", {})),
                 "| Fails:", len(st.session_state.get("af_fail_ids", set())),
                 "| Block:", st.session_state.get("block_spotify_features"))
        if "seeds" in st.session_state:
            st.json(st.session_state["seeds"])

def _connection_ui():
    st.subheader("Connexion / Participant")
    col1, col2, col3 = st.columns([2, 1.2, 1])
    with col1:
        pid = st.text_input("Participant ID", value=st.session_state.get("pid", ""), key="pid_input_main")
        if pid: st.session_state["pid"] = pid
    with col2:
        st.selectbox("Market", ["FR","US","GB","DE","ES","IT","CA"], index=0, key="market_select_main")
    with col3:
        if st.button("Reset session", key="reset_btn"):
            reset_session()
    cA, cB = st.columns(2)
    with cA:
        if st.button("Connexion Spotify / Seeds", key="connect_btn_main"):
            if st.session_state.get("pid"):
                spif = get_sp_client(st.session_state["pid"])
                st.session_state["spif"] = spif
                set_sp_global(spif.sp)
                set_cache_salt(st.session_state["pid"])
                try:
                    me = spif.sp.current_user()
                    miss = spif.missing_scopes(REQUIRED_SCOPES) or []
                    st.session_state["missing_scopes"] = miss
                    st.session_state["spotify_ready"] = True
                    st.success(f"Connecté: {me.get('display_name')} ({me.get('id')})")
                    if miss:
                        st.warning("Scopes manquants: " + ", ".join(miss))
                        manual_auth_ui(spif, key_prefix="auth_missing_inline")
                    else:
                        ensure_seeds(spif)
                except Exception as e:
                    st.session_state["spotify_ready"] = False
                    st.session_state["missing_scopes"] = []
                    st.error(f"Erreur OAuth: {e}")
                    manual_auth_ui(spif, key_prefix="auth_error_inline")
            else:
                st.warning("Renseignez un Participant ID.")
    with cB:
        if st.button("Purger caches", key="purge_btn_main"):
            n = purge_all_cache()
            st.info(f"Caches supprimés: {n}. Reconnectez-vous si nécessaire.")
    if not st.session_state.get("spotify_ready"):
        st.caption("Connectez-vous pour afficher les scénarios.")
    else:
        st.caption("Connexion Spotify active.")

def run_app():
    try:
        normalize_redirect_uri()
    except Exception:
        pass
    _connection_ui()
    _main_app()

run_app()
