from __future__ import annotations
import json, os, re, time, unicodedata, random
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Set

import streamlit as st
import spotipy
from spotipy import Spotify, SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials

# NLP (pour answers_to_text_dist)
from emotion_detection.text_utils import emotion_distribution

__all__ = [
    "fuse",
    "_fuse_heuristic",
    "build_user_profile",
    "get_personal_seeds",
    "recommend_for_emotion",
    "append_log_line",
    "audio_features_by_uri",
    "now_iso",
    "set_sp_global",
    "set_cache_salt",
    # nouveaux exports
    "answers_to_text_dist",
    "answers_to_valence_energy",
]

LABEL_MAP = {
    "happy": "happy",
    "joy": "happy",
    "joie": "happy",
    "sad": "sad",
    "sadness": "sad",
    "tristesse": "sad",
    "angry": "angry",
    "anger": "angry",
    "colere": "angry",
    "colère": "angry",
    "neutral": "neutral",
    "neutre": "neutral",
}
EMOS = ["happy", "sad", "angry", "neutral"]

TARGETS = {
    "happy": {"target_valence": 0.85, "target_energy": 0.75, "min_tempo": 110},
    "sad": {"target_valence": 0.20, "target_energy": 0.35, "max_tempo": 90},
    "angry": {"target_valence": 0.30, "target_energy": 0.90, "min_tempo": 100},
    "neutral": {"target_valence": 0.55, "target_energy": 0.50},
}
SAFE_GENRES = {
    "pop",
    "dance",
    "hip-hop",
    "rock",
    "indie",
    "electronic",
    "house",
    "r-n-b",
    "trap",
    "latin",
    "chill",
    "edm",
    "afropop",
}

# -----------------------
# Utils génériques
# -----------------------


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    # retirer accents
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_probs(p: Dict[str, float]) -> Dict[str, float]:
    out = {k: 0.0 for k in EMOS}
    for k, v in (p or {}).items():
        nk = LABEL_MAP.get((k or "").lower())
        if nk in out:
            out[nk] += float(v or 0.0)
    s = sum(out.values()) or 1.0
    return {k: v / s for k, v in out.items()}


def _fuse_heuristic(face_probs: Dict[str, float], text_probs: Dict[str, float], w_face=0.4):
    """Fusion pondérée visage/texte V1 — préservée comme fallback pour l'agent V2."""
    wf = float(max(0.0, min(1.0, w_face)))
    fp = normalize_probs(face_probs)
    tp = normalize_probs(text_probs)
    fused = {k: wf * fp[k] + (1 - wf) * tp[k] for k in EMOS}
    s = sum(fused.values()) or 1.0
    fused = {k: v / s for k, v in fused.items()}
    return fused, max(fused, key=fused.get)


def fuse(
    face_probs: Dict[str, float],
    text_probs: Dict[str, float],
    w_face: float = 0.4,
    text_raw: Optional[str] = None,
) -> Tuple[Dict[str, float], str]:
    """
    V2 : délègue à EmotionFusionAgent (LLM Claude).
    Retombe sur _fuse_heuristic() si l'agent est indisponible.
    Stocke l'AgentResult dans st.session_state['agent_result'] pour l'affichage XAI.
    Retourne (fused_distribution, dominant_emotion) pour rétrocompatibilité.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)
    try:
        from agent_logic import get_agent
        result = get_agent().analyze(
            face_probs, text_probs, text_raw=text_raw, w_face=w_face
        )
    except Exception as exc:
        _log.warning("agent_logic indisponible (%s), fallback V1.", exc)
        from agent_logic import AgentResult, _ANCHOR
        fd, dom = _fuse_heuristic(face_probs, text_probs, w_face)
        v, a = _ANCHOR.get(dom, (0.55, 0.50))
        result = AgentResult(
            valence=v, arousal=a,
            dominant_emotion=dom,
            fused_distribution=fd,
            reasoning=f"[V1 fallback] Émotion dominante : {dom}.",
            from_fallback=True,
        )
    try:
        st.session_state["agent_result"] = result
    except Exception:
        pass
    return result.fused_distribution, result.dominant_emotion


# -----------------------
# Q/R chatbot → features
# -----------------------

VAL_POS = {
    "heureux",
    "joyeux",
    "bien",
    "bien!",
    "motive",
    "motivé",
    "contente",
    "content",
    "cool",
    "genial",
    "geniale",
    "super",
    "ok",
    "satisfait",
    "satisfaite",
    "ca va",
    "ça va",
    "pas mal",
    "plutot bien",
    "plutôt bien",
}
VAL_NEG = {
    "triste",
    "sombre",
    "deprime",
    "deprimé",
    "deprimee",
    "pas bien",
    "mal",
    "fatigue",
    "fatiguee",
    "fatigué",
    "fatiguee",
    "fatiguees",
}
ENER_LOW = {"calme", "fatigue", "fatiguee", "lent", "repos", "tranquille", "faible"}
ENER_HIGH = {
    "elevee",
    "eleve",
    "haute",
    "forte",
    "intense",
    "energie",
    "energiee",
    "excite",
    "excitee",
    "excité",
    "excitee",
    "dynamique",
}


def answers_to_valence_energy(answers: List[str]) -> Dict[str, float]:
    """
    Convertit une liste de réponses en métriques Spotify-friendly:
    {'mean_valence': float, 'mean_energy': float}
    """
    v_hits = 0
    e_hits = 0
    v = 0.5  # point neutre
    e = 0.5

    for raw in answers or []:
        t = _norm_text(str(raw))

        # Nombres 0..10 -> énergie déclarée
        if t.isdigit():
            n = int(t)
            if 0 <= n <= 10:
                e = max(e, n / 10.0)
                continue

        # Oui / Non -> valence légère
        if t in {"oui", "yes", "y"}:
            v = max(v, 0.7)
        if t in {"non", "no", "n"}:
            v = min(v, 0.4)

        # Lexiques
        if any(k in t for k in VAL_POS):
            v_hits += 1
        if any(k in t for k in VAL_NEG):
            v_hits -= 1
        if any(k in t for k in ENER_HIGH):
            e_hits += 1
        if any(k in t for k in ENER_LOW):
            e_hits -= 1

    # Convertit les hits en score doux
    v = min(max(0.5 + 0.15 * v_hits, 0.0), 1.0)
    e = min(max(e + 0.15 * e_hits, 0.0), 1.0)
    return {"mean_valence": round(v, 2), "mean_energy": round(e, 2)}


def answers_to_text_dist(answers: List[str]) -> Dict[str, float]:
    """
    Agrège les réponses textuelles et calcule une distribution émotionnelle
    via le module NLP (emotion_detection.text_utils.emotion_distribution).
    """
    joined = " | ".join(answers or [])
    return emotion_distribution(joined)


# -----------------------
# Seeds / profils Spotify
# -----------------------


def canonicalize_seed_genres(sp: Spotify, raw: List[str], min_n=2) -> List[str]:
    MAP = {
        "french rap": "hip-hop",
        "hip hop": "hip-hop",
        "r&b": "r-n-b",
        "rnb": "r-n-b",
        "latin pop": "latin",
        "afro pop": "afropop",
        "afro-pop": "afropop",
    }
    out: List[str] = []
    for g in raw or []:
        g0 = (g or "").lower().strip()
        g1 = MAP.get(g0, g0)
        g2 = unicodedata.normalize("NFD", g1).encode("ascii", "ignore").decode()
        g2 = g2.replace("&", "and")
        cands = [
            g2,
            g2.replace(" ", "-"),
            re.sub(r"[^a-z0-9\-]", "", g2.replace(" ", "-")),
        ]
        for c in cands:
            if c in SAFE_GENRES and c not in out:
                out.append(c)
                break
        if len(out) >= 5:
            break
    if len(out) < min_n:
        for g in ["pop", "dance", "hip-hop", "rock", "indie", "electronic"]:
            if g not in out:
                out.append(g)
            if len(out) >= min_n:
                break
    return out[:5]


def build_user_profile(sp: Spotify) -> Dict:
    if not st.session_state.get("spotify_ready"):
        return {"artists": [], "tracks": [], "genres": [], "profile": {}}
    artist_ids: List[str] = []
    track_ids: List[str] = []
    raw_genres: List[str] = []
    try:
        arts = sp.current_user_top_artists(limit=10).get("items", [])
        artist_ids = [a["id"] for a in arts[:3] if a.get("id")]
        for a in arts:
            raw_genres.extend(a.get("genres") or [])
    except SpotifyException:
        pass
    try:
        tops = sp.current_user_top_tracks(limit=15).get("items", [])
        for t in tops[:5]:
            if t.get("id"):
                track_ids.append(t["id"])
        for t in tops:
            for a in t.get("artists", []):
                try:
                    art = sp.artist(a["id"])
                    raw_genres.extend(art.get("genres") or [])
                except SpotifyException:
                    continue
    except SpotifyException:
        pass
    genres = canonicalize_seed_genres(sp, raw_genres, 2) or ["pop", "dance"]
    return {
        "artists": artist_ids[:3],
        "tracks": track_ids[:3],
        "genres": genres[:3],
        "profile": {"mean_valence": None, "mean_energy": None, "mean_tempo": None},
    }


def get_personal_seeds(sp: Spotify) -> Tuple[List[str], List[str]]:
    prof = build_user_profile(sp)
    return prof.get("artists", [])[:2], prof.get("genres", [])[:2]


def _merge_targets_with_user(emo: Dict, prof: Dict) -> Dict:
    t = emo.copy()
    mv = prof.get("mean_valence")
    me = prof.get("mean_energy")
    mt = prof.get("mean_tempo")
    if mv is not None:
        t["target_valence"] = (t.get("target_valence", mv) + mv) / 2
    if me is not None:
        t["target_energy"] = (t.get("target_energy", me) + me) / 2
    if mt is not None:
        if "min_tempo" in t:
            t["min_tempo"] = int((t["min_tempo"] + mt) / 2)
        if "max_tempo" in t:
            t["max_tempo"] = int((t["max_tempo"] + mt) / 2)
    return t


def _rerank_by_user_profile(sp: Spotify, tracks: List[Dict], prof: Dict) -> List[Dict]:
    # Placeholder : tu pourras implémenter un vrai reranking si besoin
    return tracks


# -----------------------
# Recommandation Spotify
# -----------------------


def recommend_for_emotion(
    sp: Spotify,
    market: str,
    emotion_label: str,
    seeds: Dict,
    limit=6,
    prefer_preview=True,
    rerank=True,
    exclude_uris: Optional[Set[str]] = None,
) -> List[Dict]:
    label = LABEL_MAP.get(
        (emotion_label or "").lower(), (emotion_label or "neutral").lower()
    )
    targets = TARGETS.get(label, TARGETS["neutral"]).copy()
    # différenciation
    if label == "happy":
        targets.setdefault("min_tempo", 105)
        targets["target_energy"] = max(0.7, targets.get("target_energy", 0.7))
        targets["target_valence"] = max(0.75, targets.get("target_valence", 0.75))
    elif label == "sad":
        targets["target_energy"] = min(0.4, targets.get("target_energy", 0.4))
        targets["target_valence"] = min(0.35, targets.get("target_valence", 0.35))
        targets["max_tempo"] = min(targets.get("max_tempo", 95), 95)

    prof = (seeds or {}).get("profile") or {}
    merged = _merge_targets_with_user(targets, prof)

    seed_art = list((seeds or {}).get("artists", []) or [])[:2]
    seed_trk = list((seeds or {}).get("tracks", []) or [])[:2]
    seed_gen = canonicalize_seed_genres(sp, (seeds or {}).get("genres", []) or [], 1)[
        :3
    ]
    while len(seed_art) + len(seed_trk) + len(seed_gen) > 5 and seed_gen:
        seed_gen.pop()

    # validate tracks
    v: List[str] = []
    for tid in seed_trk:
        try:
            sp.track(tid)
            v.append(tid)
        except SpotifyException:
            continue
    seed_trk = v[:2]

    def _call(sa, sg, st_ids):
        return sp.recommendations(
            seed_artists=sa or None,
            seed_tracks=st_ids or None,
            seed_genres=sg or None,
            limit=limit * 5,
            market=market,
            **merged,
        ).get("tracks", [])

    tracks: List[Dict] = []
    try:
        tracks = _call(seed_art, seed_gen, seed_trk)
    except SpotifyException:
        tracks = []
    if not tracks and seed_trk:
        try:
            tracks = _call(seed_art, seed_gen, [])
        except SpotifyException:
            tracks = []
    if not tracks and seed_art:
        try:
            tracks = _call([], seed_gen, [])
        except SpotifyException:
            tracks = []
    if not tracks:
        for g in seed_gen or ["pop", "dance"]:
            try:
                r = sp.search(
                    q=f'genre:"{g}"', type="track", limit=limit * 5, market=market
                )
                tracks += r.get("tracks", {}).get("items", [])
            except SpotifyException:
                continue
    if not tracks:
        try:
            r = sp.search(q=label, type="track", limit=limit * 5, market=market)
            tracks += r.get("tracks", {}).get("items", [])
        except SpotifyException:
            pass

    disliked = st.session_state.get("disliked_uris", set())
    seen: Set[str] = set()
    uniq: List[Dict] = []
    for t in tracks:
        u = t.get("uri")
        if not u:
            continue
        if exclude_uris and u in exclude_uris:
            continue
        if u in disliked:
            continue
        if u not in seen:
            seen.add(u)
            uniq.append(t)

    if prefer_preview:
        with_prev = [t for t in uniq if t.get("preview_url")]
        without = [t for t in uniq if not t.get("preview_url")]
        uniq = with_prev + without

    if rerank:
        uniq = _rerank_by_user_profile(sp, uniq, prof)

    if len(uniq) < limit:
        try:
            extra = sp.search(q=f"{label} mood", type="track", limit=20, market=market)
            for t in extra.get("tracks", {}).get("items", []):
                u = t.get("uri")
                if not u or u in seen:
                    continue
                if exclude_uris and u in exclude_uris:
                    continue
                if u in disliked:
                    continue
                uniq.append(t)
                seen.add(u)
                if len(uniq) >= limit:
                    break
        except SpotifyException:
            pass

    random.seed(int(time.time()))
    random.shuffle(uniq)
    return uniq[:limit]


# -----------------------
# Logs utilitaires
# -----------------------


def append_log_line(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -----------------------
# Audio features cache + fallback client_credentials
# -----------------------

_sp_global: Optional[Spotify] = None
_cache_salt: str = ""

# client Spotipy en client_credentials (fallback)
_SP_CC: Optional[spotipy.Spotify] = None


def set_sp_global(sp: Spotify):
    global _sp_global
    _sp_global = sp


def set_cache_salt(s: str):
    global _cache_salt
    _cache_salt = str(s or "")


@lru_cache(maxsize=512)
def _audio_features_by_id_cached(track_id: str, salt: str) -> Dict:
    """
    Version cache basée sur le client utilisateur global.
    Utilisée par d'autres parties (ex: _safe_audio_features).
    """
    if _sp_global is None or not track_id or not st.session_state.get("spotify_ready"):
        return {}
    try:
        feats_list = _sp_global.audio_features([track_id])
    except SpotifyException:
        return {}
    if not feats_list or not feats_list[0]:
        return {}
    feats = feats_list[0]
    return {
        "valence": feats.get("valence"),
        "energy": feats.get("energy"),
        "tempo": feats.get("tempo"),
    }


def _get_client_credentials_sp() -> Optional[spotipy.Spotify]:
    """
    Retourne un client Spotipy basé sur client_credentials.
    Utilise SPOTIPY_CLIENT_ID et SPOTIPY_CLIENT_SECRET du .env.
    """
    global _SP_CC
    if _SP_CC is not None:
        return _SP_CC

    cid = os.getenv("SPOTIPY_CLIENT_ID")
    csec = os.getenv("SPOTIPY_CLIENT_SECRET")
    if not cid or not csec:
        return None

    auth_manager = SpotifyClientCredentials(
        client_id=cid,
        client_secret=csec,
    )
    _SP_CC = spotipy.Spotify(auth_manager=auth_manager)
    return _SP_CC


def audio_features_by_uri(sp: Spotify, uri: str) -> Dict:
    """
    Essaie de récupérer les audio_features d'une piste à partir de son URI.

    1) Essai avec le client utilisateur `sp` (si fourni).
    2) En cas d'erreur ou si vide, essai via un client 'client_credentials'
       basé sur SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET.

    Retourne un dict brut tel que renvoyé par Spotify, ou {} si échec.
    """
    try:
        if not uri:
            return {}
        track_id = uri.split(":")[-1]
        if not track_id:
            return {}

        # --- 1) Essai avec le client utilisateur ---
        if sp is not None:
            try:
                af_list = sp.audio_features([track_id]) or []
                if af_list and af_list[0]:
                    return af_list[0]
            except SpotifyException:
                pass
            except Exception:
                pass

        # --- 2) Fallback client_credentials ---
        sp_cc = _get_client_credentials_sp()
        if sp_cc is None:
            return {}

        try:
            af_list = sp_cc.audio_features([track_id]) or []
            if af_list and af_list[0]:
                return af_list[0]
        except Exception:
            return {}

        return {}
    except Exception:
        return {}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
