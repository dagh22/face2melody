# emotion_detection/text_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

__all__ = ["detect_emotion_from_text", "emotion_distribution"]

# ---------------------------
# Utils
# ---------------------------

def _norm(s: str) -> str:
    """Normalise une chaîne (minuscules, accents retirés, espaces compactés)."""
    if not s:
        return ""
    s = s.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    # remplace ponctuation multiple par espaces
    s = re.sub(r"[\s/|]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_probs(p: Dict[str, float]) -> Dict[str, float]:
    """Renvoie un dict normalisé sur {happy,sad,angry,neutral}."""
    keys = ["happy", "sad", "angry", "neutral"]
    out = {k: float(p.get(k, 0.0) or 0.0) for k in keys}
    s = sum(out.values())
    if s <= 0:
        return {"happy": 0.0, "sad": 0.0, "angry": 0.0, "neutral": 1.0}
    return {k: v / s for k, v in out.items()}


# ---------------------------
# Règles FR : idiomes et lexiques
# ---------------------------

# Idiomes positifs atténués (→ happy léger)
POS_POSITIVE_IDIOMS: List[str] = [
    r"\bpas\s+mal\b",
    r"\b(?:ne\s+)?vais?\s+pas\s+mal\b",
    r"\b(?:ca|ça)\s+va\s+pas\s+mal\b",
    r"\bpas\s+trop\s+mal\b",
    r"\bplutot\s+bien\b",  # "plutôt" devient "plutot" après normalisation
]

# Idiomes négatifs atténués (→ sad)
SAD_IDIOMS: List[str] = [
    r"\bpas\s+terrible\b",
    r"\bpas\s+top\b",
    r"\bpas\s+ouf\b",
    r"\bbof+\b",
    r"\b(?:je\s+)?(?:ne\s+)?vais?\s+pas\s+bien\b",
    r"\b(?:ca|ça)\s+(?:ne\s+)?va\s+pas\b",
]

# Négation d’adjectifs positifs (→ sad prioritaire)
# ex: "je ne suis pas heureux", "pas content", "pas joyeux"
NEGATE_POS_PATTERNS: List[str] = [
    r"\b(?:ne\s+)?suis\s+pas\s+(?:heureux|heureuse|content|contente|joyeux|joyeuse|bien)\b",
    r"\bpas\s+(?:heureux|heureuse|content|contente|joyeux|joyeuse|bien)\b",
]

# Lexiques simples
HAPPY_WORDS = {
    "heureux", "heureuse", "content", "contente", "joie", "joyeux", "joyeuse",
    "ravi", "ravie", "bien", "cool", "satisfait", "satisfaite", "enthousiaste"
}
SAD_WORDS = {
    "triste", "tristesse", "malheureux", "malheureuse", "deprime", "deprimee",
    "deprime", "deprimee", "deprimee", "deprim", "chagrin", "morose", "fatigue",
    "fatiguee", "fatiguee", "fatiguee", "lassitude", "deprimee"
}
# (on tolère des duplications bénignes, la normalisation gère)
ANGRY_WORDS = {
    "colere", "fache", "fachee", "enerve", "enervee", "agace", "agacee",
    "furieux", "furieuse", "rage", "enrage", "irrite", "irritee", "mecontent", "mecontente"
}

# Indices valence/énergie doux (guidage S2 par ex.)
VALENCE_POS_HINTS = {"lumineux", "bright", "plaisant", "positif", "positifs"}
VALENCE_NEG_HINTS = {"sombre", "dark", "deprime", "triste"}
ENERGY_POS_HINTS = {"energie", "energetique", "dynamique", "intense", "bouge"}
ENERGY_NEG_HINTS = {"calme", "repos", "relax", "tranquille", "lent"}


# ---------------------------
# Détection principale
# ---------------------------

def detect_emotion_from_text(text: str) -> Dict[str, Dict[str, float] | str]:
    """
    Analyse textuelle FR heuristique → retourne {"label": <str>, "probs": {happy,sad,angry,neutral}}.
    - idiomes atténués gérés en priorité
    - négation d’adjectifs positifs => sad
    - comptage lexiques happy/sad/angry + pondération légère
    """
    t = _norm(text)

    if not t:
        return {"label": "neutral", "probs": {"happy": 0.0, "sad": 0.0, "angry": 0.0, "neutral": 1.0}}

    # 1) Idiomes positifs atténués → happy léger
    for pat in POS_POSITIVE_IDIOMS:
        if re.search(pat, t):
            base = {"happy": 0.65, "neutral": 0.35}
            p = _normalize_probs(base)
            return {"label": max(p, key=p.get), "probs": p}

    # 2) Idiomes négatifs atténués → sad
    for pat in SAD_IDIOMS:
        if re.search(pat, t):
            base = {"sad": 0.80, "neutral": 0.20}
            p = _normalize_probs(base)
            return {"label": "sad", "probs": p}

    # 3) Négation d’un adjectif positif → sad
    for pat in NEGATE_POS_PATTERNS:
        if re.search(pat, t):
            base = {"sad": 0.85, "neutral": 0.15}
            p = _normalize_probs(base)
            return {"label": "sad", "probs": p}

    # 4) Comptage lexiques
    def _count_any(tokens: set[str]) -> int:
        c = 0
        for w in tokens:
            if re.search(rf"\b{re.escape(w)}\b", t):
                c += 1
        return c

    h_cnt = _count_any(HAPPY_WORDS)
    s_cnt = _count_any(SAD_WORDS)
    a_cnt = _count_any(ANGRY_WORDS)

    # 5) Heuristique valence/énergie douce (n’influence pas directement le label
    #    ici, mais tu peux t’en servir côté fusion ou préférences)
    v_hint = 0
    e_hint = 0
    if any(re.search(rf"\b{w}\b", t) for w in VALENCE_POS_HINTS): v_hint += 1
    if any(re.search(rf"\b{w}\b", t) for w in VALENCE_NEG_HINTS): v_hint -= 1
    if any(re.search(rf"\b{w}\b", t) for w in ENERGY_POS_HINTS):  e_hint += 1
    if any(re.search(rf"\b{w}\b", t) for w in ENERGY_NEG_HINTS):  e_hint -= 1

    # 6) Construction d’une distribution simple
    base = {"happy": 0.0, "sad": 0.0, "angry": 0.0, "neutral": 0.0}

    # Pondération douce pour chaque hit
    base["happy"] += 0.6 * h_cnt
    base["sad"]   += 0.6 * s_cnt
    base["angry"] += 0.7 * a_cnt

    # Indices valence/énergie → petite translation vers happy/sad
    if v_hint > 0:
        base["happy"] += 0.4
    elif v_hint < 0:
        base["sad"] += 0.4

    # Aucune info → neutre
    if base["happy"] == base["sad"] == base["angry"] == 0.0:
        base["neutral"] = 1.0
    else:
        base["neutral"] += 0.2  # un petit fond neutre

    probs = _normalize_probs(base)
    label = max(probs, key=probs.get)
    return {"label": label, "probs": probs}


# ---------------------------
# Distribution multi-phrases
# ---------------------------

def emotion_distribution(text_or_messages: str | List[str]) -> Dict[str, float]:
    """
    Version "distribution pure":
      - prend un texte ou une liste de messages
      - agrège les probs normalisées
      - renvoie {happy,sad,angry,neutral} normalisé
    """
    if isinstance(text_or_messages, str):
        items = [text_or_messages]
    else:
        items = list(text_or_messages or [])

    agg = {"happy": 0.0, "sad": 0.0, "angry": 0.0, "neutral": 0.0}
    for m in items:
        res = detect_emotion_from_text(m or "")
        p = res.get("probs") or {}
        for k in agg.keys():
            agg[k] += float(p.get(k, 0.0) or 0.0)

    return _normalize_probs(agg)
