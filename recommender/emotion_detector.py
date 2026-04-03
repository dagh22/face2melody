# -*- coding: utf-8 -*-
"""
recommender/emotion_detector.py
Petit wrapper autour de DeepFace pour:
- gérer l'activation via F2M_ENABLE_DEEPFACE
- choisir un backend de détection robuste (mediapipe par défaut, fallback opencv)
- fournir une API simple: is_ready(), get_status(), set_backend(), analyze_frame()
"""

from __future__ import annotations
import os
import threading
from typing import Any, Dict, Optional

# Etat du module
_LOCK = threading.Lock()
_ENABLED = os.getenv("F2M_ENABLE_DEEPFACE", "0") == "1"
_BACKEND = os.getenv(
    "F2M_DETECTOR_BACKEND", "mediapipe"
)  # 'mediapipe' | 'opencv' | 'retinaface' | ...
_STATUS = (
    "DeepFace désactivé (F2M_ENABLE_DEEPFACE=0)" if not _ENABLED else "DeepFace prêt"
)


# DeepFace est importé à la demande pour éviter les délais à l'import
def _import_deepface():
    from deepface import DeepFace  # type: ignore

    return DeepFace


# -------------------------
# API de contrôle / statut
# -------------------------


def set_enabled(flag: bool) -> None:
    """Active/désactive le module au runtime (utile pour tests)."""
    global _ENABLED, _STATUS
    with _LOCK:
        _ENABLED = bool(flag)
        _STATUS = "DeepFace prêt" if _ENABLED else "DeepFace désactivé (runtime)"


def is_ready() -> bool:
    """Retourne True si l'analyse émotionnelle est activée."""
    return _ENABLED


def get_status() -> str:
    """Chaîne de statut affichable dans l'UI."""
    return _STATUS


def set_backend(name: str) -> None:
    """Change le backend de détection (ex: 'mediapipe', 'opencv', 'retinaface')."""
    global _BACKEND
    with _LOCK:
        _BACKEND = str(name or "mediapipe")


def get_backend() -> str:
    return _BACKEND


# -------------------------
# Fonctions d'analyse
# -------------------------


def analyze_frame(
    frame,
    *,
    actions: Optional[list[str]] = None,
    detector_backend: Optional[str] = None,
    enforce_detection: bool = False,
) -> Dict[str, Any]:
    """
    Analyse une frame (numpy BGR) avec DeepFace.
    Retourne un dict de la forme DeepFace (keys: emotion, dominant_emotion, region, etc.)
    En cas d'erreur ou module désactivé, lève une RuntimeError ou renvoie {} si enforce_detection=False.
    """
    if not _ENABLED:
        raise RuntimeError("DeepFace désactivé : F2M_ENABLE_DEEPFACE=0")

    if frame is None:
        raise ValueError("Frame invalide (None)")

    actions = actions or ["emotion"]
    backend = detector_backend or _BACKEND

    DeepFace = _import_deepface()

    # premier essai: backend demandé (par défaut mediapipe)
    try:
        out = DeepFace.analyze(
            img_path=frame,  # numpy array (BGR) accepté
            actions=actions,
            detector_backend=backend,
            enforce_detection=enforce_detection,
        )
    except Exception:
        # fallback léger: opencv (fiable sur la plupart des postes)
        out = DeepFace.analyze(
            img_path=frame,
            actions=actions,
            detector_backend="opencv",
            enforce_detection=False,
        )

    # DeepFace peut retourner une liste si plusieurs visages; on prend le premier
    if isinstance(out, list):
        out = out[0] if out else {}

    # Normalisation légère
    if not isinstance(out, dict):
        out = {}

    return out


def emotions_from_frame(
    frame,
    *,
    detector_backend: Optional[str] = None,
    enforce_detection: bool = False,
) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """
    Raccourci pratique: retourne (label, distribution, meta)
      - label: string parmi 'happy', 'sad', 'angry', 'neutral', ... (selon DeepFace)
      - distribution: dict des scores d'émotions bruts renvoyés par DeepFace
      - meta: dict contenant au minimum 'region' si disponible
    Lève RuntimeError si désactivé.
    """
    out = analyze_frame(
        frame,
        actions=["emotion"],
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
    )

    emotions = out.get("emotion") or {}
    label = out.get("dominant_emotion") or "neutral"
    meta = {"region": out.get("region", {}), "raw": out}

    # Casting float + clés en lower
    dist = (
        {str(k).lower(): float(v) for k, v in emotions.items()}
        if isinstance(emotions, dict)
        else {}
    )

    return str(label).lower(), dist, meta


def analyze_emotion(
    frame,
    *,
    detector_backend: Optional[str] = None,
    enforce_detection: bool = False,
) -> Dict[str, Any]:
    """
    Compatibilité rétro : même signature/nom que l'ancien code.
    Retourne un dict DeepFace-like (keys: emotion, dominant_emotion, region, ...).
    """
    return analyze_frame(
        frame,
        actions=["emotion"],
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
    )


def reload_deepface(force: bool = False) -> str:
    """
    Compat: app.py appelle reload_deepface() pour (ré)initialiser DeepFace.
    Ici on fait un test d'import + on met à jour le statut.
    Retourne une chaîne de statut affichable.
    """
    global _STATUS
    try:
        _ = _import_deepface()  # teste l'import
        if not _ENABLED and force:
            # si on veut forcer l'activation à chaud
            set_enabled(True)
        _STATUS = f"DeepFace prêt (backend={_BACKEND})"
    except Exception as e:
        _STATUS = f"DeepFace indisponible: {e}"
    return _STATUS


# -------------------------
# Mini test (exécution directe)
# -------------------------
if __name__ == "__main__":
    import cv2

    print(
        f"[emotion_detector] enabled={_ENABLED}, backend={_BACKEND}, status={_STATUS}"
    )
    if not _ENABLED:
        print("Activez avec: export F2M_ENABLE_DEEPFACE=1")
        raise SystemExit(0)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d'ouvrir la caméra")
        raise SystemExit(1)

    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Aucune frame capturée")
        raise SystemExit(1)

    try:
        lbl, dist, meta = emotions_from_frame(frame)
        print("dominant:", lbl)
        print("dist    :", dist)
        print("region  :", meta.get("region"))
    except Exception as e:
        print("Erreur d'analyse:", e)
