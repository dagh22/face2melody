# -*- coding: utf-8 -*-
# facial_utils.py (version corrigée)
from recommender.emotion_detector import is_ready, get_status, emotions_from_frame
from recommender.emotion_detector import set_backend

set_backend("opencv")

from typing import Dict, List, Optional, Any
from collections import Counter
import cv2
import time
import numpy as np

# NOTE: EMOTION_MAP / autres constantes éventuelles inchangées si tu en avais plus haut.


def _norm01(x: float, lo: float, hi: float) -> float:
    """Mappe linéairement x vers [0,1] sur [lo,hi]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _try_smile_score(frame) -> Optional[float]:
    """
    Calcule un score de sourire dans [0,1] avec MediaPipe FaceMesh si disponible.
    Heuristique: largeur bouche / distance inter-oculaire + ouverture verticale.
    Retourne None si indisponible/échec.
    """
    try:
        import mediapipe as mp

        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True
        ) as fm:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)
            if not res.multi_face_landmarks:
                return None
            lm = res.multi_face_landmarks[0].landmark

            # Indices: lèvres 61 (gauche), 291 (droite), 13 (sup), 14 (inf)
            # Yeux externes 33 (gauche), 263 (droit)
            def pt(i):
                return np.array([lm[i].x, lm[i].y], dtype=np.float32)

            mouth_l, mouth_r = pt(61), pt(291)
            lip_u, lip_d = pt(13), pt(14)
            eye_l, eye_r = pt(33), pt(263)
            interocular = np.linalg.norm(eye_r - eye_l) + 1e-6
            mouth_w = np.linalg.norm(mouth_r - mouth_l)
            mouth_h = np.linalg.norm(lip_d - lip_u)
            w_norm = mouth_w / interocular
            h_norm = mouth_h / interocular
            # Normalisations empiriques
            w_s = _norm01(w_norm, 0.35, 0.55)
            h_s = _norm01(h_norm, 0.02, 0.08)
            smile = 0.7 * w_s + 0.3 * h_s
            return float(max(0.0, min(1.0, smile)))
    except Exception:
        return None


def _apply_smile_boost(
    dist: Dict[str, float], smile_level: float, max_transfer: float = 0.25
) -> Dict[str, float]:
    """
    Transfère une petite masse de {neutral,sad} vers happy proportionnelle au sourire.
    """
    d = dict(dist)
    transfer = min(max_transfer, 0.5 * float(smile_level))
    if transfer <= 1e-6:
        return d
    take_from_neutral = min(d.get("neutral", 0.0), transfer * 0.6)
    d["neutral"] = max(0.0, d.get("neutral", 0.0) - take_from_neutral)
    take_from_sad = min(d.get("sad", 0.0), transfer * 0.4)
    d["sad"] = max(0.0, d.get("sad", 0.0) - take_from_sad)
    d["happy"] = d.get("happy", 0.0) + take_from_neutral + take_from_sad
    # Renormalisation
    s = sum(d.values()) or 1.0
    for k in d:
        d[k] /= s
    return d


def _apply_hint_bias(
    dist: Dict[str, float], hint: Optional[str], strength: float = 0.6
) -> Dict[str, float]:
    """
    Biaise légèrement la distribution selon un hint externe:
      - hint='negative': upweight sad/angry, downweight neutral/happy
      - hint='positive': upweight happy, downweight neutral/sad
    """
    if not hint:
        return dict(dist)
    s = max(0.0, min(1.0, float(strength)))
    d = dict(dist)
    factors = {k: 1.0 for k in ["happy", "sad", "angry", "neutral"]}
    if hint == "negative":
        factors["sad"] += 0.25 * s
        factors["angry"] += 0.12 * s
        factors["neutral"] -= 0.18 * s
        factors["happy"] -= 0.08 * s
    elif hint == "positive":
        factors["happy"] += 0.25 * s
        factors["neutral"] -= 0.15 * s
        factors["sad"] -= 0.10 * s
    # Clamp pour éviter l'annulation complète
    for k in factors:
        factors[k] = max(0.05, factors[k])
    for k in d:
        d[k] = max(0.0, d.get(k, 0.0) * factors.get(k, 1.0))
    ssum = sum(d.values()) or 1.0
    for k in d:
        d[k] /= ssum
    return d


def _reduce_scores(
    raw: Dict[str, float],
    *,
    neutral_weight: float = 0.92,
    temperature: float = 0.85,
    # nouveaux paramètres anti-biais
    sad_downweight: float = 0.90,
    fear_to_neutral: float = 0.60,
    happy_upweight: float = 1.05,
) -> Dict[str, float]:
    """
    Réduit les scores bruts DeepFace vers {happy, sad, angry, neutral},
    applique une pondération légère de 'neutral' et un sharpening par température.
    Anti-biais:
      - Répartit 'fear' vers neutral (fear_to_neutral)
      - Downweight global de 'sad', léger upweight de 'happy'
    """
    reduced = {"happy": 0.0, "sad": 0.0, "angry": 0.0, "neutral": 0.0}
    rare_weights = {
        "fear": 0.55,
        "disgust": 0.9,
        "surprise": 0.9,
    }
    for k, v in raw.items():
        lab = k.lower()
        val = float(v)
        if lab == "fear":
            # Distribue fear: majoritairement vers neutral pour éviter le faux 'sad'
            v_adj = val * rare_weights.get("fear", 1.0)
            reduced["sad"] += v_adj * (1.0 - fear_to_neutral)
            reduced["neutral"] += v_adj * fear_to_neutral
            continue
        base = lab if lab in ["happy", "sad", "angry", "neutral"] else "neutral"
        w = rare_weights.get(lab, 1.0)
        reduced[base] = reduced.get(base, 0.0) + val * w

    # Ajustements anti-biais
    reduced["neutral"] *= neutral_weight
    reduced["sad"] *= sad_downweight
    reduced["happy"] *= happy_upweight

    # Normalisation avant sharpening
    s = sum(reduced.values()) or 1.0
    for k in reduced:
        reduced[k] /= s

    # Sharpening par température (T < 1 => distribution plus piquée)
    if temperature and abs(temperature - 1.0) > 1e-6:
        invT = 1.0 / max(temperature, 1e-3)
        sharpened = {k: max(v, 1e-9) ** invT for k, v in reduced.items()}
        z = sum(sharpened.values()) or 1.0
        reduced = {k: v / z for k, v in sharpened.items()}

    return reduced


def _pick_with_neutral_guard(
    scores: Dict[str, float],
    neutral_threshold: float = 0.45,  # 0.5 -> 0.45 pour réduire les neutres
    top_margin: float = 0.10,  # 0.12 -> 0.10
    delta_vs_neutral: float = 0.08,  # 0.1  -> 0.08
    # nouveaux: seuils par label (facultatifs)
    per_label_thresholds=None,
) -> str:
    """
    Sélectionne l'émotion top-1 avec garde-fous globaux, surchargés au besoin
    par per_label_thresholds[label] = {neutral_threshold, top_margin, delta_vs_neutral}.
    """
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_val = ordered[0]
    second_val = ordered[1][1] if len(ordered) > 1 else 0.0

    # Seuils spécifiques au label si fournis
    pl = (per_label_thresholds or {}).get(top_label) or {}
    thr = pl.get("neutral_threshold", neutral_threshold)
    tm = pl.get("top_margin", top_margin)
    dvn = pl.get("delta_vs_neutral", delta_vs_neutral)

    if top_label != "neutral":
        if top_val < thr:
            return "neutral"
        if (top_val - second_val) < tm:
            return "neutral"
        if (top_val - scores.get("neutral", 0.0)) < dvn:
            return "neutral"
    return top_label


def _region_valid(analysis: Dict[str, Any]) -> bool:
    """Vérifie que la région de visage détectée est plausible."""
    region = analysis.get("region") or {}
    try:
        w = int(region.get("w", 0))
        h = int(region.get("h", 0))
    except Exception:
        w, h = 0, 0
    return (w * h) >= 40 * 40  # seuil simple


def _frame_quality_ok(
    frame,
    min_blur_var: float = 45.0,
    brightness_range: tuple[int, int] = (30, 220),
) -> bool:
    """
    Ignore les frames floues/sombres qui tirent l'EMA vers 'neutral'.
    - Flou: variance du Laplacien >= min_blur_var
    - Luminosité: moyenne en niveaux de gris dans l'intervalle autorisé
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception:
        return False
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_luma = float(gray.mean())
    return (blur_var >= min_blur_var) and (
        brightness_range[0] <= mean_luma <= brightness_range[1]
    )


def _should_update(
    reduced: Dict[str, float], min_conf: float = 0.32, min_margin: float = 0.06
) -> bool:
    """
    N'actualise pas l'EMA si la distribution est trop plate / incertaine.
    - top1 >= min_conf OU (top1 - top2) >= min_margin
    """
    ordered = sorted(reduced.values(), reverse=True)
    top1 = ordered[0] if ordered else 0.0
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    return (top1 >= min_conf) or ((top1 - top2) >= min_margin)


def _overlay_info(frame, emotion: str, scores: Dict[str, float]) -> None:
    """
    Overlays emotion and scores on the given frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame, f"Emotion: {emotion}", (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA
    )
    y_offset = 60
    for label, score in scores.items():
        cv2.putText(
            frame,
            f"{label}: {score:.2f}",
            (10, y_offset),
            font,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_offset += 20


def detect_facial_emotion(
    duration: int = 10,
    st_placeholder=None,
    analyze_every: float = 0.7,
    record_video: bool = False,
    camera_index: int = 0,
    neutral_threshold: float = 0.45,  # 0.5 -> 0.45
    ema_alpha: float = 0.6,
    detector_backend: str = "mediapipe",  # <-- défaut corrigé
    top_margin: float = 0.10,  # 0.12 -> 0.10
    delta_vs_neutral: float = 0.08,  # 0.1  -> 0.08
    min_update_conf: float = 0.25,  # 0.35 -> 0.32
    min_update_margin: float = 0.05,  # 0.07 -> 0.06
    # Nouveaux paramètres (compatibles rétro):
    reduce_temperature: float = 0.85,
    neutral_downweight: float = 0.92,
    min_blur_var: float = 20.0,
    brightness_range: tuple[int, int] = (20, 240),
    return_details: bool = False,
    # Anti-biais supplémentaires:
    sad_downweight: float = 0.90,
    fear_to_neutral: float = 0.60,
    happy_upweight: float = 1.05,
    per_label_thresholds=None,
    # Sourire (optionnel):
    smile_boost_enabled: bool = True,
    smile_threshold: float = 0.35,
    smile_ema_alpha: float = 0.6,
    # Persistance minimale pour 'sad':
    sad_persistence: int = 2,
    # Hints utilisateur (chatbot):
    user_affect_hint: Optional[str] = None,  # "negative" | "positive" | None
    hint_strength: float = 0.6,
) -> Any:
    """
    Capture la caméra et agrège les émotions via une EMA + vote majoritaire,
    avec garde-fous et contrôle qualité pour limiter le biais vers 'neutral'.
    Retourne soit un str (label), soit un dict détaillé si return_details=True.
    """
    from deepface import DeepFace

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra")

    writer = None
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = None
    if record_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        out_path = f"facial_emotion_{int(time.time())}.mp4"
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    start_time = time.time()
    last_analyze = 0.0

    ema_scores: Dict[str, float] = {
        # Initialisation uniforme pour éviter le démarrage pro-'neutral'
        "happy": 0.25,
        "sad": 0.25,
        "angry": 0.25,
        "neutral": 0.25,
    }
    votes: List[str] = []
    last_display = "neutral"
    smile_ema: float = 0.0
    sad_streak: int = 0

    # Persistance 'sad' assouplie si l'utilisateur dit "je ne vais pas bien"
    sad_persistence_eff = sad_persistence
    if user_affect_hint == "negative":
        sad_persistence_eff = max(
            1, int(round(sad_persistence * (1.0 - 0.5 * hint_strength)))
        )

    # Seuils par label par défaut si non fournis: happy plus permissif, sad plus strict
    if per_label_thresholds is None:
        per_label_thresholds = {
            "happy": {
                "neutral_threshold": max(0.35, neutral_threshold - 0.07),
                "top_margin": max(0.06, top_margin - 0.02),
                "delta_vs_neutral": max(0.05, delta_vs_neutral - 0.03),
            },
            "sad": {
                "neutral_threshold": min(0.60, neutral_threshold + 0.07),
                "top_margin": top_margin + 0.03,
                "delta_vs_neutral": delta_vs_neutral + 0.05,
            },
        }

    # Initialiser pl_thr pour la décision finale même si 0 itération
    pl_thr = per_label_thresholds

    total_frames_analyzed = 0

    while (time.time() - start_time) < duration:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        do_analyze = (now - last_analyze) >= analyze_every

        if do_analyze:
            last_analyze = now
            try:
                # --- ANALYSE EMOTIONNELLE (backend mediapipe, fallback opencv) ---
                try:
                    analysis = DeepFace.analyze(
                        img_path=frame,  # numpy BGR ok
                        actions=["emotion"],
                        enforce_detection=False,
                        detector_backend=detector_backend or "mediapipe",
                    )
                except Exception:
                    analysis = DeepFace.analyze(
                        img_path=frame,
                        actions=["emotion"],
                        enforce_detection=False,
                        detector_backend="opencv",
                    )

                if isinstance(analysis, list):
                    analysis = analysis[0] if analysis else {}

                # Contrôle de la région + qualité
                region_ok = _region_valid(analysis) if analysis else False
                quality_ok = _frame_quality_ok(
                    frame, min_blur_var=min_blur_var, brightness_range=brightness_range
                )

                # Smile (optionnel)
                smile_score = _try_smile_score(frame) if smile_boost_enabled else None
                if smile_score is not None:
                    smile_ema = (
                        smile_ema_alpha * smile_score
                        + (1.0 - smile_ema_alpha) * smile_ema
                    )

                raw_scores = (analysis.get("emotion", {}) or {}) if analysis else {}
                reduced = _reduce_scores(
                    {k.lower(): float(v) for k, v in raw_scores.items()},
                    neutral_weight=neutral_downweight,
                    temperature=reduce_temperature,
                    sad_downweight=sad_downweight,
                    fear_to_neutral=fear_to_neutral,
                    happy_upweight=happy_upweight,
                )
                if st_placeholder is not None:
                    top_label = max(reduced, key=reduced.get)
                    top_val = reduced[top_label]
                    st_placeholder.caption(
                        f"region_ok={region_ok} | quality_ok={quality_ok} | top={top_label}:{top_val:.2f} | updates={len(votes)}"
                    )

                # Boost 'happy' si sourire fort et frame valide
                if (
                    region_ok
                    and quality_ok
                    and smile_boost_enabled
                    and (smile_ema >= smile_threshold)
                ):
                    reduced = _apply_smile_boost(reduced, smile_ema)

                # Biais léger issu du chatbot (prior utilisateur)
                if user_affect_hint in ("negative", "positive"):
                    reduced = _apply_hint_bias(reduced, user_affect_hint, hint_strength)

                updated = False
                if (
                    region_ok
                    and quality_ok
                    and _should_update(
                        reduced, min_conf=min_update_conf, min_margin=min_update_margin
                    )
                ):
                    # EMA pour lisser
                    for k in ema_scores.keys():
                        ema_scores[k] = ema_alpha * reduced.get(k, 0.0) + (
                            1.0 - ema_alpha
                        ) * ema_scores.get(k, 0.0)
                    updated = True
                    total_frames_analyzed += 1

                # Seuils dynamiques selon sourire
                pl_thr = per_label_thresholds
                if smile_boost_enabled and (smile_ema >= smile_threshold):
                    pl_thr = dict(per_label_thresholds)  # copie superficielle
                    h = dict(pl_thr.get("happy", {}))
                    h["neutral_threshold"] = max(
                        0.30, (h.get("neutral_threshold", neutral_threshold) - 0.05)
                    )
                    h["top_margin"] = max(
                        0.05, (h.get("top_margin", top_margin) - 0.02)
                    )
                    h["delta_vs_neutral"] = max(
                        0.04, (h.get("delta_vs_neutral", delta_vs_neutral) - 0.02)
                    )
                    pl_thr["happy"] = h
                    s = dict(pl_thr.get("sad", {}))
                    s["neutral_threshold"] = min(
                        0.65, (s.get("neutral_threshold", neutral_threshold) + 0.05)
                    )
                    s["top_margin"] = s.get("top_margin", top_margin) + 0.02
                    s["delta_vs_neutral"] = (
                        s.get("delta_vs_neutral", delta_vs_neutral) + 0.02
                    )
                    pl_thr["sad"] = s

                # Ajouts: assouplir/resserrer selon le hint utilisateur
                if user_affect_hint == "negative":
                    pl_thr = dict(pl_thr)
                    s = dict(pl_thr.get("sad", {}))
                    s["neutral_threshold"] = max(
                        0.30,
                        (
                            s.get("neutral_threshold", neutral_threshold)
                            - 0.07 * hint_strength
                        ),
                    )
                    s["top_margin"] = max(
                        0.04, (s.get("top_margin", top_margin) - 0.03 * hint_strength)
                    )
                    s["delta_vs_neutral"] = max(
                        0.03,
                        (
                            s.get("delta_vs_neutral", delta_vs_neutral)
                            - 0.03 * hint_strength
                        ),
                    )
                    pl_thr["sad"] = s
                elif user_affect_hint == "positive":
                    pl_thr = dict(pl_thr)
                    h = dict(pl_thr.get("happy", {}))
                    h["neutral_threshold"] = max(
                        0.30,
                        (
                            h.get("neutral_threshold", neutral_threshold)
                            - 0.07 * hint_strength
                        ),
                    )
                    h["top_margin"] = max(
                        0.04, (h.get("top_margin", top_margin) - 0.03 * hint_strength)
                    )
                    h["delta_vs_neutral"] = max(
                        0.03,
                        (
                            h.get("delta_vs_neutral", delta_vs_neutral)
                            - 0.03 * hint_strength
                        ),
                    )
                    pl_thr["happy"] = h

                # Décision courante
                current = _pick_with_neutral_guard(
                    ema_scores,
                    neutral_threshold=neutral_threshold,
                    top_margin=top_margin,
                    delta_vs_neutral=delta_vs_neutral,
                    per_label_thresholds=pl_thr,
                )

                # Votes avec persistance stricte pour 'sad'
                if updated:
                    if current == "sad":
                        sad_streak += 1
                        if sad_streak >= max(1, int(sad_persistence_eff)):
                            votes.append(current)
                    else:
                        sad_streak = 0
                        votes.append(current)

                last_display = current

                # Affichage
                if st_placeholder is not None:
                    vis = frame.copy()
                    _overlay_info(vis, last_display, ema_scores)
                    # width="stretch" n'est pas valide; utilise use_column_width
                    st_placeholder.image(vis, channels="BGR", use_column_width=True)
                if writer is not None:
                    vis = frame.copy()
                    _overlay_info(vis, last_display, ema_scores)
                    writer.write(vis)

            except Exception as e:
                if st_placeholder is not None:
                    st_placeholder.caption(f"Erreur analyse: {e}")

        time.sleep(0.005)

    cap.release()
    if writer is not None:
        writer.release()

    # Si aucune frame n'a pu être vraiment analysée, fallback neutre explicite
    if total_frames_analyzed == 0:
        if return_details:
            return {
                "label": "neutral",
                "ema": ema_scores,
                "votes": votes,
                "smile_ema": smile_ema,
                "user_affect_hint": user_affect_hint,
                "reason": "no_frames",
            }
        return "neutral"

    # Décision finale
    final_from_ema = _pick_with_neutral_guard(
        ema_scores,
        neutral_threshold=neutral_threshold,
        top_margin=top_margin,
        delta_vs_neutral=delta_vs_neutral,
        per_label_thresholds=(
            per_label_thresholds
            if (not smile_boost_enabled or smile_ema < smile_threshold)
            else pl_thr
        ),
    )
    final_label = final_from_ema

    if votes:
        counts = Counter(votes)
        if final_from_ema == "neutral":
            non_neutral_counts = {k: v for k, v in counts.items() if k != "neutral"}
            if non_neutral_counts:
                best_non_neutral = max(non_neutral_counts, key=non_neutral_counts.get)
                ratio = non_neutral_counts[best_non_neutral] / max(len(votes), 1)
                # Assouplissement: accepte un non-neutral si ratio >= 0.4 ou au moins 2 votes
                if (ratio >= 0.40) or (non_neutral_counts[best_non_neutral] >= 2):
                    final_label = best_non_neutral
                else:
                    final_label = "neutral"
            else:
                final_label = "neutral"
        else:
            final_label = max(counts, key=counts.get)
            # Si sourire soutenu, éviter un 'sad' marginal
            if final_label == "sad" and (
                smile_boost_enabled and smile_ema >= (smile_threshold + 0.1)
            ):
                if ema_scores.get("happy", 0.0) >= ema_scores.get("sad", 0.0) - 0.04:
                    final_label = "happy"

    # Post-ajustement final via hint si la sortie reste 'neutral'
    if final_label == "neutral" and user_affect_hint in ("negative", "positive"):
        if user_affect_hint == "negative":
            if (
                ema_scores.get("sad", 0.0)
                >= ema_scores.get("neutral", 0.0) - 0.08 * hint_strength
            ) or (("votes" in locals()) and any(v == "sad" for v in votes)):
                final_label = "sad"
        elif user_affect_hint == "positive":
            if (
                ema_scores.get("happy", 0.0)
                >= ema_scores.get("neutral", 0.0) - 0.08 * hint_strength
            ) or (("votes" in locals()) and any(v == "happy" for v in votes)):
                final_label = "happy"

    # Option: détails pour la fusion/debug
    if return_details:
        return {
            "label": final_label,
            "ema": ema_scores,
            "votes": votes,
            "smile_ema": smile_ema,
            "user_affect_hint": user_affect_hint,
        }

    return final_label
