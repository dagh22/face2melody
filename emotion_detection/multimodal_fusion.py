"""
Functions for combining emotion predictions from multiple modalities.

These utilities allow you to fuse facial emotion distributions with
text-based emotion distributions using simple weighted averaging. For
scenarios where one modality may be more reliable than the other, you
can adjust the weight parameter accordingly.
"""

from typing import Dict, Tuple


def fuse_distributions(
    face_dist: Dict[str, float],
    text_dist: Dict[str, float],
    face_weight: float = 0.6,
    text_weight: float = 0.4,
) -> Dict[str, float]:
    """
    Combine two emotion distributions using weighted averaging.

    Parameters
    ----------
    face_dist : dict
        Probability distribution over emotions from facial detection.
    text_dist : dict
        Probability distribution over emotions from text analysis.
    face_weight : float
        Relative weight for the facial modality. Must be between 0 and 1.
    text_weight : float
        Relative weight for the text modality. Must be between 0 and 1. Should
        sum to 1 - face_weight for meaningful results.

    Returns
    -------
    dict
        A fused distribution over emotions.
    """
    # Normalize weights to sum to 1
    total_weight = face_weight + text_weight
    fw = face_weight / total_weight
    tw = text_weight / total_weight

    emotions = set(face_dist.keys()) | set(text_dist.keys())
    fused: Dict[str, float] = {}
    for emo in emotions:
        f_val = face_dist.get(emo, 0.0)
        t_val = text_dist.get(emo, 0.0)
        fused[emo] = fw * f_val + tw * t_val
    return fused


def dominant_emotion(distribution: Dict[str, float]) -> Tuple[str, float]:
    """
    Return the emotion with the highest probability and its probability.
    """
    if not distribution:
        return "neutral", 1.0
    emo, prob = max(distribution.items(), key=lambda kv: kv[1])
    return emo, prob
