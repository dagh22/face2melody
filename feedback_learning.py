# feedback_learning.py
import os, json
from collections import defaultdict

def _safe_bool(x):
    return bool(x) if x is not None else False

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default
    
def _norm_emo(emo: str) -> str:
    emo = (emo or "").strip().lower()
    return emo if emo in {"happy", "sad", "angry", "neutral"} else "neutral"


def load_feedback_stats(log_path: str):
    """
    Lit logs experiment_*.jsonl et construit :
    - disliked_uris (set)
    - liked_uris (set)
    - rating_map[uri] = avg_rating
    - emo_stats[emotion][uri] = dict(like_count, dislike_count, avg_rating)
    """
    disliked = set()
    liked = set()
    rating_sum = defaultdict(int)
    rating_cnt = defaultdict(int)

    emo_like = defaultdict(lambda: defaultdict(int))
    emo_dis = defaultdict(lambda: defaultdict(int))
    emo_rating_sum = defaultdict(lambda: defaultdict(int))
    emo_rating_cnt = defaultdict(lambda: defaultdict(int))

    if not log_path or not os.path.exists(log_path):
        return {
            "disliked": disliked,
            "liked": liked,
            "rating_avg": {},
            "emo_stats": {}
        }

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            uri = obj.get("track_uri")
            if not uri:
                continue
            
            emo = _norm_emo(obj.get("fused_label") or obj.get("final_emotion"))
            like = _safe_bool(obj.get("like"))
            dislike = _safe_bool(obj.get("dislike"))
            rating = _safe_int(obj.get("rating"), default=0)

            if dislike:
                disliked.add(uri)
                emo_dis[emo][uri] += 1
            if like:
                liked.add(uri)
                emo_like[emo][uri] += 1

            if rating > 0:
                rating_sum[uri] += rating
                rating_cnt[uri] += 1

                emo_rating_sum[emo][uri] += rating
                emo_rating_cnt[emo][uri] += 1

    rating_avg = {u: (rating_sum[u] / rating_cnt[u]) for u in rating_cnt}

    emo_stats = {}
    for emo in set(list(emo_like.keys()) + list(emo_dis.keys()) + list(emo_rating_sum.keys())):
        emo_stats[emo] = {}
        uris = set(list(emo_like[emo].keys()) + list(emo_dis[emo].keys()) + list(emo_rating_sum[emo].keys()))
        for u in uris:
            rs = emo_rating_sum[emo][u]
            rc = emo_rating_cnt[emo][u]
            emo_stats[emo][u] = {
                "like": emo_like[emo][u],
                "dislike": emo_dis[emo][u],
                "avg_rating": (rs / rc) if rc else None
            }

    return {
        "disliked": disliked,
        "liked": liked,
        "rating_avg": rating_avg,
        "emo_stats": emo_stats
    }

def score_track(uri: str, final_emotion: str, stats: dict):
    """
    Score simple :
    - disliked => -999 (exclusion)
    - like => +2
    - avg_rating => + (avg_rating - 3) * 0.8
    - stats émotion => bonus si la piste a bien marché pour cette émotion
    """
    if not uri:
        return 0.0

    emo = (final_emotion or "").lower().strip()
    disliked = stats.get("disliked", set())
    liked = stats.get("liked", set())
    rating_avg = stats.get("rating_avg", {})
    emo_stats = stats.get("emo_stats", {})

    if uri in disliked:
        return -999.0

    s = 0.0
    if uri in liked:
        s += 2.0

    r = rating_avg.get(uri)
    if r is not None:
        s += (float(r) - 3.0) * 0.8

    # Bonus spécifique à l’émotion
    es = emo_stats.get(emo, {}).get(uri)
    if es:
        if es.get("dislike", 0) > 0:
            s -= 3.0
        if es.get("like", 0) > 0:
            s += 1.5
        ar = es.get("avg_rating")
        if ar is not None:
            s += (float(ar) - 3.0) * 0.6

    return s

def rerank_tracks_with_feedback(
    tracks: list,
    final_emotion: str,
    stats: dict,
    top_k=None,
    strict_mode: bool = False,
    min_rating: int = 4,
    require_like: bool = True,
):
    """
    strict_mode=True :
      - garde seulement les titres déjà validés (like=True) ET rating >= min_rating
      - exclut systématiquement les disliked
    sinon : rerank normal (scoring)
    """
    stats = stats or {}
    disliked = stats.get("disliked", set())
    emo = _norm_emo(final_emotion)

    # stats par émotion : emo_stats[emo][uri] = {"like":..,"dislike":..,"avg_rating":..}
    emo_stats = (stats.get("emo_stats") or {}).get(emo, {})

    if strict_mode:
        kept = []
        for t in (tracks or []):
            uri = t.get("uri")
            if not uri or uri in disliked:
                continue

            es = emo_stats.get(uri) or {}
            lk = int(es.get("like", 0) or 0)
            dk = int(es.get("dislike", 0) or 0)
            ar = es.get("avg_rating")

            if dk > 0:
                continue
            if require_like and lk <= 0:
                continue
            if ar is None or float(ar) < float(min_rating):
                continue

            kept.append(t)

        return kept[:top_k] if top_k else kept

    # mode normal (score)
    scored = []
    for t in (tracks or []):
        uri = t.get("uri")
        if not uri or uri in disliked:
            continue
        sc = score_track(uri, final_emotion, stats)
        if sc <= -900:
            continue
        scored.append((sc, t))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = [t for _, t in scored]
    return out[:top_k] if top_k else out
