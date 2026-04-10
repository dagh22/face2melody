# ── YouTube Search — Face2Melody V4 ───────────────────────────────────────────
# Recherche optionnelle via YouTube Data API v3.
# Requiert YOUTUBE_API_KEY dans .env (gratuit, 10 000 unités/jour).
# Fallback silencieux si la clé est absente ou si le quota est dépassé.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

_YT_URL  = "https://www.googleapis.com/youtube/v3/search"
_TIMEOUT = 5  # secondes


def search_youtube(query: str, api_key: str, max_results: int = 3) -> List[Dict]:
    """
    Retourne des vidéos YouTube embedables pour la query donnée.

    Args:
        query:       Requête de recherche (ex: "Daft Punk electronic happy")
        api_key:     Clé YouTube Data API v3
        max_results: Nombre de vidéos à retourner (max recommandé : 3)

    Returns:
        Liste de dicts : [{"video_id": str, "title": str, "thumbnail": str}]
        Liste vide en cas d'erreur ou si api_key est vide.
    """
    if not api_key or not query:
        return []

    try:
        resp = requests.get(
            _YT_URL,
            params={
                "q":               query,
                "type":            "video",
                "key":             api_key,
                "maxResults":      max_results,
                "part":            "snippet",
                "videoEmbeddable": "true",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("items") or []
    except Exception as exc:
        logger.debug("YouTube API error: %s", exc)
        return []

    results: List[Dict] = []
    for item in items:
        video_id = (item.get("id") or {}).get("videoId")
        snippet  = item.get("snippet") or {}
        if video_id:
            results.append({
                "video_id":  video_id,
                "title":     snippet.get("title", ""),
                "thumbnail": (
                    (snippet.get("thumbnails") or {})
                    .get("medium", {})
                    .get("url", "")
                ),
            })

    return results
