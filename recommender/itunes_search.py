# ── iTunes Search — Face2Melody V4 ────────────────────────────────────────────
# Recherche gratuite via l'API iTunes (aucune authentification requise).
# Utilisé pour enrichir les tracks Last.fm avec un embed Apple Music.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_ITUNES_URL = "https://itunes.apple.com/search"
_TIMEOUT    = 4  # secondes — court pour ne pas bloquer la génération


def get_apple_embed_url(title: str, artist: str) -> Optional[str]:
    """
    Retourne une URL embed Apple Music via l'API iTunes (gratuite, sans auth).

    Args:
        title:  Titre de la chanson (ex: "Blinding Lights")
        artist: Nom de l'artiste   (ex: "The Weeknd")

    Returns:
        URL de la forme "https://embed.music.apple.com/fr/album/{collectionId}?i={trackId}"
        ou None si aucun résultat ou erreur réseau.
    """
    if not title or not artist:
        return None

    try:
        resp = requests.get(
            _ITUNES_URL,
            params={
                "term":   f"{artist} {title}",
                "media":  "music",
                "entity": "song",
                "limit":  1,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
    except Exception as exc:
        logger.debug("iTunes API error: %s", exc)
        return None

    if not results:
        return None

    item          = results[0]
    track_id      = item.get("trackId")
    collection_id = item.get("collectionId")

    if not track_id or not collection_id:
        return None

    return f"https://embed.music.apple.com/fr/album/{collection_id}?i={track_id}"


def get_apple_embed_url_batch(tracks: list[dict]) -> list[dict]:
    """
    Enrichit une liste de dicts Last.fm avec la clé 'embed_url'.
    Modifie la liste en place. Silencieux en cas d'erreur.

    Chaque dict doit avoir les clés 'title' et 'artist'.
    Retourne la liste pour chaînage.
    """
    for track in tracks:
        track["embed_url"] = (
            get_apple_embed_url(track.get("title", ""), track.get("artist", "")) or ""
        )
    return tracks
