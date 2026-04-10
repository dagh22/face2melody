# ── Last.fm Interface — Face2Melody V3 ────────────────────────────────────────
# Alternative gratuite à Spotify (sans Premium requis).
# Utilise l'API Last.fm via pylast pour récupérer des recommandations
# basées sur l'émotion détectée par la fusion trimode.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import random
from typing import Dict, List, Optional

# Importation conditionnelle pour éviter un crash si pylast n'est pas installé
try:
    import pylast
    _PYLAST_OK = True
except ImportError:
    _PYLAST_OK = False

# ── Mapping émotion → tags Last.fm ───────────────────────────────────────────
_LASTFM_TAGS: Dict[str, List[str]] = {
    "happy":   ["happy", "feel-good", "upbeat", "summer", "danceable"],
    "sad":     ["melancholy", "sad", "emotional", "introspective", "rainy day"],
    "angry":   ["aggressive", "intense", "hard rock", "metal", "powerful"],
    "neutral": ["chill", "ambient", "focus", "lo-fi", "relax"],
}

# Nombre de tracks à récupérer par tag avant d'échantillonner
_FETCH_PER_TAG = 15


class LastFMInterface:
    """
    Client Last.fm pour la recommandation musicale basée sur l'émotion.

    Deux modes de fonctionnement :
    - Anonyme   : recommandations par tags émotionnels (tag.getTopTracks)
    - Personnalisé : personnalisation via l'historique utilisateur Last.fm
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        username: Optional[str] = None,
    ) -> None:
        if not _PYLAST_OK:
            raise ImportError(
                "pylast n'est pas installé. Exécute : pip install pylast>=5.1.0"
            )
        self.network = pylast.LastFMNetwork(
            api_key=api_key,
            api_secret=api_secret,
        )
        self.username = username.strip() if username else None

    # ── API publique ──────────────────────────────────────────────────────────

    def get_recommendations(
        self, emotion: str, limit: int = 5
    ) -> List[Dict[str, str]]:
        """
        Retourne une liste de tracks adaptés à l'émotion détectée.

        Stratégie 1 (username fourni) :
            - Récupère les artistes écoutés récemment par l'utilisateur
            - Trouve des artistes similaires via l'émotion courante
        Stratégie 2 (anonyme) :
            - Utilise tag.getTopTracks() avec les tags émotionnels mappés

        Retourne : [{"title": str, "artist": str, "url": str, "image_url": str}]
        """
        if self.username:
            try:
                return self._recommendations_from_user(emotion, limit)
            except Exception:
                # Dégradation gracieuse vers les tags si erreur utilisateur
                pass
        return self._recommendations_from_tags(emotion, limit)

    def get_user_top_artists(self, limit: int = 5) -> List[str]:
        """
        Retourne les noms des artistes les plus écoutés par l'utilisateur
        sur les 3 derniers mois. Nécessite un username.
        """
        if not self.username:
            return []
        try:
            user = self.network.get_user(self.username)
            top = user.get_top_artists(period=pylast.PERIOD_3MONTHS, limit=limit)
            return [item.item.name for item in top]
        except Exception:
            return []

    def get_similar_artists(self, artist_name: str, limit: int = 5) -> List[str]:
        """
        Retourne des artistes similaires à artist_name via Last.fm.
        """
        try:
            artist = self.network.get_artist(artist_name)
            similar = artist.get_similar(limit=limit)
            return [item.item.name for item in similar]
        except Exception:
            return []

    # ── Méthodes internes ─────────────────────────────────────────────────────

    def _recommendations_from_tags(
        self, emotion: str, limit: int
    ) -> List[Dict[str, str]]:
        """
        Recommande via tag.getTopTracks() pour chaque tag émotionnel.
        Mélange les résultats pour éviter la répétition.
        """
        tags = _LASTFM_TAGS.get(emotion, _LASTFM_TAGS["neutral"])
        all_tracks: List[Dict[str, str]] = []

        for tag_name in tags:
            try:
                tag = self.network.get_tag(tag_name)
                top_tracks = tag.get_top_tracks(limit=_FETCH_PER_TAG)
                for item in top_tracks:
                    track = item.item
                    entry = self._track_to_dict(track)
                    if entry:
                        all_tracks.append(entry)
            except Exception:
                continue

        # Déduplique par (title, artist) et mélange
        seen: set = set()
        unique: List[Dict[str, str]] = []
        for t in all_tracks:
            key = (t["title"].lower(), t["artist"].lower())
            if key not in seen:
                seen.add(key)
                unique.append(t)

        random.shuffle(unique)
        return unique[:limit]

    def _recommendations_from_user(
        self, emotion: str, limit: int
    ) -> List[Dict[str, str]]:
        """
        Personnalise les recommandations via l'historique de l'utilisateur :
        1. Récupère les artistes récents de l'utilisateur
        2. Pour chaque artiste, récupère des artistes similaires
        3. Filtre les tracks de ces artistes similaires par tags émotionnels
        """
        user = self.network.get_user(self.username)

        # Récupère les artistes récents (3 mois)
        try:
            top_artists = user.get_top_artists(
                period=pylast.PERIOD_3MONTHS, limit=3
            )
            seed_artists = [item.item.name for item in top_artists]
        except Exception:
            seed_artists = []

        if not seed_artists:
            return self._recommendations_from_tags(emotion, limit)

        # Trouve des artistes similaires
        similar_names: List[str] = []
        for name in seed_artists[:2]:
            similar_names.extend(self.get_similar_artists(name, limit=4))

        # Récupère les top tracks de ces artistes similaires
        tracks: List[Dict[str, str]] = []
        for artist_name in similar_names[:6]:
            try:
                artist = self.network.get_artist(artist_name)
                top = artist.get_top_tracks(limit=3)
                for item in top:
                    entry = self._track_to_dict(item.item)
                    if entry:
                        tracks.append(entry)
            except Exception:
                continue

        if not tracks:
            return self._recommendations_from_tags(emotion, limit)

        random.shuffle(tracks)
        return tracks[:limit]

    def _track_to_dict(self, track) -> Optional[Dict[str, str]]:
        """
        Convertit un objet pylast.Track en dict normalisé.
        Retourne None si les données essentielles manquent.
        """
        try:
            title = track.title or track.get_name()
            artist = track.artist.name if track.artist else ""
            if not title or not artist:
                return None
            url = track.get_url() or ""
            # Tentative de récupération de la cover (peut échouer)
            image_url = ""
            try:
                image_url = track.get_cover_image() or ""
            except Exception:
                pass
            return {
                "title": title,
                "artist": artist,
                "url": url,
                "image_url": image_url,
            }
        except Exception:
            return None


# ── Vérification de disponibilité ────────────────────────────────────────────

def lastfm_available() -> bool:
    """Retourne True si pylast est installé."""
    return _PYLAST_OK
