# python
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv(override=True)

from recommender.spotify_interface import SpotifyInterface


def generate_playlist(final_emotion: str, n_tracks: int = 20) -> List[Dict]:
    si = SpotifyInterface()
    tracks = si.recommend_by_emotion(final_emotion, market="CA")
    return tracks[:n_tracks] if tracks else []
