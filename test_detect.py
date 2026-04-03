import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Initialize the Spotify client
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

# Fetch audio features
sp.audio_features(["4uLU6hMCjMI75M1A2tKUQC"])
