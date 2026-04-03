import os, random, logging
from typing import Dict, List, Optional
from urllib.parse import urlparse
import socket

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

import spotipy
from spotipy import SpotifyException
from spotipy.oauth2 import SpotifyPKCE, SpotifyOAuth

logger = logging.getLogger(__name__)

SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")  # optionnel
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")

_missing = [
    k
    for k, v in {
        "SPOTIPY_CLIENT_ID": SPOTIPY_CLIENT_ID,
        "SPOTIPY_REDIRECT_URI": SPOTIPY_REDIRECT_URI,
    }.items()
    if not v
]
if _missing:
    raise RuntimeError(
        f"Variables d'environnement manquantes: {', '.join(_missing)} (SECRET optionnel pour PKCE)"
    )

_FORCE_PKCE = os.getenv("SPOTIPY_PKCE_ONLY") == "1"
DEFAULT_SCOPE = "user-library-read user-top-read playlist-modify-private playlist-modify-public"
EMOTION_PARAMS: Dict[str, Dict] = {
    "happy": {
        "seed_genres": ["pop", "dance", "happy"],
        "target_valence": 0.85,
        "target_energy": 0.75,
    },
    "sad": {
        "seed_genres": ["acoustic", "sad", "piano"],
        "target_valence": 0.20,
        "target_energy": 0.35,
        "max_tempo": 90,
    },
    "angry": {
        "seed_genres": ["metal", "rock", "hard-rock"],
        "target_valence": 0.30,
        "target_energy": 0.90,
        "min_tempo": 100,
    },
    "neutral": {
        "seed_genres": ["indie", "pop", "alt-rock"],
        "target_valence": 0.55,
        "target_energy": 0.50,
    },
    "_default": {
        "seed_genres": ["pop", "indie", "rock"],
        "target_valence": 0.60,
        "target_energy": 0.60,
    },
}


def _port_free(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.25)
            return s.connect_ex((host, port)) != 0
    except Exception:
        return False


def _choose_redirect_uri() -> str:
    """
    Sélectionne le premier Redirect URI dont le port local est libre.
    Accepte 127.0.0.1 et localhost (127.0.0.1 prioritaire).
    """
    raw = os.getenv("SPOTIPY_REDIRECT_URIS") or os.getenv("SPOTIPY_REDIRECT_URI") or ""
    all_candidates = [u.strip() for u in raw.split(",") if u.strip()]
    candidates = []
    for uri in all_candidates:
        try:
            p = urlparse(uri)
            host = (p.hostname or "").lower()
            if host in ("127.0.0.1", "localhost"):
                candidates.append(uri)
        except Exception:
            continue
    if not candidates:
        raise RuntimeError(
            "Aucun Redirect URI valide défini. Exemple: http://127.0.0.1:8503/callback (à ajouter dans le Dashboard Spotify)."
        )
    # Préférer 127.0.0.1 avant localhost
    candidates = sorted(
        candidates,
        key=lambda u: 0 if (urlparse(u).hostname or "").lower() == "127.0.0.1" else 1,
    )
    for uri in candidates:
        try:
            p = urlparse(uri)
            host = (p.hostname or "127.0.0.1").lower()
            check_host = "127.0.0.1" if host == "localhost" else host
            port = p.port or (443 if p.scheme == "https" else 80)
            if _port_free(check_host, port):
                logger.info("Redirect URI sélectionné: %s (port libre)", uri)
                return uri
        except Exception:
            continue
    logger.warning(
        "Aucun port libre parmi %s. Utilisation par défaut: %s",
        candidates,
        candidates[0],
    )
    return candidates[0]


def _build_auth_manager(scope: str, cache_path: str, show_dialog: bool):
    cid = os.getenv("SPOTIPY_CLIENT_ID")
    secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    redirect = _choose_redirect_uri()
    use_pkce = _FORCE_PKCE or not secret
    if use_pkce:
        logger.info("Authentification Spotify en mode PKCE (secret absent ou forcé).")
        return SpotifyPKCE(
            client_id=cid,
            redirect_uri=redirect,
            scope=scope,
            show_dialog=show_dialog,
            cache_path=cache_path,
        )
    return SpotifyOAuth(
        client_id=cid,
        client_secret=secret,
        redirect_uri=redirect,
        scope=scope,
        show_dialog=show_dialog,
        cache_path=cache_path,
    )


class SpotifyInterface:
    """
    Wrapper simplifié autour de Spotipy:
      - Auth PKCE si secret absent (ou forcé).
      - Méthodes de recommandations émotionnelles.
    """

    def __init__(
        self,
        scope: Optional[str] = None,
        market: str = "FR",
        cache_path: Optional[str] = None,
        requests_timeout: int = 10,
        retries: int = 5,
        backoff_factor: float = 0.5,
        show_dialog: bool = True,
        force_pkce: bool = False,
    ) -> None:
        self.market = (market or "FR").upper()
        scope = scope or DEFAULT_SCOPE
        self._cache_path = cache_path or ".cache-spotipy"
        if force_pkce:
            os.environ["SPOTIPY_PKCE_ONLY"] = "1"
        self._auth_manager = _build_auth_manager(scope, self._cache_path, show_dialog)
        self.sp = spotipy.Spotify(
            auth_manager=self._auth_manager,
            requests_timeout=requests_timeout,
            retries=retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
        )
        self._required_scope = set((scope or DEFAULT_SCOPE).split())
        # Vérification post-auth
        try:
            tok = (
                self._auth_manager.validate_token(
                    self._auth_manager.cache_handler.get_cached_token()
                )
                or {}
            )
            self._granted_scopes = set(tok.get("scope", "").split())
        except Exception:
            self._granted_scopes = set()
        if not self._required_scope.issubset(self._granted_scopes):
            # Était en warning -> passe en debug pour éviter le spam terminal; l'UI gère l'affichage.
            logger.debug(
                "Scopes manquants (%s). Ré-auth potentiellement nécessaire.",
                " ".join(self._required_scope - self._granted_scopes),
            )

    # ---- Gestion scopes / cache ----
    def clear_cache(self):
        """Supprime le fichier cache pour forcer ré-auth complète."""
        try:
            if os.path.exists(self._cache_path):
                os.remove(self._cache_path)
                logger.info("Cache Spotipy supprimé: %s", self._cache_path)
        except Exception as e:
            logger.warning("Suppression cache impossible: %s", e)

    def ensure_scopes(self, needed: List[str]) -> bool:
        needed_set = set(needed)
        if not hasattr(self, "_granted_scopes"):
            return False
        ok = needed_set.issubset(self._granted_scopes)
        if not ok:
            logger.warning(
                "Scopes requis non accordés: %s (granted=%s)",
                needed_set - self._granted_scopes,
                self._granted_scopes,
            )
        return ok

    def missing_scopes(self, needed: List[str]) -> List[str]:
        """Retourne la liste des scopes manquants (sans log)."""
        needed_set = set(needed)
        return list(needed_set - getattr(self, "_granted_scopes", set()))

    def has_token(self) -> bool:
        """True si un token existe en cache (utile pour savoir si ré-auth nécessaire)."""
        try:
            return bool(self._auth_manager.cache_handler.get_cached_token())
        except Exception:
            return False

    def authorize_url(self) -> str:
        """Construit l’URL d’autorisation à ouvrir dans le navigateur (auth manuelle)."""
        try:
            return self._auth_manager.get_authorize_url()
        except Exception as e:
            logger.warning("Authorize URL erreur: %s", e)
            return ""

    def complete_auth_from_redirect(self, redirected_url: str) -> bool:
        """
        Finalise l’autorisation à partir de l’URL complète (copiée après consentement).
        Met à jour les scopes accordés et réinitialise le client.
        """
        try:
            code = self._auth_manager.parse_response_code(redirected_url)
            if not code:
                from urllib.parse import urlparse, parse_qs

                q = parse_qs(urlparse(redirected_url).query)
                code = (q.get("code") or [None])[0]
            if not code:
                return False
            # Spotipy >= 2.19: get_access_token retourne un dict token_info
            token_info = self._auth_manager.get_access_token(code, check_cache=False)
            if not token_info or not token_info.get("access_token"):
                return False
            self._granted_scopes = set((token_info.get("scope") or "").split())
            # Reconstruit le client avec le même auth_manager (token déjà en cache)
            self.sp = spotipy.Spotify(auth_manager=self._auth_manager)
            return True
        except Exception as e:
            logger.warning("Complétion auth échouée: %s", e)
            return False

    def current_redirect_uri(self) -> str:
        """Retourne le redirect_uri configuré par l'auth manager (utile pour debug UI)."""
        try:
            return getattr(self._auth_manager, "redirect_uri", "") or ""
        except Exception:
            return ""

    # ---- Utilitaires profil ----
    def set_market(self, market: str) -> None:
        self.market = market.upper()

    # ---- Audio features simples (sans blocage global) ----
    def get_track_valence_energy(self, track_id: str) -> Dict[str, Optional[float]]:
        """
        Retourne la valence et l'énergie d'un morceau Spotify.
        Exemple: {"valence": 0.84, "energy": 0.63}.
        Renvoie None pour chaque champ en cas d'erreur.
        """
        if not track_id:
            return {"valence": None, "energy": None}

        try:
            features_list = self.sp.audio_features([track_id])
            if not features_list or features_list[0] is None:
                logger.warning(
                    "Aucune audio_feature trouvée pour le track_id=%s", track_id
                )
                return {"valence": None, "energy": None}

            f = features_list[0]
            return {
                "valence": f.get("valence"),
                "energy": f.get("energy"),
            }
        except SpotifyException as e:
            logger.warning("Erreur Spotify audio_features pour %s: %s", track_id, e)
            return {"valence": None, "energy": None}
        except Exception as e:
            logger.warning("Erreur inconnue audio_features pour %s: %s", track_id, e)
            return {"valence": None, "energy": None}
