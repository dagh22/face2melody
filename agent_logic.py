"""
agent_logic.py — Face2Melody V3 : Agent Cognitif de Fusion Trimode (Chapitre 6)

Module central de l'architecture agentique V3. Remplace la fusion bimodale V2
par une fusion trimodale (Vision + Texte + Physiologie/BPM) :
  - Signal facial : distribution VGG-Face (DeepFace)
  - Signal textuel : analyse sémantique NLP du chat utilisateur
  - Signal physiologique : rythme cardiaque BPM (réel ou simulé)

Règles d'arbitrage V3 :
  - BPM > 100 + visage neutre → suspecter anxiété latente ou stress physique
  - Luminosité caméra faible → poids texte=70%, BPM=30%, visage=0%
  - Dissonance multimodale → arbitrage XAI via LLM (Claude Sonnet 4.6)
  - Fallback V1 heuristique si API LLM indisponible

Point d'entrée principal : process_multimodal_emotions(face_probs, user_text, user_bpm)
Compatibilité ascendante : AgentResult conserve tous les champs utilisés par app.py
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes globales
# ──────────────────────────────────────────────────────────────────────────────

EMOS = ["happy", "sad", "angry", "neutral"]

# Ancres (valence, energy) par émotion — alignées avec experiment_utils.TARGETS
_ANCHOR: Dict[str, Tuple[float, float]] = {
    "happy":   (0.85, 0.75),
    "sad":     (0.20, 0.35),
    "angry":   (0.30, 0.90),
    "neutral": (0.55, 0.50),
}

# Genres musicaux par émotion — max 3, conformes SAFE_GENRES Spotify
_GENRE_MAP: Dict[str, List[str]] = {
    "happy":   ["pop", "dance", "indie"],
    "sad":     ["classical", "acoustic", "lo-fi"],
    "angry":   ["rock", "hip-hop", "metal"],
    "neutral": ["jazz", "lo-fi", "indie"],
}

# Artistes suggérés par émotion (platform-agnostic : Spotify + YouTube)
_ARTIST_MAP: Dict[str, List[str]] = {
    "happy":   ["Pharrell Williams", "Dua Lipa", "Bruno Mars"],
    "sad":     ["Adele", "Bon Iver", "Sufjan Stevens"],
    "angry":   ["Kendrick Lamar", "Rage Against the Machine", "Eminem"],
    "neutral": ["Norah Jones", "Bonobo", "Tycho"],
}

# Mapping camera_status → (w_face, w_text, w_bpm)
# CLAUDE.md V3 : luminosité faible = 0% face, 70% texte, 30% BPM
_CAMERA_WEIGHT: Dict[str, Tuple[float, float, float]] = {
    "ok":          (0.60, 0.30, 0.10),
    "low_light":   (0.00, 0.70, 0.30),  # V3 : caméra sombre → visage ignoré
    "obstructed":  (0.00, 0.80, 0.20),  # caméra bouchée → texte dominant
    "unavailable": (0.00, 0.70, 0.30),  # aucune caméra → idem low_light
}

# Alias rétrocompat pour app.py qui attend _CAMERA_WEIGHT[status] → float
_CAMERA_WEIGHT_FACE: Dict[str, float] = {k: v[0] for k, v in _CAMERA_WEIGHT.items()}

# Lexique bilingue FR/EN pour la conversion texte → distribution d'émotions
_LEX_SAD     = ("triste", "malheureux", "malheur", "déprim", "deprim", "chagrin",
                "pleurer", "pleure", "mélancolie", "mélancolique", "épuisé", "épuisée")
_LEX_ANGRY   = ("nerveux", "nerveuse", "stressé", "stressée", "stresse", "stressee",
                "anxieux", "anxieuse", "angoissé", "angoissee", "fâché", "fâchée",
                "énervé", "énervée", "furieux", "frustr")
_LEX_HAPPY   = ("heureux", "heureuse", "content", "contente", "joie", "ravi", "ravie",
                "enthousiaste", "excité", "excitée", "génial", "super", "bien")
_LEX_NEUTRAL = ("neutre", "ok", "normal", "calme", "tranquille", "bof", "moyen")


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaire BPM — signal physiologique
# ──────────────────────────────────────────────────────────────────────────────

def _bpm_to_analysis(bpm: Optional[int]) -> Tuple[str, str]:
    """
    Convertit un BPM en (emotion_hint, bpm_analysis_text).

    Règles d'arbitrage CLAUDE.md V3 :
    - BPM > 120 : stress aigu ou excitation intense → hint 'angry'
    - BPM 100-120 : anxiété latente ou effort physique → hint 'anxious' (arbitré par LLM)
    - BPM 80-100 : état équilibré → hint 'happy'
    - BPM 60-80 : repos → hint 'neutral'
    - BPM < 60 : fatigue ou mélancolie → hint 'sad'

    Args:
        bpm: Rythme cardiaque en battements par minute (None si indisponible).

    Returns:
        Tuple (hint émotion, texte d'analyse) pour injection dans le prompt LLM.
    """
    if bpm is None:
        return "unknown", "Aucune donnée physiologique disponible."

    if bpm > 120:
        hint = "angry"
        txt = (
            f"BPM={bpm} (très élevé) : activation physiologique intense suggérant "
            f"un stress aigu ou une excitation forte."
        )
    elif bpm > 100:
        hint = "anxious"  # le LLM arbitre entre 'angry' et 'neutral'
        txt = (
            f"BPM={bpm} (modéré-élevé) : signal d'anxiété latente ou d'effort physique. "
            f"Si le visage est neutre, suspecter un stress dissimulé."
        )
    elif bpm >= 80:
        hint = "happy"
        txt = f"BPM={bpm} (normal-actif) : état physiologique équilibré et énergique."
    elif bpm >= 60:
        hint = "neutral"
        txt = f"BPM={bpm} (repos) : calme physiologique, état de détente."
    else:
        hint = "sad"
        txt = (
            f"BPM={bpm} (bas) : possibilité de fatigue physique ou de mélancolie. "
            f"Vérifier la cohérence avec les signaux facial et textuel."
        )

    return hint, txt


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaire texte — conversion texte → distribution d'émotions
# ──────────────────────────────────────────────────────────────────────────────

def _text_to_probs(text: str) -> Dict[str, float]:
    """
    Convertit un texte brut en distribution de probabilités émotionnelles.
    Matching lexical bilingue FR/EN + normalisation.
    Distribution neutre uniforme si aucun mot-clé reconnu.
    """
    if not text or not text.strip():
        return {k: 0.25 for k in EMOS}

    t = text.lower()
    scores: Dict[str, float] = {k: 0.0 for k in EMOS}

    for kw in _LEX_SAD:
        if kw in t:
            scores["sad"] += 1.0
    for kw in _LEX_ANGRY:
        if kw in t:
            scores["angry"] += 1.0
    for kw in _LEX_HAPPY:
        if kw in t:
            scores["happy"] += 1.0
    for kw in _LEX_NEUTRAL:
        if kw in t:
            scores["neutral"] += 1.0

    total = sum(scores.values())
    if total < 0.01:
        return {"happy": 0.15, "sad": 0.15, "angry": 0.15, "neutral": 0.55}

    return {k: v / total for k, v in scores.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass de résultat — schéma unifié V3 avec rétrocompatibilité V2
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """
    Résultat structuré de l'agent cognitif V3.

    Champs de compatibilité ascendante (utilisés par app.py V2) :
        valence, arousal, dominant_emotion, fused_distribution, reasoning, from_fallback

    Nouveaux champs CLAUDE.md V3 :
        emotion_unifiee, bpm_analysis, analyse_cognitive_interne,
        message_utilisateur, confidence_score, suggested_genres, suggested_artists

    Propriétés alias V2→V3 (pour éviter de casser les accès existants dans app.py) :
        analyse_cognitive → analyse_cognitive_interne
        justification_empathique → message_utilisateur
        seed_genres → suggested_genres
    """
    # ── Champs conservés — utilisés directement par app.py ────────────────
    valence: float                          # Cible musicale : valence émotionnelle [0.0, 1.0]
    arousal: float                          # Cible musicale : énergie / activation [0.0, 1.0]
    dominant_emotion: str                   # Émotion principale parmi EMOS
    fused_distribution: Dict[str, float]    # Distribution trimodale fusionnée normalisée
    reasoning: str                          # Message affiché dans l'UI (= message_utilisateur)
    from_fallback: bool = False             # True si le LLM était indisponible

    # ── Champs V3 CLAUDE.md ───────────────────────────────────────────────
    emotion_unifiee: str = ""               # Alias de dominant_emotion
    bpm_analysis: str = ""                  # Interprétation du signal BPM (NOUVEAU V3)
    analyse_cognitive_interne: str = ""     # Raisonnement XAI technique (pour le rapport)
    message_utilisateur: str = ""           # Justification empathique en français pour l'UI
    confidence_score: float = 0.0           # Confiance de l'agent [0.0, 1.0]
    suggested_genres: List[str] = field(default_factory=list)   # Genres musicaux (ex-seed_genres)
    suggested_artists: List[str] = field(default_factory=list)  # Artistes (NOUVEAU V3, pour YouTube)

    # ── Propriétés alias V2 → V3 (rétrocompatibilité) ────────────────────

    @property
    def analyse_cognitive(self) -> str:
        """Alias V2 : pointe vers analyse_cognitive_interne."""
        return self.analyse_cognitive_interne

    @property
    def justification_empathique(self) -> str:
        """Alias V2 : pointe vers message_utilisateur."""
        return self.message_utilisateur

    @property
    def seed_genres(self) -> List[str]:
        """Alias V2 : pointe vers suggested_genres."""
        return self.suggested_genres

    def to_dict(self) -> Dict:
        """
        Sérialise au format JSON strict défini dans CLAUDE.md V3.
        Retourné par process_multimodal_emotions().
        """
        return {
            "emotion_unifiee": self.emotion_unifiee,
            "bpm_analysis": self.bpm_analysis,
            "analyse_cognitive_interne": self.analyse_cognitive_interne,
            "message_utilisateur": self.message_utilisateur,
            "confidence_score": self.confidence_score,
            "music_params": {
                "target_valence": self.valence,
                "target_energy": self.arousal,
                "suggested_artists": self.suggested_artists,
                "suggested_genres": self.suggested_genres,
            },
            "from_fallback": self.from_fallback,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Agent cognitif principal — Fusion Trimode (V3)
# ──────────────────────────────────────────────────────────────────────────────

class EmotionFusionAgent:
    """
    Agent cognitif LLM (Claude Sonnet 4.6) pour la fusion trimodale.

    Orchestre trois signaux :
    1. Vision (VGG-Face / DeepFace) — distribution faciale
    2. Texte (NLP) — analyse sémantique du chat
    3. Physiologie (BPM) — rythme cardiaque réel ou simulé

    Applique les règles d'arbitrage CLAUDE.md V3 et produit un JSON aligné
    sur le schéma V3. Retombe sur la fusion heuristique V1 si l'API échoue.
    """

    _SYSTEM = """\
Tu es un expert en Informatique Affective (Affective Computing) pour Face2Melody V3.
Ton rôle : analyser trois signaux émotionnels (visage + texte + BPM) et produire
une interprétation nuancée pour guider une recommandation musicale personnalisée.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT avec un objet JSON valide. Aucun texte avant ou après.
2. Respecte EXACTEMENT ce schéma (clés identiques) :
   {
     "emotion_unifiee": "<happy|sad|angry|neutral>",
     "bpm_analysis": "<interprétation du signal physiologique BPM : 1-2 phrases>",
     "analyse_cognitive_interne": "<raisonnement XAI technique : 2-3 phrases>",
     "message_utilisateur": "<justification empathique en français : 2-3 phrases>",
     "confidence_score": <float entre 0.0 et 1.0>,
     "music_params": {
       "target_valence": <float entre 0.0 et 1.0>,
       "target_energy":  <float entre 0.0 et 1.0>,
       "suggested_artists": ["<artiste1>", "<artiste2>", "<artiste3>"],
       "suggested_genres":  ["<genre1>", "<genre2>", "<genre3>"]
     }
   }

RÈGLES BPM (signal physiologique) :
- BPM > 100 + visage NEUTRE → suspecter anxiété latente ou stress dissimulé.
  Signale-le dans bpm_analysis et ajuste emotion_unifiee en conséquence.
- BPM > 120 → activation intense (stress aigu ou excitation) → envisager 'angry'.
- BPM 60-80 → repos physiologique → cohérent avec 'neutral' ou 'sad'.
- Si BPM indisponible → bpm_analysis = "Aucune donnée physiologique disponible."

RÈGLES CAMÉRA :
- Si w_face=0.0 (low_light/obstructed) → ignore le signal facial, base-toi sur texte + BPM.
- Si w_face élevé → le signal facial est fiable et prioritaire.

ANCRES ÉMOTIONNELLES (valence, energy) pour target_valence/target_energy :
  happy → (0.85, 0.75) | sad → (0.20, 0.35) | angry → (0.30, 0.90) | neutral → (0.55, 0.50)

ARTISTES SUGGÉRÉS PAR DÉFAUT (tu peux en proposer d'autres adaptés au contexte) :
  happy → Pharrell Williams, Dua Lipa, Bruno Mars
  sad → Adele, Bon Iver, Sufjan Stevens
  angry → Kendrick Lamar, Rage Against the Machine, Eminem
  neutral → Norah Jones, Bonobo, Tycho

GENRES SUGGÉRÉS (choisis parmi : pop, dance, indie, classical, acoustic, lo-fi,
  rock, hip-hop, metal, jazz, r-n-b, soul) — maximum 3.

GESTION DE LA DISSONANCE COGNITIQUE :
- Visage ≠ texte → analyse si c'est sarcasme, masquage émotionnel ou ambivalence.
- BPM contradictoire (ex: BPM élevé mais texte calme) → signaler dans bpm_analysis.
- Détaille l'arbitrage dans analyse_cognitive_interne (pour le rapport XAI).
- message_utilisateur : empathique, bienveillant, sans jargon technique, en français.
"""

    def __init__(self, model: str = "claude-sonnet-4-6", timeout: float = 15.0):
        self.model = model
        self.timeout = timeout
        self._client = None  # Initialisation lazy pour éviter l'import au démarrage

    def _get_client(self):
        """Initialise le client Anthropic à la première utilisation (lazy init)."""
        if self._client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "SDK anthropic manquant. Installe-le : pip install anthropic>=0.30.0"
                ) from exc
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY absent de l'environnement. "
                    "Définis-le dans le fichier .env à la racine du projet."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    # ── Point d'entrée public — rétrocompatible avec experiment_utils.fuse() ──

    def analyze(
        self,
        face_probs: Dict[str, float],
        text_probs: Dict[str, float],
        text_raw: Optional[str] = None,
        w_face: float = 0.60,
        user_bpm: Optional[int] = None,
    ) -> AgentResult:
        """
        Analyse trimodale via LLM. Retombe sur la fusion V1 si l'API échoue.

        Interface rétrocompatible utilisée par experiment_utils.fuse().
        Pour la nouvelle API V3, utilise process_multimodal_emotions() à la place.

        Args:
            face_probs:  Distribution faciale DeepFace/VGG-Face (4 classes normalisées).
            text_probs:  Distribution textuelle NLP (4 classes normalisées).
            text_raw:    Texte brut de l'utilisateur — enrichit le prompt LLM.
            w_face:      Poids suggéré pour le signal facial [0.0, 1.0].
            user_bpm:    Rythme cardiaque en BPM (None si indisponible).

        Returns:
            AgentResult avec tous les champs V2 + nouveaux champs V3.
        """
        try:
            return self._call_llm(face_probs, text_probs, text_raw, w_face, user_bpm)
        except Exception as exc:
            logger.warning("Appel LLM agent échoué (%s) — activation du fallback V1.", exc)
            return self._fallback(face_probs, text_probs, w_face, user_bpm, reason=str(exc))

    # ── Appel LLM et parsing du schéma CLAUDE.md V3 ──────────────────────────

    def _call_llm(
        self,
        face_probs: Dict[str, float],
        text_probs: Dict[str, float],
        text_raw: Optional[str],
        w_face: float,
        user_bpm: Optional[int],
    ) -> AgentResult:
        """
        Envoie les données trimodales au LLM et parse la réponse JSON V3.
        """
        client = self._get_client()

        # Calcul des poids texte et BPM à partir de w_face
        # On suppose une répartition linéaire : w_text = (1 - w_face) * 0.75, w_bpm = reste
        w_text = round((1.0 - w_face) * 0.75, 2)
        w_bpm  = round(1.0 - w_face - w_text, 2)

        bpm_hint, bpm_txt = _bpm_to_analysis(user_bpm)
        user_msg = self._build_user_message(
            face_probs, text_probs, text_raw, w_face, w_text, w_bpm, user_bpm, bpm_hint, bpm_txt
        )

        # Appel API Anthropic — température basse pour une sortie JSON déterministe
        response = client.messages.create(
            model=self.model,
            max_tokens=700,
            temperature=0.2,
            system=self._SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text.strip()
        data = self._parse_llm_response(raw)

        # ── Extraction et validation des champs V3 ────────────────────────
        emotion = str(data.get("emotion_unifiee", "neutral")).lower()
        if emotion not in EMOS:
            emotion = "neutral"

        mp = data.get("music_params", {})
        valence  = float(max(0.0, min(1.0, mp.get("target_valence", _ANCHOR[emotion][0]))))
        energy   = float(max(0.0, min(1.0, mp.get("target_energy",  _ANCHOR[emotion][1]))))
        genres   = [str(g) for g in mp.get("suggested_genres",  _GENRE_MAP[emotion])][:3]
        artists  = [str(a) for a in mp.get("suggested_artists", _ARTIST_MAP[emotion])][:3]
        if not genres:
            genres = _GENRE_MAP[emotion]
        if not artists:
            artists = _ARTIST_MAP[emotion]

        confidence    = float(max(0.0, min(1.0, data.get("confidence_score", 0.5))))
        bpm_analysis  = str(data.get("bpm_analysis", bpm_txt))
        analyse_cog   = str(data.get("analyse_cognitive_interne", ""))
        msg_user      = str(data.get("message_utilisateur", ""))

        # Reconstruction de fused_distribution (pour compatibilité app.py)
        fp = self._norm(face_probs)
        tp = self._norm(text_probs)
        fd = {k: w_face * fp[k] + (1.0 - w_face) * tp[k] for k in EMOS}
        s  = sum(fd.values()) or 1.0
        fd = {k: v / s for k, v in fd.items()}
        # Force l'émotion LLM comme dominante dans la distribution
        if emotion in fd:
            fd[emotion] = max(fd[emotion], 0.5)
            s = sum(fd.values())
            fd = {k: v / s for k, v in fd.items()}

        return AgentResult(
            # Champs rétrocompatibles
            valence=valence,
            arousal=energy,
            dominant_emotion=emotion,
            fused_distribution=fd,
            reasoning=msg_user,         # L'UI Streamlit affiche reasoning
            from_fallback=False,
            # Champs V3
            emotion_unifiee=emotion,
            bpm_analysis=bpm_analysis,
            analyse_cognitive_interne=analyse_cog,
            message_utilisateur=msg_user,
            confidence_score=confidence,
            suggested_genres=genres,
            suggested_artists=artists,
        )

    def _parse_llm_response(self, raw: str) -> Dict:
        """
        Nettoie et parse la réponse JSON du LLM.
        Supprime les blocs markdown et valide les champs obligatoires V3.
        """
        # Supprime les délimiteurs markdown ```json ... ``` si présents
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()

        data = json.loads(clean)

        # Validation minimale : champs obligatoires V3
        required = {
            "emotion_unifiee", "bpm_analysis", "analyse_cognitive_interne",
            "message_utilisateur", "confidence_score", "music_params"
        }
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Champs JSON manquants dans la réponse LLM V3 : {missing}")

        return data

    # ── Construction du message utilisateur envoyé au LLM ────────────────────

    def _build_user_message(
        self,
        face_probs: Dict[str, float],
        text_probs: Dict[str, float],
        text_raw: Optional[str],
        w_face: float,
        w_text: float,
        w_bpm: float,
        bpm: Optional[int],
        bpm_hint: str,
        bpm_txt: str,
    ) -> str:
        """
        Construit le message utilisateur décrivant les trois signaux trimodaux.
        Inclut les avertissements de dissonance et les règles BPM si nécessaire.
        """
        fp = self._norm(face_probs)
        tp = self._norm(text_probs)
        face_dom = max(fp, key=fp.get)
        text_dom = max(tp, key=tp.get)
        dissonance_bimodale = face_dom != text_dom

        lines = [
            "## Données trimodales de l'utilisateur (Face2Melody V3)",
            "",
            f"**Signal facial** (DeepFace / VGG-Face, poids w_face={w_face:.2f}) :",
            (
                f"  happy={fp['happy']:.3f} | sad={fp['sad']:.3f} | "
                f"angry={fp['angry']:.3f} | neutral={fp['neutral']:.3f}"
            ),
            f"  → Émotion dominante visage : **{face_dom}** ({fp[face_dom]:.1%})",
            "",
            f"**Signal textuel** (NLP, poids w_text={w_text:.2f}) :",
            (
                f"  happy={tp['happy']:.3f} | sad={tp['sad']:.3f} | "
                f"angry={tp['angry']:.3f} | neutral={tp['neutral']:.3f}"
            ),
            f"  → Émotion dominante texte : **{text_dom}** ({tp[text_dom]:.1%})",
        ]

        # Ajout du texte brut — crucial pour la détection du sarcasme
        if text_raw:
            safe = text_raw.replace('"', "'")[:300]
            lines += ["", f'**Texte brut** : "{safe}"']

        # Signal BPM — toujours présent (même si "N/A")
        lines += [
            "",
            f"**Signal physiologique** (BPM, poids w_bpm={w_bpm:.2f}) :",
            f"  BPM mesuré : {bpm if bpm is not None else 'N/A'}",
            f"  Hint émotion BPM : {bpm_hint}",
            f"  → {bpm_txt}",
        ]

        # Règle critique V3 : BPM > 100 + visage neutre = anxiété latente
        if bpm is not None and bpm > 100 and face_dom == "neutral":
            lines += [
                "",
                "⚠️  RÈGLE BPM V3 ACTIVÉE : BPM > 100 avec visage NEUTRE.",
                "   Suspecter une anxiété latente ou un stress physique dissimulé.",
                "   Prioriser 'angry' ou 'sad' malgré l'apparence calme du visage.",
            ]

        # Dissonance bimodale visage vs texte
        if dissonance_bimodale and w_face > 0.0:
            lines += [
                "",
                f"⚠️  DISSONANCE BIMODALE DÉTECTÉE :",
                f"   Visage → '{face_dom}' | Texte → '{text_dom}'",
                "   Analyse si c'est du sarcasme, un masquage émotionnel ou une ambivalence.",
                "   Justifie l'arbitrage dans 'analyse_cognitive_interne'.",
            ]
        elif w_face == 0.0:
            lines += [
                "",
                "ℹ️  CAMÉRA DÉGRADÉE (w_face=0.0) : signal facial ignoré.",
                "   Base ta décision uniquement sur le texte et le BPM.",
            ]
        elif not dissonance_bimodale:
            lines += [
                "",
                f"✓ COHÉRENCE BIMODALE : visage et texte s'accordent sur '{face_dom}'.",
            ]

        lines += ["", "Produis le JSON V3 en respectant strictement le schéma demandé."]
        return "\n".join(lines)

    # ── Fallback V1 — fusion heuristique si le LLM est indisponible ──────────

    def _fallback(
        self,
        face_probs: Dict[str, float],
        text_probs: Dict[str, float],
        w_face: float,
        user_bpm: Optional[int] = None,
        reason: str = "",
    ) -> AgentResult:
        """
        Fusion heuristique pondérée V1 — utilisée quand le LLM est indisponible.
        Préserve la continuité du service tout en signalant son origine (from_fallback).
        """
        from experiment_utils import _fuse_heuristic  # import lazy → évite la dépendance circulaire

        fd, dominant = _fuse_heuristic(face_probs, text_probs, w_face)
        v, a = _ANCHOR.get(dominant, (0.55, 0.50))
        _, bpm_txt = _bpm_to_analysis(user_bpm)

        short_reason = reason[:100] if reason else "API indisponible"
        analyse_cog = (
            f"Fallback V1 activé ({short_reason}). "
            f"Fusion heuristique pondérée appliquée avec w_face={w_face:.2f}."
        )
        msg = (
            f"Le système de recommandation avancé est temporairement indisponible. "
            f"Nous vous proposons une sélection basée sur votre émotion principale détectée : '{dominant}'."
        )

        return AgentResult(
            # Champs rétrocompatibles
            valence=v,
            arousal=a,
            dominant_emotion=dominant,
            fused_distribution=fd,
            reasoning=msg,
            from_fallback=True,
            # Champs V3
            emotion_unifiee=dominant,
            bpm_analysis=bpm_txt,
            analyse_cognitive_interne=analyse_cog,
            message_utilisateur=msg,
            confidence_score=0.30,
            suggested_genres=_GENRE_MAP.get(dominant, ["pop", "indie", "lo-fi"]),
            suggested_artists=_ARTIST_MAP.get(dominant, []),
        )

    # ── Utilitaires ──────────────────────────────────────────────────────────

    @staticmethod
    def _norm(p: Dict[str, float]) -> Dict[str, float]:
        """Normalise un dict de probabilités sur les 4 classes EMOS (somme = 1.0)."""
        out = {k: float(p.get(k, 0.0)) for k in EMOS}
        total = sum(out.values()) or 1.0
        return {k: v / total for k, v in out.items()}


# ──────────────────────────────────────────────────────────────────────────────
# API publique V3 — point d'entrée principal du module
# ──────────────────────────────────────────────────────────────────────────────

def process_multimodal_emotions(
    face_probs: Dict[str, float],
    user_text: str,
    user_bpm: Optional[int] = None,
    camera_status: str = "ok",
) -> Dict:
    """
    Point d'entrée principal de l'agent V3 — Late Fusion Trimode.

    Orchestre la fusion des signaux visuels (VGG-Face), textuels (NLP) et
    physiologiques (BPM) selon les règles d'arbitrage CLAUDE.md V3.

    Args:
        face_probs:     Distribution de probabilités faciale (DeepFace/VGG-Face).
                        Format : {"happy": float, "sad": float, "angry": float, "neutral": float}
        user_text:      Texte brut saisi par l'utilisateur dans le chat (peut être "").
        user_bpm:       Rythme cardiaque en BPM (None si indisponible ou non mesuré).
        camera_status:  Qualité du flux caméra :
                          "ok"          → caméra fonctionnelle (w_face=0.60)
                          "low_light"   → luminosité faible (w_face=0.00, V3)
                          "obstructed"  → caméra obstruée (w_face=0.00)
                          "unavailable" → pas de caméra (w_face=0.00)

    Returns:
        dict JSON conforme au schéma CLAUDE.md V3 :
        {
            "emotion_unifiee": str,
            "bpm_analysis": str,
            "analyse_cognitive_interne": str,
            "message_utilisateur": str,
            "confidence_score": float,
            "music_params": {
                "target_valence": float,
                "target_energy": float,
                "suggested_artists": List[str],
                "suggested_genres": List[str]
            },
            "from_fallback": bool
        }
    """
    # Étape 1 — Calcul des poids trimodaux selon la qualité de la caméra
    w_face, w_text, w_bpm = _CAMERA_WEIGHT.get(camera_status, _CAMERA_WEIGHT["ok"])
    logger.debug("camera_status='%s' → w_face=%.2f, w_text=%.2f, w_bpm=%.2f",
                 camera_status, w_face, w_text, w_bpm)

    # Étape 2 — Conversion du texte brut en distribution d'émotions
    text_probs = _text_to_probs(user_text)
    logger.debug("text_probs depuis '%s...' : %s", user_text[:30], text_probs)

    # Étape 3 — Fusion trimodale via l'agent LLM
    result = get_agent().analyze(
        face_probs=face_probs,
        text_probs=text_probs,
        text_raw=user_text if user_text.strip() else None,
        w_face=w_face,
        user_bpm=user_bpm,
    )

    # Étape 4 — Sérialisation au format JSON strict CLAUDE.md V3
    return result.to_dict()


# ──────────────────────────────────────────────────────────────────────────────
# Singleton module-level (init lazy, thread-safe pour Streamlit single-thread)
# ──────────────────────────────────────────────────────────────────────────────

_agent: Optional[EmotionFusionAgent] = None


def get_agent() -> EmotionFusionAgent:
    """Retourne le singleton EmotionFusionAgent (créé à la première utilisation)."""
    global _agent
    if _agent is None:
        _agent = EmotionFusionAgent()
    return _agent
