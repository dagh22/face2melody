# Face2Melody V3 — Compagnon Émotionnel par IA Affective

> Projet de recherche M.Sc. Intelligence Artificielle — UQAM  
> Système multimodal de détection émotionnelle et recommandation musicale en temps réel

![Python](https://img.shields.io/badge/Python-3.9--3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![Claude API](https://img.shields.io/badge/Claude-Sonnet%204.6-orange)
![Last.fm](https://img.shields.io/badge/Last.fm-API-darkred)
![License](https://img.shields.io/badge/License-Academic-green)

---

## 🆕 Nouveautés V3 — Fusion Trimode & Agnosticisme Plateforme

| Fonctionnalité | V1/V2 | V3 |
|---|---|---|
| Modalités d'entrée | Vision + Texte | **Vision + Texte + Physiologie (BPM)** |
| Agent cognitif | Heuristiques pondérées | **Claude Sonnet 4.6 (LLM Late Fusion)** |
| Plateformes musicales | Spotify uniquement | **Spotify + Last.fm + YouTube + Apple Music** |
| Explicabilité (XAI) | Label émotion | **Raisonnement technique + message empathique** |
| Détection d'anomalies | Non | **Anxiété latente, sarcasme, masquage émotionnel** |
| Qualité caméra | Non prise en compte | **Ajustement dynamique des poids selon luminosité** |

---

## Vue d'ensemble

**Face2Melody** est passé d'un outil de recommandation musicale à un **Compagnon Émotionnel**. Il détecte l'état affectif de l'utilisateur via trois modalités complémentaires et génère des recommandations musicales adaptées sur plusieurs plateformes, avec une justification XAI empathique en français.

### Question de recherche

> *La fusion trimodale (vision + texte + physiologie) améliore-t-elle la pertinence perçue des recommandations musicales par rapport aux approches bimodales ou heuristiques ?*

---

## Architecture V3 — Fusion Trimode (Late Fusion)

```
┌──────────────────────────────────────────────────────────────────┐
│                     Streamlit Chat Interface                      │
│                    (st.chat_message — app.py)                    │
└──────────┬───────────────────┬──────────────────┬────────────────┘
           │                   │                  │
   ┌───────▼──────┐   ┌────────▼───────┐  ┌──────▼───────┐
   │  Vision      │   │  Texte (NLP)   │  │  Physiologie  │
   │  VGG-Face    │   │  Analyse       │  │  BPM (réel   │
   │  DeepFace    │   │  sémantique    │  │  ou simulé)  │
   │  7 → 4 cls   │   │  & tonale FR   │  └──────┬───────┘
   └───────┬──────┘   └────────┬───────┘         │
           │                   │                  │
           └───────────────────┴──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  EmotionFusionAgent  │
                    │  (agent_logic.py)   │
                    │  Claude Sonnet 4.6  │
                    │                     │
                    │  Règles d'arbitrage:│
                    │  BPM>100 + neutre   │
                    │  → anxiété latente  │
                    │  Faible lumière     │
                    │  → texte 70%,       │
                    │    BPM 30%          │
                    └──────────┬──────────┘
                               │ JSON XAI structuré
                               ▼
          ┌────────────────────────────────────────┐
          │           Multi-Platform Discovery      │
          ├─────────────┬───────────┬──────────────┤
          │  Spotify    │  Last.fm  │  YouTube +   │
          │  (Premium)  │  (Gratuit)│  Apple Music │
          └─────────────┴───────────┴──────────────┘
```

---

## Protocole de Fusion Trimode

L'agent LLM arbitre entre trois signaux pour générer l'état affectif unifié :

1. **Vision (VGG-Face)** — Probabilités brutes des expressions faciales (7 classes → 4)
2. **Texte (NLP)** — Analyse sémantique et tonale du chat en français
3. **Physiologie (BPM)** — Rythme cardiaque réel ou simulé

**Règles d'arbitrage :**
- Si BPM > 100 mais visage neutre → suspecter anxiété latente ou stress physique
- Si luminosité caméra faible → poids Texte = 70%, BPM = 30%, Vision = 0%
- Si divergence texte/visage > seuil → détecter sarcasme ou masquage émotionnel
- Fallback V1 heuristique si l'API Claude est indisponible

**Format de sortie JSON de l'agent :**
```json
{
  "emotion_unifiee": "anxiété",
  "bpm_analysis": "BPM élevé (115) contredisant expression neutre",
  "analyse_cognitive_interne": "Signal physiologique dominant — stress somatique probable",
  "message_utilisateur": "Votre rythme cardiaque suggère un besoin de calme...",
  "confidence_score": 0.82,
  "music_params": {
    "target_valence": 0.3,
    "target_energy": 0.2,
    "suggested_artists": ["Nils Frahm", "Ólafur Arnalds"],
    "suggested_genres": ["ambient", "piano", "classical"]
  }
}
```

---

## Classes Émotionnelles & Paramètres Musicaux

| Émotion  | Valence | Énergie | Tempo  | Genres suggérés        |
|----------|---------|---------|--------|------------------------|
| Heureux  | Élevée  | Élevée  | Rapide | pop, dance, funk       |
| Triste   | Faible  | Faible  | Lent   | acoustic, piano, indie |
| Anxieux  | Faible  | Moyenne | Varié  | ambient, classical     |
| En colère| Faible  | Élevée  | Rapide | metal, rock, hip-hop   |
| Neutre   | Moyenne | Moyenne | Moyen  | ambient, chill, lo-fi  |

---

## Intégrations Musicales

### Spotify (Premium requis)
- Recommandations par artistes seeds, genres, et paramètres audio
- 5+ stratégies de fallback en cascade

### Last.fm (Gratuit — nouveau en V3)
- Alternative sans quota Spotify Premium
- Mapping émotion → tags Last.fm
- Recommandations personnalisées via historique d'écoute

### YouTube (optionnel — nouveau en V3)
- YouTube Data API v3
- Vidéos intégrables avec thumbnails
- Fallback gracieux si quota épuisé

### Apple Music via iTunes (nouveau en V3)
- URLs d'embed Apple Music sans authentification
- Enrichissement des pistes Last.fm

---

## Stack Technique

| Couche | Technologie |
|---|---|
| UI | Streamlit ≥ 1.35 (mode chat `st.chat_message`) |
| Agent LLM | Claude Sonnet 4.6 (Anthropic API) |
| Détection faciale | DeepFace 0.0.93 + VGG-Face |
| Landmarks | MediaPipe FaceMesh |
| Vision | OpenCV ≥ 4.8 |
| Deep Learning | TensorFlow 2.15 (tensorflow-macos sur Apple Silicon) |
| NLP | HuggingFace Transformers ≥ 4.44 + PyTorch CPU |
| Musique | Spotipy ≥ 2.19 · pylast ≥ 5.1 · YouTube Data API v3 · iTunes API |
| Data | pandas, numpy < 2.0, Altair |
| Config | python-dotenv |

---

## Installation

### Prérequis
- Python 3.9–3.11
- Webcam (pour la détection faciale)
- Compte [Spotify Developer](https://developer.spotify.com/dashboard) (optionnel)
- Clé API [Last.fm](https://www.last.fm/api/account/create) (gratuite, recommandée)
- Clé API [YouTube Data v3](https://console.cloud.google.com/) (optionnelle)

### 1. Cloner le dépôt
```bash
git clone https://github.com/<your-username>/face2melody.git
cd face2melody
```

### 2. Environnement virtuel
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Installer les dépendances

**Standard (Intel/AMD) :**
```bash
pip install --upgrade pip
pip install tensorflow==2.15.1
pip install -r requirements.txt
```

**Apple Silicon (M1/M2/M3) :**
```bash
pip install --upgrade pip
pip install tensorflow-macos==2.15.0 tensorflow-metal==1.1.0
pip install -r requirements.txt
```

### 4. Variables d'environnement

Créer un fichier `.env` à la racine :
```env
# Agent LLM (requis pour la fusion trimode V3)
ANTHROPIC_API_KEY=your_anthropic_api_key

# Spotify (optionnel — recommandations Premium)
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8501/callback

# Last.fm (recommandé — gratuit)
LASTFM_API_KEY=your_lastfm_api_key
LASTFM_API_SECRET=your_lastfm_api_secret

# YouTube (optionnel)
YOUTUBE_API_KEY=your_youtube_api_key

# Paramètres app
PARTICIPANT_ID=YOUR_NAME
F2M_ENABLE_DEEPFACE=1
```

### 5. Lancer l'application
```bash
streamlit run app.py
```

---

## Structure du Projet

```
face2melody/
├── app.py                          # Application Streamlit principale (V2 + V3)
├── app.v3.py                       # Prototype standalone V3 (Trimode)
├── agent_logic.py                  # EmotionFusionAgent (Claude Sonnet 4.6)
├── experiment_utils.py             # Moteur de fusion + logging
├── feedback_learning.py            # Reclassement par feedback utilisateur
├── analytics_gen.py                # Dashboard analytique recherche
├── test_v1_vs_v2.py                # Suite de validation V1 vs V2 vs V3
├── requirements.txt
├── recommender/
│   ├── emotion_detector.py         # Wrapper DeepFace (lazy init, thread-safe)
│   ├── spotify_interface.py        # Client Spotify OAuth + recommandations
│   ├── lastfm_interface.py         # Client Last.fm (nouveau V3)
│   ├── youtube_search.py           # Recherche YouTube Data API (nouveau V3)
│   ├── itunes_search.py            # Embed Apple Music via iTunes (nouveau V3)
│   └── playlist_generator.py      # Génération de playlists en lot
├── archive/                        # Pages Streamlit dépréciées (référence)
└── logs/                           # Logs JSONL participants (gitignored)
```

---

## Suite de Tests

`test_v1_vs_v2.py` contient 5 cas de test couvrant :
- Congruence émotionnelle (signaux alignés)
- Dissonance (signaux contradictoires)
- Détection de sarcasme
- Signaux faibles / ambigus
- Neutralité

Génère un rapport de comparaison dans `logs/v1_vs_v2_comparison.json`.

---

## Confidentialité & Données

Les logs de participants sont stockés localement dans `logs/` et exclus du contrôle de version. Aucune donnée personnelle n'est transmise à un tiers autre que les APIs configurées. L'analyse faciale s'exécute entièrement en local via DeepFace — aucune image n'est stockée ou envoyée à distance.

---

## Auteur

**Daghsen** — Projet de Recherche M.Sc. Intelligence Artificielle  
Informatique Cognitive / Affective Computing — UQAM

---

## Licence

Ce projet est publié à des fins académiques et de recherche. Voir [LICENSE](LICENSE) pour les détails.
