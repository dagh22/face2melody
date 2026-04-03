# Face2Melody — Emotion-Driven Music Recommender

> Master's Research Project — Human-Computer Interaction & Affective Computing  
> A real-time, multimodal emotion detection system that recommends Spotify tracks matched to your emotional state.

---

## Overview

**Face2Melody** is a research prototype that bridges **affective computing** and **music recommendation**. It detects a user's emotional state through two complementary modalities — **facial expression analysis** (computer vision) and **natural language input** (French NLP) — then queries the Spotify API to surface tracks whose acoustic profile matches the detected emotion.

The project was designed as a controlled human study with two experiment scenarios, enabling direct comparison of recommendation quality between an automatic fast-fusion method and a guided conversational method.

---

## Research Question

> *Does incorporating multimodal emotion detection (face + text) improve the perceived relevance of music recommendations compared to unimodal or rule-based approaches?*

The dual-scenario design (S1 vs S2) allows head-to-head comparison of user satisfaction ratings, like/dislike rates, and audio-feature distance-to-target metrics across both methods.

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Frontend                  │
│         (Dark Spotify-themed UI — app.py)            │
└────────────┬────────────────────────┬────────────────┘
             │                        │
     ┌───────▼────────┐      ┌────────▼───────┐
     │  Facial Module │      │   Text Module  │
     │ facial_utils.py│      │ text_utils.py  │
     │  (DeepFace +   │      │ (French NLP,   │
     │   OpenCV +     │      │  rule-based)   │
     │   MediaPipe)   │      └────────┬───────┘
     └───────┬────────┘               │
             │   7-class scores        │  probability vector
             │   → mapped to 4        │
             └──────────┬─────────────┘
                        │
               ┌────────▼────────┐
               │  Fusion Engine  │   face: 55–60%  /  text: 40–45%
               │  experiment_    │   text-override when confidence > 65%
               │  utils.py       │
               └────────┬────────┘
                        │  final emotion label
                        ▼
               ┌────────────────┐
               │  Spotify Reco  │   seed artists/genres from user profile
               │  Engine        │   audio feature targets (valence, energy,
               │  spotify_       │   tempo) per emotion class
               │  interface.py  │
               └────────┬───────┘
                        │  6 tracks
                        ▼
               ┌────────────────┐
               │  Feedback      │   per-URI like/dislike/rating history
               │  Reranker      │   loaded from JSONL logs at session start
               │  feedback_     │
               │  learning.py   │
               └────────┬───────┘
                        │
                        ▼
               Track list + Spotify player embed
               User rates (1–5 stars), likes, listens
                        │
                        ▼
               logs/experiment_<participant_id>.jsonl
```

---

## Emotion Classes & Spotify Targets

| Emotion  | Valence | Energy | Tempo  | Example Genres         |
|----------|---------|--------|--------|------------------------|
| Happy    | High    | High   | Fast   | pop, dance, funk       |
| Sad      | Low     | Low    | Slow   | acoustic, piano, indie |
| Angry    | Low     | High   | Fast   | metal, rock, hip-hop   |
| Neutral  | Medium  | Medium | Medium | ambient, chill, lo-fi  |

---

## Experiment Design

### Scenario S1 — Quick Fusion
1. 10-second webcam capture (OpenCV + DeepFace, EMA-smoothed)
2. Optional French text correction by the user
3. Face (60%) + text (40%) fusion → final emotion
4. 6 Spotify tracks displayed with embedded player
5. Per-track feedback: star rating, like/dislike, listened toggle

### Scenario S2 — Guided Chat
1. Automated 10-second webcam capture
2. 7-step French chatbot eliciting emotional state, energy level, mood, stress, genre preferences, etc.
3. Text aggregated into probability distribution via NLP
4. Audio feature targets (valence/energy) derived from keyword heuristics
5. Face (55%) + text (45%) fusion → final emotion
6. Same 6-track display + feedback flow as S1

Comparing S1 and S2 allows measuring whether the added conversational effort of S2 improves recommendation quality.

---

## Key Technical Features

- **Anti-bias facial detection pipeline**: Custom score reduction mapping 7 DeepFace emotion classes to 4, with temperature sharpening, per-label confidence thresholds, EMA smoothing, frame quality gating (blur + brightness checks), and optional MediaPipe smile-geometry boost to counteract DeepFace's documented over-prediction of "neutral".
- **Multimodal fusion with text override**: When text classifier confidence exceeds 65%, text signal overrides the face result — handling cases where facial expression contradicts stated mood.
- **Cascading Spotify fallbacks**: 5+ fallback strategies (with/without audio features, genre-only, keyword search) ensure tracks are always returned even when the Spotify Recommendations API restricts certain endpoints.
- **Persistent feedback learning**: Past user ratings and likes/dislikes are stored in JSONL logs and used to rerank future recommendations without a database — a lightweight personalization layer.
- **Research logging**: Every interaction is logged as a structured JSONL record (emotion labels, audio features, feedback, seeds, timing) for offline analysis.
- **Apple Silicon support**: Conditional TensorFlow stack (`tensorflow-macos` + `tensorflow-metal`) for M1/M2/M3 Macs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit >= 1.35 (dark Spotify-green theme) |
| Facial Emotion | DeepFace 0.0.93 + RetinaFace |
| Face Landmarks | MediaPipe FaceMesh (smile geometry) |
| Computer Vision | OpenCV >= 4.8 |
| Deep Learning | TensorFlow 2.15 (tensorflow-macos on Apple Silicon) |
| NLP | HuggingFace Transformers >= 4.44 + PyTorch (CPU) |
| Music API | Spotipy >= 2.19 (Spotify Web API, PKCE OAuth) |
| Data | pandas, numpy < 2.0, Altair |
| Config | python-dotenv |

---

## Installation

### Prerequisites
- Python 3.9–3.11
- A [Spotify Developer](https://developer.spotify.com/dashboard) account with a registered app
- A webcam (for facial detection)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/face2melody.git
cd face2melody
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

**Standard (Intel/AMD):**
```bash
pip install --upgrade pip
pip install tensorflow==2.15.1
pip install -r requirements.txt
```

**Apple Silicon (M1/M2/M3):**
```bash
pip install --upgrade pip
pip install tensorflow-macos==2.15.0 tensorflow-metal==1.1.0
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file at the project root:
```env
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8501/callback
PARTICIPANT_ID=YOUR_NAME
F2M_ENABLE_DEEPFACE=1
```

> **Note:** Add `http://127.0.0.1:8501/callback` as a Redirect URI in your Spotify Developer Dashboard.

### 5. Run the application
```bash
streamlit run app.py
```

---

## Project Structure

```
face2melody/
├── app.py                      # Main Streamlit application (UI + experiment logic)
├── experiment_utils.py         # Fusion engine, Spotify recommendation, logging
├── feedback_learning.py        # Feedback-based track reranking
├── analytics_gen.py            # Research analytics dashboard
├── requirements.txt
├── emotion_detection/
│   ├── facial_utils.py         # Webcam capture + DeepFace pipeline
│   ├── text_utils.py           # French NLP emotion classifier
│   └── multimodal_fusion.py    # Weighted probability fusion
├── recommender/
│   ├── emotion_detector.py     # DeepFace wrapper (thread-safe, lazy import)
│   ├── spotify_interface.py    # Spotify OAuth + recommendations client
│   └── playlist_generator.py  # Bulk playlist generation helper
└── logs/                       # Participant JSONL logs (gitignored)
```

---

## Analytics

The `analytics_gen.py` module provides a standalone research dashboard that reads all participant JSONL logs and computes:

- Mean rating, like rate, dislike rate, preview play rate per scenario
- Mean Spotify audio feature distance to emotion target (valence, energy, tempo)
- Mean recommendation generation time
- Side-by-side S1 vs S2 scenario comparison
- Interactive charts (Altair)

---

## Privacy & Data

Participant experiment logs are stored locally in `logs/` and are excluded from version control via `.gitignore`. No personal data is transmitted to any third-party service other than Spotify (standard OAuth flow). Facial analysis runs entirely on-device using DeepFace — no images are stored or sent remotely.

---

## Author

**Daghsen** — Master's Research Project  
Human-Computer Interaction / Affective Computing

---

## License

This project is released for academic and research purposes. See [LICENSE](LICENSE) for details.
