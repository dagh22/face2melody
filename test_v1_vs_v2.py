"""
test_v1_vs_v2.py — Comparaison V1 (heuristique) vs V2 (agent cognitif LLM)
Face2Melody — Chapitre 6 du mémoire

Usage :
    python test_v1_vs_v2.py

Prérequis :
    - ANTHROPIC_API_KEY défini dans .env
    - pip install anthropic>=0.30.0
"""
from __future__ import annotations

import json
import os
import sys

# Charger .env avant toute chose
from dotenv import load_dotenv
load_dotenv(override=True)

from experiment_utils import _fuse_heuristic
from agent_logic import EmotionFusionAgent

# ─────────────────────────────────────────────────────────────────────────────
# Cas de test
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "1. Congruence happy (face + texte concordent)",
        "face_probs": {"happy": 0.75, "sad": 0.05, "angry": 0.05, "neutral": 0.15},
        "text_probs": {"happy": 0.60, "sad": 0.10, "angry": 0.05, "neutral": 0.25},
        "text_raw": "Je me sens vraiment bien aujourd'hui, plein d'énergie !",
    },
    {
        "name": "2. Dissonance clé thèse : visage joyeux + texte triste",
        "face_probs": {"happy": 0.80, "sad": 0.05, "angry": 0.05, "neutral": 0.10},
        "text_probs": {"happy": 0.05, "sad": 0.70, "angry": 0.05, "neutral": 0.20},
        "text_raw": "je me sens un peu fatigué et mélancolique, c'est dur aujourd'hui",
    },
    {
        "name": "3. Sarcasme : visage joyeux + texte en colère",
        "face_probs": {"happy": 0.70, "sad": 0.05, "angry": 0.10, "neutral": 0.15},
        "text_probs": {"happy": 0.05, "sad": 0.10, "angry": 0.75, "neutral": 0.10},
        "text_raw": "Super, encore une réunion inutile... quelle journée formidable.",
    },
    {
        "name": "4. Signal facial faible + texte très triste",
        "face_probs": {"happy": 0.25, "sad": 0.25, "angry": 0.25, "neutral": 0.25},
        "text_probs": {"happy": 0.05, "sad": 0.80, "angry": 0.05, "neutral": 0.10},
        "text_raw": "je suis vraiment déprimé, tout va mal",
    },
    {
        "name": "5. Neutre + pas de texte",
        "face_probs": {"happy": 0.10, "sad": 0.10, "angry": 0.10, "neutral": 0.70},
        "text_probs": {"happy": 0.25, "sad": 0.25, "angry": 0.25, "neutral": 0.25},
        "text_raw": None,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────────────────────────

def run_v1(case: dict) -> dict:
    fd, dom = _fuse_heuristic(case["face_probs"], case["text_probs"], w_face=0.6)
    return {
        "dominant_emotion": dom,
        "fused_distribution": fd,
        "valence": None,
        "arousal": None,
        "reasoning": "[V1 heuristique — pas de raisonnement]",
        "from_fallback": False,
    }


def run_v2(case: dict, agent: EmotionFusionAgent) -> dict:
    result = agent.analyze(
        case["face_probs"],
        case["text_probs"],
        text_raw=case.get("text_raw"),
        w_face=0.6,
    )
    return result.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Affichage
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_dist(d: dict) -> str:
    return "  ".join(f"{k}={v:.2f}" for k, v in sorted(d.items()))


def print_comparison(case: dict, v1: dict, v2: dict) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"TEST : {case['name']}")
    if case.get("text_raw"):
        print(f"Texte : \"{case['text_raw']}\"")
    print()
    print(f"  V1 → émotion={v1['dominant_emotion']}")
    print(f"       distrib={_fmt_dist(v1['fused_distribution'])}")
    print()
    print(f"  V2 → émotion={v2['dominant_emotion']}")
    if v2.get("valence") is not None:
        print(f"       valence={v2['valence']:.2f}  arousal={v2['arousal']:.2f}")
    print(f"       distrib={_fmt_dist(v2['fused_distribution'])}")
    print(f"  Raisonnement V2 :")
    print(f"    {v2['reasoning']}")
    if v2.get("from_fallback"):
        print("  *** Fallback V1 utilisé (LLM indisponible) ***")
    print()
    # Résumé concordance
    agree = v1["dominant_emotion"] == v2["dominant_emotion"]
    print(f"  → V1 et V2 concordent : {'OUI' if agree else 'NON (divergence intéressante !)'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Face2Melody — Comparaison V1 vs V2 (Chapitre 6)")
    print(f"ANTHROPIC_API_KEY présente : {'OUI' if os.getenv('ANTHROPIC_API_KEY') else 'NON — fallback V1 sera utilisé'}")

    agent = EmotionFusionAgent()
    results = []

    for case in TEST_CASES:
        v1 = run_v1(case)
        v2 = run_v2(case, agent)
        print_comparison(case, v1, v2)
        results.append({"case": case["name"], "text_raw": case.get("text_raw"), "v1": v1, "v2": v2})

    # Sauvegarde
    os.makedirs("logs", exist_ok=True)
    out_path = "logs/v1_vs_v2_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nRésultats sauvegardés dans : {out_path}")

    # Bilan divergences
    divergences = [r for r in results if r["v1"]["dominant_emotion"] != r["v2"]["dominant_emotion"]]
    print(f"\nBilan : {len(divergences)}/{len(results)} cas avec divergence V1/V2")
    for d in divergences:
        print(f"  - {d['case']} : V1={d['v1']['dominant_emotion']} / V2={d['v2']['dominant_emotion']}")


if __name__ == "__main__":
    main()
