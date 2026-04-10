# Face2Melody V3 - Plateforme d'IA Affective Conversationnelle

## 🎯 Vision Scientifique (M.Sc. IA - UQAM)
- **Concept :** Passer d'un outil de recommandation à un "Compagnon Émotionnel".
- **Innovation Majeure :** Fusion Trimode (Vision + Texte + Physiologie/BPM).
- **Agnosticisme :** Recommandation universelle (Spotify, YouTube, Apple Music).

## 🧠 Protocole de Fusion Trimode (Late Fusion)
L'agent doit arbitrer entre trois signaux d'entrée pour générer l'état affectif :
1. **Vision (VGG-Face) :** Probabilités brutes des expressions faciales.
2. **Texte (NLP) :** Analyse sémantique et tonale du chat.
3. **Physiologie (BPM) :** Rythme cardiaque (réel ou simulé).
   - *Règle d'arbitrage :* Si le BPM est > 100 mais le visage est neutre, suspecter une anxiété latente ou un stress physique.
   - *Règle de fiabilité :* Si la luminosité caméra est faible, donner un poids de 70% au Texte et 30% au BPM.

## 🤖 Comportement de l'Agent (Persona)
- **Ton :** Empathique, professionnel et scientifique (XAI).
- **Langue :** Français (pour l'utilisateur) / Commentaires de code en Français.
- **Rôle :** Ne pas se contenter de donner une liste ; expliquer *pourquoi* (ex: "Votre rythme cardiaque élevé suggère un besoin de calme...").

## 🛠️ Spécifications Techniques
- **Interface :** Mode Chat (Streamlit `st.chat_message`).
- **Sortie de l'Agent (Format Strict JSON) :**
  ```json
  {
    "emotion_unifiee": string,
    "bpm_analysis": string,
    "analyse_cognitive_interne": string (XAI technique),
    "message_utilisateur": string (Justification empathique),
    "confidence_score": float,
    "music_params": {
      "target_valence": float (0.0 to 1.0),
      "target_energy": float (0.0 to 1.0),
      "suggested_artists": list,
      "suggested_genres": list
    }
  }