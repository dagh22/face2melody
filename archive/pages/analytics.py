# python
import glob, json
import pandas as pd
import streamlit as st
import altair as alt

st.title("Tableau de bord expérimentation")

rows = []
for path in glob.glob("logs/experiment_*.jsonl"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except:
                    pass
    except FileNotFoundError:
        pass

if not rows:
    st.info("Pas encore de données.")
    st.stop()

df = pd.json_normalize(rows)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Participants", df["participant_id"].nunique())
c2.metric("Pistes évaluées", len(df))
c3.metric("Like rate (%)", f"{100*df['like'].mean():.1f}" if "like" in df else "—")
c4.metric("Note moyenne", f"{df['rating'].mean():.2f}" if "rating" in df else "—")

if {"scenario", "rating"}.issubset(df.columns):
    st.altair_chart(
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="scenario",
            y=alt.Y("mean(rating):Q", title="Note moyenne"),
            color="scenario",
        ),
        use_container_width=True,
    )

if "like" in df.columns:
    like_df = df.groupby("scenario")["like"].mean().reset_index()
    like_df["like"] = like_df["like"] * 100
    st.altair_chart(
        alt.Chart(like_df)
        .mark_bar()
        .encode(
            x="scenario", y=alt.Y("like:Q", title="Like rate (%)"), color="scenario"
        ),
        use_container_width=True,
    )

if "final_emotion" in df.columns:
    st.altair_chart(
        alt.Chart(df)
        .mark_bar()
        .encode(x="final_emotion", y="count()", color="final_emotion"),
        use_container_width=True,
    )

if "gen_time_ms" in df.columns:
    st.altair_chart(
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="scenario",
            y=alt.Y("mean(gen_time_ms):Q", title="Temps moyen génération (ms)"),
            color="scenario",
        ),
        use_container_width=True,
    )

st.download_button(
    "Télécharger CSV",
    df.to_csv(index=False).encode("utf-8"),
    "experimentation.csv",
    "text/csv",
)
