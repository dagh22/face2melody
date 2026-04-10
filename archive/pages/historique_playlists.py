# python
"""
Page Streamlit pour consulter l'historique des playlists et des feedbacks.
"""
import os, glob, json
import streamlit as st
import pandas as pd

LOG_DIR = "logs"
PLAYLIST_LOG = os.path.join(LOG_DIR, "playlist_log.json")
FEEDBACK_LOG = os.path.join(LOG_DIR, "feedback_log.json")


def load_jsonl_glob(pattern: str):
    rows = []
    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except:
                        continue
        except FileNotFoundError:
            continue
    return rows


def load_json_lines(path: str):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except:
                continue
    return rows


def main():
    st.title("Historique playlists & feedbacks")

    st.subheader("Expérimentations (JSONL)")
    data = load_jsonl_glob(os.path.join(LOG_DIR, "experiment_*.jsonl"))
    if data:
        df = pd.json_normalize(data)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Exporter CSV",
            df.to_csv(index=False).encode("utf-8"),
            "experiment_logs.csv",
            "text/csv",
        )
    else:
        st.info("Aucun fichier experiment_*.jsonl trouvé.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Playlists (legacy)")
        rows = load_json_lines(PLAYLIST_LOG)
        if rows:
            st.dataframe(pd.json_normalize(rows), use_container_width=True)
        else:
            st.caption("Aucun playlist_log.json")

    with col2:
        st.subheader("Feedbacks (legacy)")
        rows = load_json_lines(FEEDBACK_LOG)
        if rows:
            st.dataframe(pd.json_normalize(rows), use_container_width=True)
        else:
            st.caption("Aucun feedback_log.json")


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    main()
