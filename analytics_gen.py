import os, glob, json
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

# Dépendances optionnelles
try:
    import altair as alt

    HAS_ALT = True
except Exception:
    HAS_ALT = False

import streamlit as st

DEFAULT_TARGETS = {
    "happy": {"val": 0.85, "eng": 0.75},
    "sad": {"val": 0.20, "eng": 0.35},
    "angry": {"val": 0.30, "eng": 0.90},
    "neutral": {"val": 0.55, "eng": 0.50},
}


def _find_feedback_files(
    log_dir: str, pattern: str = "experiment_*.jsonl"
) -> List[str]:
    p = os.path.join(log_dir, pattern)
    return sorted(glob.glob(p))


def _load_jsonl(paths: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            continue
    return rows

def _extract_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    # --- 0. Normalisation du timestamp ---
    ts = rec.get("timestamp")
    # Dans tes logs récents c'est un entier en millisecondes depuis l'époque Unix
    if isinstance(ts, (int, float)):
        try:
            ts = datetime.fromtimestamp(ts / 1000.0)
        except Exception:
            ts = None

    # 1. Audio features brutes (peuvent être vides si Spotify renvoie 403)
    af = rec.get("audio_features") or {}
    seeds = rec.get("seeds") or {}
    prof = seeds.get("profile") or {}

    # 2. Emotion finale = final_emotion OU fused_label OU neutral
    final_emotion = (
        (rec.get("final_emotion") or rec.get("fused_label") or "neutral")
    ).lower()

    # 3. Cibles de valence / énergie (pour calculer dist2_target uniquement)
    tgt_cfg = DEFAULT_TARGETS.get(final_emotion, DEFAULT_TARGETS["neutral"])
    tgt_val = prof.get("mean_valence", tgt_cfg["val"])
    tgt_eng = prof.get("mean_energy", tgt_cfg["eng"])

    # 4. VRAIES valeurs valence/energy : on ne met plus les valeurs cibles par défaut
    val = af.get("valence")
    eng = af.get("energy")

    # 5. Distance au target (seulement si on a des vraies features numériques)
    d2 = None
    try:
        if isinstance(val, (int, float)) and isinstance(eng, (int, float)):
            dv = float(val) - float(tgt_val)
            de = float(eng) - float(tgt_eng)
            d2 = dv * dv + de * de
    except Exception:
        d2 = None

    # 6. Extraction track_id depuis l'URI
    uri = rec.get("track_uri")
    tid = None
    if isinstance(uri, str) and ":" in uri:
        tid = uri.split(":")[-1]

    # 7. Construction de la ligne normalisée
    return {
        "participant_id": rec.get("participant_id") or rec.get("pid"),
        "scenario": rec.get("scenario"),
        "timestamp": ts,  # <-- on met le timestamp normalisé ici
        "final_emotion": final_emotion,
        "rating": rec.get("rating"),
        "like": bool(rec.get("like", False)),
        "dislike": bool(rec.get("dislike", False)),
        "preview_played": bool(rec.get("preview_played") or rec.get("played", False)),
        "track_uri": uri,
        "track_id": tid,
        "valence": val,
        "energy": eng,
        "dist2_target": d2,
        "gen_time_ms": rec.get("gen_time_ms"),
    }

def _build_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    data = [_extract_row(r) for r in rows]
    df = pd.DataFrame(data)

    if "timestamp" in df.columns:
        with pd.option_context("mode.chained_assignment", None):
            # 1) On convertit TOUT en datetime avec utc=True
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
                utc=True,
            )
            # 2) Puis on enlève la timezone pour éviter les conflits tz-aware / tz-naive
            df["timestamp"] = df["timestamp"].dt.tz_convert(None)

    for col in ("rating", "gen_time_ms", "valence", "energy", "dist2_target"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _kpis(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    k = {}
    k["n_feedbacks"] = int(len(df))
    k["n_pids"] = int(df["participant_id"].nunique()) if "participant_id" in df else 0
    k["mean_rating"] = float(df["rating"].dropna().mean()) if "rating" in df else None
    k["like_rate"] = float(df["like"].mean()) if "like" in df else None
    k["dislike_rate"] = float(df["dislike"].mean()) if "dislike" in df else None
    k["preview_rate"] = (
        float(df["preview_played"].mean()) if "preview_played" in df else None
    )
    k["mean_gen_ms"] = (
        float(df["gen_time_ms"].dropna().mean()) if "gen_time_ms" in df else None
    )
    k["mean_dist2"] = (
        float(df["dist2_target"].dropna().mean()) if "dist2_target" in df else None
    )
    return k

def _prepare_progress(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute colonnes utiles pour analyse S1 vs S2."""
    if df.empty:
        return df

    d = df.copy()
    d["rating"] = pd.to_numeric(d.get("rating"), errors="coerce")
    d["like"] = d.get("like", False).fillna(False).astype(bool)
    d["dislike"] = d.get("dislike", False).fillna(False).astype(bool)

    d["is_positive"] = d["like"] | (d["rating"].fillna(0) >= 4)
    d["is_negative"] = d["dislike"] | (d["rating"].fillna(0) <= 2)

    d = d.sort_values("timestamp")
    d["event_idx"] = np.arange(len(d)) + 1
    d["scenario_event_idx"] = d.groupby("scenario").cumcount() + 1
    return d


def _scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    d = _prepare_progress(df)
    g = d.groupby("scenario", dropna=False)

    out = g.agg(
        feedbacks=("scenario", "size"),
        participants=("participant_id", pd.Series.nunique),
        avg_rating=("rating", "mean"),
        like_rate=("like", "mean"),
        dislike_rate=("dislike", "mean"),
        positive_rate=("is_positive", "mean"),
        negative_rate=("is_negative", "mean"),
        mean_gen_ms=("gen_time_ms", "mean") if "gen_time_ms" in d.columns else ("scenario", "size"),
    ).reset_index()

    if "gen_time_ms" not in d.columns:
        out["mean_gen_ms"] = np.nan

    out["avg_rating"] = out["avg_rating"].round(2)
    for c in ["like_rate", "dislike_rate", "positive_rate", "negative_rate"]:
        out[c] = (out[c] * 100).round(1)
    out["mean_gen_ms"] = out["mean_gen_ms"].round(0)
    return out


def _before_after(df: pd.DataFrame) -> pd.DataFrame:
    """Début vs Fin (2 moitiés) par scénario."""
    if df.empty:
        return pd.DataFrame()

    d = _prepare_progress(df)
    d = d.sort_values(["scenario", "timestamp"])

    # Découpe en 2 parties par scénario
    def _split_two(s):
        r = s.rank(method="first")
        return pd.qcut(r, 2, labels=["Début", "Fin"])

    d["phase"] = d.groupby("scenario")["timestamp"].transform(_split_two)

    out = d.groupby(["scenario", "phase"]).agg(
        n=("scenario", "size"),
        avg_rating=("rating", "mean"),
        like_rate=("like", "mean"),
        dislike_rate=("dislike", "mean"),
        positive_rate=("is_positive", "mean"),
    ).reset_index()

    out["avg_rating"] = out["avg_rating"].round(2)
    for c in ["like_rate", "dislike_rate", "positive_rate"]:
        out[c] = (out[c] * 100).round(1)
    return out


def _advice(df: pd.DataFrame) -> List[str]:
    adv = []
    if df.empty:
        return adv
    if "rating" in df and df["rating"].notna().any():
        by_em = (
            df.groupby("final_emotion")["rating"].mean().sort_values(ascending=False)
        )
        if len(by_em) >= 1:
            top_em = by_em.index[0]
            adv.append(
                f"Les titres associés à '{top_em}' ont la meilleure note moyenne ({by_em.iloc[0]:.2f})."
            )
    if "like" in df:
        like_by_em = (
            df.groupby("final_emotion")["like"].mean().sort_values(ascending=False)
        )
        if len(like_by_em) >= 1:
            adv.append(
                f"Taux de like le plus élevé pour '{like_by_em.index[0]}' ({like_by_em.iloc[0]*100:.1f}%)."
            )
    if "dislike" in df:
        dis_by_em = (
            df.groupby("final_emotion")["dislike"].mean().sort_values(ascending=False)
        )
        if len(dis_by_em) >= 1 and dis_by_em.iloc[0] > 0:
            adv.append(
                f"Attention: '{dis_by_em.index[0]}' a le plus de dislikes ({dis_by_em.iloc[0]*100:.1f}%)."
            )
    if "dist2_target" in df and df["dist2_target"].notna().any():
        by_em_d = df.groupby("final_emotion")["dist2_target"].mean().sort_values()
        adv.append(
            f"Alignement cible (valence/énergie) le plus serré sur '{by_em_d.index[0]}' (dist²={by_em_d.iloc[0]:.3f})."
        )
    return adv


def _charts(df: pd.DataFrame) -> Tuple[Any, Any, Any]:
    if not HAS_ALT or df.empty:
        return None, None, None
    base = alt.Chart(df)
    # Bar: note moyenne par émotion
    c1 = (
        base.mark_bar()
        .encode(
            x=alt.X("final_emotion:N", title="Émotion"),
            y=alt.Y("mean(rating):Q", title="Note moyenne"),
            tooltip=[alt.Tooltip("mean(rating):Q", title="Note moyenne", format=".2f")],
        )
        .properties(height=200)
    )
    # Bar: like/dislike par émotion
    df_lr = df.assign(
        like_rate=df["like"].astype(float), dislike_rate=df["dislike"].astype(float)
    )
    c2_like = (
        alt.Chart(df_lr)
        .mark_bar(color="#4caf50")
        .encode(
            x=alt.X("final_emotion:N", title="Émotion"),
            y=alt.Y("mean(like_rate):Q", title="Like rate"),
            tooltip=[alt.Tooltip("mean(like_rate):Q", title="Like", format=".1%")],
        )
        .properties(height=140)
    )
    c2_dis = (
        alt.Chart(df_lr)
        .mark_bar(color="#f44336")
        .encode(
            x=alt.X("final_emotion:N", title="Émotion"),
            y=alt.Y("mean(dislike_rate):Q", title="Dislike rate"),
            tooltip=[
                alt.Tooltip("mean(dislike_rate):Q", title="Dislike", format=".1%")
            ],
        )
        .properties(height=140)
    )
    # Scatter: valence vs énergie
    df_sc = df.dropna(subset=["valence", "energy"]).copy()
    c3 = None
    if len(df_sc) >= 5:
        c3 = (
            alt.Chart(df_sc)
            .mark_circle(size=70, opacity=0.6)
            .encode(
                x=alt.X("valence:Q", scale=alt.Scale(domain=[0, 1]), title="Valence"),
                y=alt.Y("energy:Q", scale=alt.Scale(domain=[0, 1]), title="Énergie"),
                color=alt.Color("final_emotion:N", title="Émotion"),
                tooltip=[
                    "participant_id",
                    "scenario",
                    "rating",
                    "like",
                    "dislike",
                    "valence",
                    "energy",
                ],
            )
            .properties(height=280)
        )

    return c1, (c2_like | c2_dis), c3


def _export_reports(df: pd.DataFrame, out_dir: str = "reports") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    paths = {}
    # CSV feedback
    p_csv = os.path.join(out_dir, f"feedback_{ts}.csv")
    df.to_csv(p_csv, index=False)
    paths["csv"] = p_csv
    # KPIs JSON
    p_json = os.path.join(out_dir, f"kpis_{ts}.json")
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(_kpis(df), f, ensure_ascii=False, indent=2)
    paths["kpis"] = p_json
    # Graphiques en HTML (si Altair)
    if HAS_ALT:
        try:
            c1, c2, c3 = _charts(df)
            if c1 is not None:
                p_html1 = os.path.join(out_dir, f"ratings_by_emotion_{ts}.html")
                c1.save(p_html1)
                paths["ratings_html"] = p_html1
            if c2 is not None:
                p_html2 = os.path.join(out_dir, f"like_dislike_{ts}.html")
                c2.save(p_html2)
                paths["like_dislike_html"] = p_html2
            if c3 is not None:
                p_html3 = os.path.join(out_dir, f"valence_energy_{ts}.html")
                c3.save(p_html3)
                paths["valence_energy_html"] = p_html3
        except Exception:
            pass
    return paths


# --- UI rendue réutilisable depuis l'app principale ---
def render():
    # Déplacé: toute l'UI Streamlit vit maintenant dans cette fonction.
    st.title("Face2Melody — Analytics")

    with st.sidebar:
        st.header("Sources")
        log_dir = st.text_input("Dossier logs", value="logs", key="ag_logs_dir")
        pattern = st.text_input(
            "Motif fichiers", value="experiment_*.jsonl", key="ag_pattern"
        )
        files = _find_feedback_files(log_dir, pattern)
        st.caption(f"{len(files)} fichiers trouvés.")
        reload_btn = st.button("Recharger", key="ag_reload")

    rows = _load_jsonl(files)
    df = _build_df(rows)

    # Filtres
    c1, c2, c3 = st.columns(3)
    with c1:
        pids = (
            sorted([p for p in df["participant_id"].dropna().unique()])
            if not df.empty and "participant_id" in df
            else []
        )
        sel_pids = st.multiselect(
            "Participant ID",
            pids,
            default=pids[:5] if len(pids) > 5 else pids,
            key="ag_pids",
        )
    with c2:
        scens = (
            sorted([s for s in df["scenario"].dropna().unique()])
            if not df.empty and "scenario" in df
            else []
        )
        sel_sc = st.multiselect("Scénarios", scens, default=scens, key="ag_sc")
    with c3:
        min_rating = st.slider("Note min", 1, 5, 1, key="ag_min_rating")
        export_now = st.button("Exporter (reports/)", key="ag_export")

    if not df.empty:
        import pandas as _pd

        m = _pd.Series(True, index=df.index)
        if sel_pids:
            m &= df["participant_id"].isin(sel_pids)
        if sel_sc:
            m &= df["scenario"].isin(sel_sc)
        if "rating" in df:
            m &= df["rating"].fillna(0) >= min_rating
        dff = df[m].copy()
    else:
        dff = df

    k = _kpis(dff) if not dff.empty else {}
    cA, cB, cC, cD, cE, cF = st.columns(6)
    with cA:
        st.metric("Feedbacks", k.get("n_feedbacks", 0))
    with cB:
        st.metric("Participants", k.get("n_pids", 0))
    with cC:
        st.metric(
            "Note moyenne",
            (
                f"{k.get('mean_rating', 0):.2f}"
                if k.get("mean_rating") is not None
                else "—"
            ),
        )
    with cD:
        st.metric(
            "Like rate",
            (
                f"{k.get('like_rate', 0)*100:.1f}%"
                if k.get("like_rate") is not None
                else "—"
            ),
        )
    with cE:
        st.metric(
            "Dislike rate",
            (
                f"{k.get('dislike_rate', 0)*100:.1f}%"
                if k.get("dislike_rate") is not None
                else "—"
            ),
        )
    with cF:
        st.metric(
            "Temps gen (ms)",
            (
                f"{k.get('mean_gen_ms', 0):.0f}"
                if k.get("mean_gen_ms") is not None
                else "—"
            ),
        )

    adv = _advice(dff)
    if adv:
        with st.expander("Conseils rapides"):
            for a in adv:
                st.write("•", a)
                
            # =========================
    # Comparaison S1 vs S2
    # =========================
    st.divider()
    st.subheader("Comparaison S1 vs S2 (qualité des suggestions)")

    dff_ab = dff[dff["scenario"].isin(["S1", "S2"])].copy() if not dff.empty else dff

    if dff_ab is None or dff_ab.empty:
        st.info("Pas assez de données pour comparer S1 vs S2 (filtre actuel).")
    else:
        # Tableau KPI par scénario
        sum_df = _scenario_summary(dff_ab)
        st.dataframe(sum_df, use_container_width=True)

        # Avant vs Après
        st.markdown("### Avant vs Après (début/fin)")
        ba = _before_after(dff_ab)
        st.dataframe(ba, use_container_width=True)

        # Progression (rolling)
        st.markdown("### Progression (graduellement)")
        window = st.slider("Fenêtre rolling", 5, 60, 20, 5, key="ag_roll")

        dprog = _prepare_progress(dff_ab).sort_values(["scenario", "timestamp"])
        dprog["like_roll"] = (
            dprog.groupby("scenario")["like"]
            .transform(lambda s: s.rolling(window, min_periods=max(5, window//3)).mean())
        )
        dprog["rating_roll"] = (
            dprog.groupby("scenario")["rating"]
            .transform(lambda s: s.rolling(window, min_periods=max(5, window//3)).mean())
        )

        if HAS_ALT:
            # Like rate rolling
            st.write("**Like rate (rolling)**")
            line1 = (
                alt.Chart(dprog.dropna(subset=["like_roll"]))
                .mark_line()
                .encode(
                    x=alt.X("scenario_event_idx:Q", title="Itération (dans le scénario)"),
                    y=alt.Y("like_roll:Q", title="Like rate (rolling)"),
                    color="scenario:N",
                    tooltip=["scenario", "scenario_event_idx", alt.Tooltip("like_roll:Q", format=".2f")],
                )
                .properties(height=220)
            )
            st.altair_chart(line1, use_container_width=True)

            # Rating rolling
            st.write("**Note moyenne (rolling)**")
            line2 = (
                alt.Chart(dprog.dropna(subset=["rating_roll"]))
                .mark_line()
                .encode(
                    x=alt.X("scenario_event_idx:Q", title="Itération (dans le scénario)"),
                    y=alt.Y("rating_roll:Q", title="Note moyenne (rolling)"),
                    color="scenario:N",
                    tooltip=["scenario", "scenario_event_idx", alt.Tooltip("rating_roll:Q", format=".2f")],
                )
                .properties(height=220)
            )
            st.altair_chart(line2, use_container_width=True)

        else:
            st.caption("Altair non disponible : courbes rolling non affichées.")


    # Graphiques
    if HAS_ALT and not dff.empty:
        ch1, ch2, ch3 = _charts(dff)
        if ch1 is not None:
            st.subheader("Notes moyennes par émotion")
            st.altair_chart(ch1, use_container_width=True)
        if ch2 is not None:
            st.subheader("Like / Dislike par émotion")
            st.altair_chart(ch2, use_container_width=True)
        if ch3 is not None:
            st.subheader("Valence vs Énergie")
            st.altair_chart(ch3, use_container_width=True)
        else:
            st.subheader("Valence vs Énergie")
            st.info("Valence/Énergie non disponibles dans les logs (audio_features vides). Graphique masqué.")

        if dff.empty:
            st.info("Aucun feedback chargé.")
        else:
            st.caption("Altair non disponible, affichage des tableaux uniquement.")

    # Tableaux
    st.subheader("Derniers feedbacks")
    if not dff.empty:
        st.dataframe(
            dff.sort_values("timestamp", ascending=False).head(200)[
                [
                    "timestamp",
                    "participant_id",
                    "scenario",
                    "final_emotion",
                    "rating",
                    "like",
                    "dislike",
                    "valence",
                    "energy",
                    "dist2_target",
                    "track_id",
                ]
            ],
            use_container_width=True,
        )

    # Export
    if export_now and not dff.empty:
        out = _export_reports(dff, out_dir="reports")
        st.success("Export réalisé.")
        for k_name, pth in out.items():
            st.write(f"{k_name}: {pth}")

    st.caption(
        "Astuce: Altair est optionnel. Pour un usage autonome: `streamlit run analytics_gen.py`."
    )


# --- Mode autonome (lance la page seule) ---
if __name__ == "__main__":
    st.set_page_config(
        page_title="Face2Melody — Analytics", page_icon="📈", layout="wide"
    )
    render()
