"""
app.py
Streamlit dashboard for the Personal Expense Anomaly Detector.

Run with:
    streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from data.generate_data import generate_expenses
from src.preprocess import engineer_features, get_feature_matrix
from src.model import ExpenseAnomalyDetector


# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Expense Anomaly Detector",
    page_icon="💳",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

uploaded = st.sidebar.file_uploader(
    "Upload your own CSV",
    type="csv",
    help="CSV must have columns: date, category, amount",
)

months = st.sidebar.slider("Months of synthetic data", 3, 12, 6)
n_anomalies = st.sidebar.slider("Injected anomalies (synthetic)", 4, 15, 8)
km_threshold = st.sidebar.slider("K-Means sensitivity (quantile)", 0.80, 0.99, 0.92, step=0.01)
if_contamination = st.sidebar.slider("Isolation Forest contamination", 0.01, 0.15, 0.05, step=0.01)

run_btn = st.sidebar.button("🔍 Run Detection", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works**\n\n"
    "1. Transactions are featurised (amount, category, time)\n"
    "2. K-Means clusters normal spending patterns\n"
    "3. Isolation Forest scores each transaction\n"
    "4. Both must agree to flag an anomaly\n"
)

# ── Main ──────────────────────────────────────────────────────────────────
st.title("💳 Personal Expense Anomaly Detector")
st.markdown("Detect unusual spending patterns using **K-Means clustering** and **Isolation Forest**.")

if "results_df" not in st.session_state:
    st.session_state.results_df = None

if run_btn or st.session_state.results_df is None:
    with st.spinner("Loading data and running models…"):

        # ── Load data ─────────────────────────────────────────────────────
        if uploaded is not None:
            raw_df = pd.read_csv(uploaded)
            raw_df["is_anomaly"] = False   # no ground truth for real uploads
        else:
            raw_df = generate_expenses(months=months, anomaly_count=n_anomalies)

        # ── Feature engineering ───────────────────────────────────────────
        df = engineer_features(raw_df)
        X, scaler = get_feature_matrix(df)

        # ── Model ─────────────────────────────────────────────────────────
        detector = ExpenseAnomalyDetector(
            km_threshold_q=km_threshold,
            if_contamination=if_contamination,
        )
        detector.fit(X)
        preds = detector.predict(X)

        # ── Merge results ─────────────────────────────────────────────────
        results_df = pd.concat([df.reset_index(drop=True), preds], axis=1)
        results_df["date_str"] = results_df["date"].dt.strftime("%Y-%m-%d %H:%M")

        # ── PCA for 2-D scatter ───────────────────────────────────────────
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
        results_df["pca_x"] = coords[:, 0]
        results_df["pca_y"] = coords[:, 1]

        st.session_state.results_df = results_df
        st.session_state.detector   = detector

results_df = st.session_state.results_df

# ── Summary metrics ───────────────────────────────────────────────────────
total       = len(results_df)
n_flagged   = results_df["anomaly"].sum()
flagged_pct = round(n_flagged / total * 100, 1)
total_spend = round(results_df["amount"].sum(), 2)
anomaly_amt = round(results_df.loc[results_df["anomaly"], "amount"].sum(), 2)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total:,}")
col2.metric("Anomalies Detected", f"{n_flagged}", f"{flagged_pct}% of total")
col3.metric("Total Spend", f"${total_spend:,.2f}")
col4.metric("Anomalous Spend", f"${anomaly_amt:,.2f}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Anomalies", "🗺️ Cluster View", "📈 Evaluation"])

# ── Tab 1 — Overview ──────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        spend_by_cat = (
            results_df.groupby("category")["amount"].sum().reset_index()
        )
        fig = px.bar(
            spend_by_cat.sort_values("amount", ascending=True),
            x="amount", y="category",
            orientation="h",
            title="Total spend by category",
            labels={"amount": "Total ($)", "category": ""},
            color="amount",
            color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        monthly = (
            results_df.assign(month=results_df["date"].dt.to_period("M").astype(str))
            .groupby(["month", "anomaly"])["amount"].sum().reset_index()
        )
        monthly["Type"] = monthly["anomaly"].map({True: "Anomalous", False: "Normal"})
        fig2 = px.bar(
            monthly, x="month", y="amount", color="Type",
            title="Monthly spend — normal vs anomalous",
            labels={"amount": "Total ($)", "month": ""},
            color_discrete_map={"Normal": "#4A90D9", "Anomalous": "#E24B4A"},
        )
        fig2.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # Amount distribution
    fig3 = px.histogram(
        results_df, x="amount", color="anomaly",
        nbins=50,
        title="Transaction amount distribution",
        labels={"amount": "Amount ($)", "anomaly": "Anomaly"},
        color_discrete_map={True: "#E24B4A", False: "#4A90D9"},
        opacity=0.75,
    )
    fig3.update_layout(barmode="overlay", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig3, use_container_width=True)


# ── Tab 2 — Anomalies ─────────────────────────────────────────────────────
with tab2:
    anomalies = results_df[results_df["anomaly"]].copy()

    if anomalies.empty:
        st.info("No anomalies detected. Try lowering the sensitivity settings.")
    else:
        st.markdown(f"**{len(anomalies)} transactions flagged by both models:**")

        display_cols = ["date_str", "category", "amount", "description",
                        "centroid_dist", "if_score"]
        rename_map   = {
            "date_str":      "Date",
            "category":      "Category",
            "amount":        "Amount ($)",
            "description":   "Description",
            "centroid_dist": "Cluster Distance",
            "if_score":      "Isolation Score",
        }
        st.dataframe(
            anomalies[display_cols].rename(columns=rename_map)
                      .sort_values("Amount ($)", ascending=False)
                      .reset_index(drop=True),
            use_container_width=True,
        )

        # Timeline of anomalies
        fig4 = px.scatter(
            anomalies,
            x="date", y="amount",
            color="category",
            size="centroid_dist",
            hover_data=["description", "centroid_dist"],
            title="Anomaly timeline",
            labels={"amount": "Amount ($)", "date": ""},
        )
        fig4.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig4, use_container_width=True)


# ── Tab 3 — Cluster View ──────────────────────────────────────────────────
with tab3:
    st.markdown(
        "Each point is a transaction projected onto 2 dimensions using PCA. "
        "Red points are detected anomalies."
    )
    fig5 = px.scatter(
        results_df,
        x="pca_x", y="pca_y",
        color=results_df["anomaly"].map({True: "Anomaly", False: "Normal"}),
        symbol="category",
        hover_data=["date_str", "category", "amount", "description"],
        title="K-Means cluster view (PCA projection)",
        labels={"pca_x": "PC 1", "pca_y": "PC 2",
                "color": "Type"},
        color_discrete_map={"Normal": "#4A90D9", "Anomaly": "#E24B4A"},
        opacity=0.7,
    )
    fig5.update_traces(marker=dict(size=7))
    fig5.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig5, use_container_width=True)


# ── Tab 4 — Evaluation ────────────────────────────────────────────────────
with tab4:
    if "is_anomaly" in results_df.columns and results_df["is_anomaly"].any():
        detector = st.session_state.detector
        eval_results = detector.evaluate(
            results_df[["anomaly"]], results_df["is_anomaly"]
        )

        st.subheader("Classification report")
        st.code(eval_results["report"])

        st.subheader("Confusion matrix")
        cm = eval_results["confusion_matrix"]
        fig6 = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["Normal", "Anomaly"],
            y=["Normal", "Anomaly"],
            color_continuous_scale="Blues",
            title="Confusion matrix",
        )
        fig6.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig6, use_container_width=True)

        # Per-method comparison
        st.subheader("Method comparison")
        comp = pd.DataFrame({
            "Method":    ["K-Means", "Isolation Forest", "Ensemble (both)"],
            "Flagged":   [
                results_df["km_flag"].sum(),
                results_df["if_flag"].sum(),
                results_df["anomaly"].sum(),
            ],
            "True Positives": [
                (results_df["km_flag"] & results_df["is_anomaly"]).sum(),
                (results_df["if_flag"] & results_df["is_anomaly"]).sum(),
                (results_df["anomaly"] & results_df["is_anomaly"]).sum(),
            ],
        })
        comp["Precision"] = (comp["True Positives"] / comp["Flagged"]).round(2)
        st.dataframe(comp, use_container_width=True)
    else:
        st.info(
            "Ground-truth labels are only available for synthetic data. "
            "Upload your own CSV and this tab will show precision/recall metrics "
            "if you include an `is_anomaly` column."
        )
