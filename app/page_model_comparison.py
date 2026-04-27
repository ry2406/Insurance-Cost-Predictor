"""Page 3: Model comparison dashboard."""

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SAVED_DIR = ROOT / "saved_models"


@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_quantile_predictions():
    path = SAVED_DIR / "quantile_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def build_comparison_table():
    rows = []

    rf_path = SAVED_DIR / "rf_metrics.json"
    if rf_path.exists():
        rf = load_json(rf_path)
        rows.append({
            "Model": "Random Forest",
            "MAE": rf["metrics"]["MAE"],
            "RMSE": rf["metrics"]["RMSE"],
            "R2": rf["metrics"]["R2"],
            "Interval Coverage": None,
            "Avg Interval Width": None,
            "Type": "Point prediction",
        })

    xgb_path = SAVED_DIR / "xgb_metrics.json"
    if xgb_path.exists():
        xgb = load_json(xgb_path)
        rows.append({
            "Model": "XGBoost",
            "MAE": xgb["metrics"]["MAE"],
            "RMSE": xgb["metrics"]["RMSE"],
            "R2": xgb["metrics"]["R2"],
            "Interval Coverage": None,
            "Avg Interval Width": None,
            "Type": "Point prediction",
        })

    mlp_path = SAVED_DIR / "mlp_metrics.json"
    if mlp_path.exists():
        mlp = load_json(mlp_path)
        rows.append({
            "Model": "MLP",
            "MAE": mlp["mae"],
            "RMSE": mlp["rmse"],
            "R2": mlp["r2"],
            "Interval Coverage": None,
            "Avg Interval Width": None,
            "Type": "Point prediction",
        })

    q_path = SAVED_DIR / "quantile_metrics.json"
    if q_path.exists():
        q = load_json(q_path)
        rows.append({
            "Model": "Quantile Regression",
            "MAE": q["mae_median_dollar"],
            "RMSE": q["rmse_median_dollar"],
            "R2": q["r2_median_dollar"],
            "Interval Coverage": q["interval_80_coverage"],
            "Avg Interval Width": q["avg_interval_width_dollar"],
            "Type": "Prediction interval",
        })

    return pd.DataFrame(rows)


def metric_bar_chart(df, metric, title):
    plot_df = df.dropna(subset=[metric]).copy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(plot_df["Model"], plot_df[metric])
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig


def quantile_interval_plot(pred_df, n=50):
    plot_df = pred_df.head(n).copy()

    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(plot_df))

    ax.plot(x, plot_df["actual_charges"], label="Actual")
    ax.plot(x, plot_df["q50"], label="Median prediction")
    ax.fill_between(
        x,
        plot_df["q10"],
        plot_df["q90"],
        alpha=0.25,
        label="80% interval"
    )

    ax.set_title("Quantile Regression Prediction Intervals (first 50 test cases)")
    ax.set_xlabel("Test sample")
    ax.set_ylabel("Charges ($)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def show():
    st.title("Model Comparison Dashboard")
    st.write(
        "This page compares the main predictive models used in the project. "
        "We report standard regression metrics for point prediction models, "
        "and also show uncertainty information for quantile regression."
    )

    df = build_comparison_table()

    if df.empty:
        st.error("No saved model metrics found. Please run the training scripts first.")
        return

    st.markdown("---")
    st.subheader("Overall Comparison Table")
    st.dataframe(df)

    st.markdown("---")
    st.subheader("Best Model Snapshot")

    best_mae_model = df.dropna(subset=["MAE"]).sort_values("MAE").iloc[0]
    best_rmse_model = df.dropna(subset=["RMSE"]).sort_values("RMSE").iloc[0]
    best_r2_model = df.dropna(subset=["R2"]).sort_values("R2", ascending=False).iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Lowest MAE", best_mae_model["Model"], f'{best_mae_model["MAE"]:.2f}')
    c2.metric("Lowest RMSE", best_rmse_model["Model"], f'{best_rmse_model["RMSE"]:.2f}')
    c3.metric("Highest R2", best_r2_model["Model"], f'{best_r2_model["R2"]:.4f}')

    st.markdown("---")
    st.subheader("Metric Comparison")
    tab1, tab2, tab3, tab4 = st.tabs(["MAE", "RMSE", "R2", "Quantile Intervals"])

    with tab1:
        st.pyplot(metric_bar_chart(df, "MAE", "MAE by Model"))

    with tab2:
        st.pyplot(metric_bar_chart(df, "RMSE", "RMSE by Model"))

    with tab3:
        st.pyplot(metric_bar_chart(df, "R2", "R2 by Model"))

    with tab4:
        q_path = SAVED_DIR / "quantile_metrics.json"
        pred_df = load_quantile_predictions()

        if q_path.exists():
            q = load_json(q_path)
            c1, c2 = st.columns(2)
            c1.metric("80% Interval Coverage", f'{q["interval_80_coverage"]:.3f}')
            c2.metric("Average Interval Width", f'${q["avg_interval_width_dollar"]:,.2f}')

        if pred_df is not None:
            st.pyplot(quantile_interval_plot(pred_df))
        else:
            st.info("No quantile prediction CSV found.")

    st.markdown("---")
    st.subheader("Takeaways")
    st.write(
        "- XGBoost currently has the strongest overall point-prediction performance.\n"
        "- Random Forest is close behind and remains easy to interpret.\n"
        "- MLP performs reasonably well and captures nonlinear patterns.\n"
        "- Quantile Regression is weaker for point prediction, but it provides useful uncertainty intervals."
    )