"""
eda_utils.py
Medical Insurance Cost Predictor — EDA Visualization Utilities

Reusable plotting functions extracted from eda.ipynb.
Each function accepts a raw (unencoded) DataFrame and returns a
matplotlib Figure object for use with st.pyplot(fig) in Streamlit.

Author: Ruide Yin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


# Section 1.1 — Numerical feature distributions (2×2 hist + KDE)

def plot_numerical_distributions(df):
    """Histogram + KDE for age, bmi, children, charges with mean/median lines."""
    num_cols = ["age", "bmi", "children", "charges"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, col in zip(axes.flatten(), num_cols):
        sns.histplot(df[col], kde=True, bins=30, ax=ax, edgecolor="white")
        skew_val = df[col].skew()
        ax.set_title(f"Distribution of {col} (skew={skew_val:.2f})")
        ax.axvline(df[col].mean(), color="red", linestyle="--", label="mean")
        ax.axvline(df[col].median(), color="green", linestyle="--", label="median")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# Section 1.1 — Charges vs log(charges) skew comparison

def plot_charges_log_comparison(df):
    """Side-by-side histogram of raw charges and log-transformed charges."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].set_title(f"charges (skew={df['charges'].skew():.2f})")
    sns.histplot(df["charges"], kde=True, bins=30, ax=axes[0], edgecolor="white")

    log_charges = np.log1p(df["charges"])
    axes[1].set_title(f"log(charges) (skew={log_charges.skew():.2f})")
    sns.histplot(log_charges, kde=True, bins=30, ax=axes[1],
                 edgecolor="white", color="orange")

    fig.tight_layout()
    return fig


# Section 1.2 — Categorical feature value counts

def plot_categorical_counts(df):
    """Count plot with percentage annotations for sex, smoker, region."""
    cat_cols = ["sex", "smoker", "region"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, col in zip(axes, cat_cols):
        sns.countplot(data=df, x=col, hue=col, palette="Set2", ax=ax, edgecolor="white", legend=False)
        ax.set_title(f"Count of {col}")
        total = len(df)
        for p in ax.patches:
            pct = f"{100 * p.get_height() / total:.1f}%"
            ax.annotate(pct,
                        (p.get_x() + p.get_width() / 2.0, p.get_height()),
                        ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


# Section 2.1 — Categorical features vs charges (box plots)

def plot_categorical_vs_charges(df):
    """Box plot of charges grouped by each categorical feature."""
    cat_cols = ["sex", "smoker", "region"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col in zip(axes, cat_cols):
        sns.boxplot(data=df, x=col, y="charges", hue=col, palette="coolwarm", ax=ax, legend=False)
        ax.set_title(f"Charges by {col}")

    fig.tight_layout()
    return fig


# Section 2.1 — Numerical features vs charges (scatter, colored by smoker)

def plot_scatter_vs_charges(df):
    """Scatter plots of age, bmi, children vs charges, hue = smoker."""
    scatter_cols = ["age", "bmi", "children"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col in zip(axes, scatter_cols):
        sns.scatterplot(data=df, x=col, y="charges", hue="smoker",
                        palette="coolwarm", alpha=0.6, ax=ax, edgecolor=None)
        ax.set_title(f"{col} vs Charges")
        ax.legend(title="Smoker", fontsize=8)

    fig.tight_layout()
    return fig


# Section 2.1 — Charges by number of children (box plot)

def plot_charges_by_children(df):
    """Box plot of charges grouped by number of children."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="children", y="charges", hue="children", palette="viridis", ax=ax, legend=False)
    ax.set_title("Charges by Number of Children")
    return fig


# Section 2.1 — Age vs charges with linear fit by smoker status

def plot_age_vs_charges_regression(df):
    """Regression plot of age vs charges, separate fits for smoker/non-smoker."""
    g = sns.lmplot(data=df, x="age", y="charges", hue="smoker",
                   palette="coolwarm", height=5, aspect=1.6,
                   scatter_kws={"alpha": 0.5, "edgecolor": None})
    g.figure.suptitle("Age vs Charges with Linear Fit by Smoker Status", y=1.02)
    return g.figure


# Section 2.1 — Charges by age group and smoker status

def plot_charges_by_age_group(df):
    """Box plot of charges split by age group and smoker status."""
    tmp = df.copy()
    tmp["age_group"] = pd.cut(tmp["age"], bins=[17, 30, 45, 200],
                              labels=["18-30", "31-45", "46+"])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=tmp, x="age_group", y="charges", hue="smoker",
                palette="coolwarm", ax=ax)
    ax.set_title("Charges by Age Group and Smoker Status")
    ax.legend(title="Smoker")
    return fig


# Section 2.1 — Pairplot (age, bmi, charges by smoker)

def plot_pairplot(df):
    """Pairplot of age, bmi, charges colored by smoker status.

    Note: seaborn pairplot creates its own figure internally.
    """
    g = sns.pairplot(df[["age", "bmi", "charges", "smoker"]],
                     hue="smoker", palette="coolwarm",
                     plot_kws={"alpha": 0.5, "edgecolor": None},
                     diag_kws={"alpha": 0.5})
    g.figure.suptitle("Pairplot: age, bmi, charges (by smoker)", y=1.02)
    return g.figure


# Section 2.2 — Pearson correlation heatmap

def plot_correlation_heatmap(df):
    """Heatmap of Pearson correlations (encodes categoricals internally)."""
    df_enc = df.copy()
    df_enc["sex"] = df_enc["sex"].map({"female": 0, "male": 1})
    df_enc["smoker"] = df_enc["smoker"].map({"no": 0, "yes": 1})
    df_enc["region"] = df_enc["region"].map({
        "northeast": 1, "northwest": 2, "southeast": 3, "southwest": 4
    })

    # Drop non-numeric columns that may have been added elsewhere
    df_enc = df_enc.select_dtypes(include="number")

    corr = df_enc.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Pearson Correlation Matrix")
    return fig


# Section 2.3 — Smoker vs non-smoker charge KDE

def plot_smoker_charge_kde(df):
    """Overlapping KDE of charges for smoker vs non-smoker."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, group in df.groupby("smoker"):
        sns.kdeplot(group["charges"], label=label, fill=True, alpha=0.4, ax=ax)

    ax.set_title("Charge Distribution by Smoker Status")
    ax.set_xlabel("Charges")
    ax.legend(title="Smoker")
    return fig
