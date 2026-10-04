# ============================================================
#  CodeAlpha Internship - Task 1: Iris Dataset EDA
#  Author : Kiro
#  Description: Exploratory Data Analysis and Visualization
#               of the classic Iris flower dataset.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
import os

warnings.filterwarnings("ignore")

# ── Output folder for saved plots ──────────────────────────
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Seaborn theme ──────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
PALETTE = sns.color_palette("Set2", 3)

# ===========================================================
# 1. LOAD DATA
# ===========================================================
print("=" * 60)
print("  IRIS DATASET — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

raw = load_iris()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["species"] = pd.Categorical.from_codes(raw.target, raw.target_names)

print(f"\n✔  Dataset loaded  →  {df.shape[0]} rows × {df.shape[1]} columns")

# ===========================================================
# 2. BASIC OVERVIEW
# ===========================================================
print("\n─── First 5 rows ───────────────────────────────────────")
print(df.head())

print("\n─── Dataset Info ───────────────────────────────────────")
print(df.dtypes.to_string())
print(f"\nMissing values: {df.isnull().sum().sum()}")

print("\n─── Class Distribution ─────────────────────────────────")
print(df["species"].value_counts().to_string())

# ===========================================================
# 3. DESCRIPTIVE STATISTICS
# ===========================================================
print("\n─── Descriptive Statistics (all species) ───────────────")
print(df.describe().round(2).to_string())

print("\n─── Mean per species ───────────────────────────────────")
print(df.groupby("species").mean().round(2).to_string())

# ===========================================================
# 4. VISUALIZATIONS
# ===========================================================

features = raw.feature_names          # 4 numeric columns
species  = raw.target_names           # setosa, versicolor, virginica


# ── 4.1  Distribution plots (histogram + KDE) ─────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Feature Distributions by Species", fontsize=15, fontweight="bold")

for ax, feat in zip(axes.flatten(), features):
    for sp, color in zip(species, PALETTE):
        subset = df[df["species"] == sp][feat]
        ax.hist(subset, bins=15, alpha=0.45, color=color, edgecolor="white", density=True)
        subset.plot.kde(ax=ax, color=color, linewidth=2)
    ax.set_title(feat.replace(" (cm)", "").title())
    ax.set_xlabel("cm")
    ax.set_ylabel("Density")

handles = [mpatches.Patch(color=PALETTE[i], label=sp) for i, sp in enumerate(species)]
fig.legend(handles=handles, loc="upper right", title="Species", frameon=True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_feature_distributions.png", dpi=150)
plt.close()
print("\n✔  Saved: 01_feature_distributions.png")


# ── 4.2  Box plots ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Box Plots — Feature vs Species", fontsize=15, fontweight="bold")

for ax, feat in zip(axes.flatten(), features):
    sns.boxplot(data=df, x="species", y=feat, palette="Set2", ax=ax,
                width=0.5, linewidth=1.2)
    ax.set_title(feat.replace(" (cm)", "").title())
    ax.set_xlabel("")
    ax.set_ylabel("cm")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_boxplots.png", dpi=150)
plt.close()
print("✔  Saved: 02_boxplots.png")


# ── 4.3  Violin plots ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Violin Plots — Feature Distribution", fontsize=15, fontweight="bold")

for ax, feat in zip(axes.flatten(), features):
    sns.violinplot(data=df, x="species", y=feat, palette="Set2",
                   inner="quartile", ax=ax, linewidth=1.1)
    ax.set_title(feat.replace(" (cm)", "").title())
    ax.set_xlabel("")
    ax.set_ylabel("cm")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_violinplots.png", dpi=150)
plt.close()
print("✔  Saved: 03_violinplots.png")


# ── 4.4  Pair plot ─────────────────────────────────────────
pair_fig = sns.pairplot(df, hue="species", palette="Set2",
                        diag_kind="kde", plot_kws={"alpha": 0.6, "s": 40})
pair_fig.fig.suptitle("Pair Plot — All Feature Combinations", y=1.02,
                       fontsize=14, fontweight="bold")
pair_fig.savefig(f"{OUTPUT_DIR}/04_pairplot.png", dpi=150)
plt.close()
print("✔  Saved: 04_pairplot.png")


# ── 4.5  Correlation heat-map ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
corr = df[features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # upper triangle only
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heat-Map", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_correlation_heatmap.png", dpi=150)
plt.close()
print("✔  Saved: 05_correlation_heatmap.png")


# ── 4.6  Scatter — petal length vs petal width (most discriminative) ──
fig, ax = plt.subplots(figsize=(8, 6))
for sp, color in zip(species, PALETTE):
    sub = df[df["species"] == sp]
    ax.scatter(sub["petal length (cm)"], sub["petal width (cm)"],
               label=sp, color=color, alpha=0.75, s=60, edgecolors="white", linewidth=0.4)
ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.set_title("Petal Length vs Petal Width", fontsize=13, fontweight="bold")
ax.legend(title="Species")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_petal_scatter.png", dpi=150)
plt.close()
print("✔  Saved: 06_petal_scatter.png")


# ── 4.7  Scatter — sepal length vs sepal width ────────────
fig, ax = plt.subplots(figsize=(8, 6))
for sp, color in zip(species, PALETTE):
    sub = df[df["species"] == sp]
    ax.scatter(sub["sepal length (cm)"], sub["sepal width (cm)"],
               label=sp, color=color, alpha=0.75, s=60, edgecolors="white", linewidth=0.4)
ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Sepal Width (cm)")
ax.set_title("Sepal Length vs Sepal Width", fontsize=13, fontweight="bold")
ax.legend(title="Species")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_sepal_scatter.png", dpi=150)
plt.close()
print("✔  Saved: 07_sepal_scatter.png")


# ── 4.8  Class distribution bar chart ─────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["species"].value_counts()
bars = ax.bar(counts.index, counts.values, color=PALETTE, edgecolor="white", width=0.5)
ax.bar_label(bars, fmt="%d", padding=3, fontsize=11)
ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Species")
ax.set_ylabel("Count")
ax.set_ylim(0, counts.max() + 10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_class_distribution.png", dpi=150)
plt.close()
print("✔  Saved: 08_class_distribution.png")


# ===========================================================
# 5. KEY INSIGHTS SUMMARY
# ===========================================================
print("\n" + "=" * 60)
print("  KEY INSIGHTS")
print("=" * 60)
print("""
1. Dataset       : 150 samples, 4 features, 3 classes (50 each).
2. No missing values detected.

3. Petal features are the most discriminative:
   - Setosa has clearly smaller petals.
   - Versicolor and Virginica overlap slightly.

4. Sepal width shows the least separation between classes.

5. Strong positive correlation (≈0.96) between
   petal length and petal width.

6. Sepal length and petal length are moderately
   positively correlated (≈0.87).

7. Sepal width is slightly negatively correlated
   with the other three features.
""")
print(f"All plots saved to: ./{OUTPUT_DIR}/")
print("=" * 60)
