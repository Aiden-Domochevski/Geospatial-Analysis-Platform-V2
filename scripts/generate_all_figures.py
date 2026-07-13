"""Generate additional figures for Chapter 4: BCG matrix, PCA clusters, dataset stats."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb
import json
import os
import warnings
warnings.filterwarnings('ignore')

output_dir = r"final_model\documentation\figures"
os.makedirs(output_dir, exist_ok=True)

# Load data
df_scored = pd.read_csv(r"final_model\data\inputs\branches_scored.csv")
df_perf = pd.read_csv(r"final_model\data\inputs\branch_performance.csv")
df = pd.merge(df_scored, df_perf, on="name", how="inner")

features = ["score_commercial", "score_office", "score_residential",
            "score_accessibility", "score_competition", "score_cannibalization"]

print(f"Dataset: {len(df)} branches, {len(df.columns)} columns")
print(f"Features used: {features}")

# ---- 1. PCA Cluster Visualization ----
X = StandardScaler().fit_transform(df[features].fillna(0))
km = KMeans(5, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(X)

cluster_means = df.groupby("cluster")["composite_score"].mean().sort_values(ascending=False)
rank_map = {cid: rank for rank, cid in enumerate(cluster_means.index)}
LABELS = {0: "High Performer", 1: "Commercial Hub", 2: "Residential Anchor",
          3: "Underperformer", 4: "At Risk / Redundant"}
df["cluster_label"] = df["cluster"].map(rank_map).map(LABELS)

pca = PCA(2, random_state=42)
coords = pca.fit_transform(X)
df["pca_x"] = coords[:, 0]
df["pca_y"] = coords[:, 1]

fig, ax = plt.subplots(figsize=(12, 8))
colors = ["#22C55E", "#3B82F6", "#A855F7", "#F97316", "#EF4444"]
for i, label in enumerate(LABELS.values()):
    sub = df[df["cluster_label"] == label]
    ax.scatter(sub["pca_x"], sub["pca_y"], c=colors[i], label=label, alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("Projection PCA 2D des 5 Clusters K-Means du Réseau Bancaire")
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pca_clusters.png"), dpi=300)
plt.close()
print("Saved: pca_clusters.png")
print(f"  PCA Variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

# ---- 2. Cluster Distribution ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cluster_counts = df["cluster_label"].value_counts()
axes[0].barh(cluster_counts.index, cluster_counts.values, color=colors[:len(cluster_counts)])
axes[0].set_xlabel("Nombre d'Agences")
axes[0].set_title("Répartition des Agences par Cluster")
axes[0].invert_yaxis()

cluster_scores = df.groupby("cluster_label")["composite_score"].mean().reindex(LABELS.values())
axes[1].barh(cluster_scores.index, cluster_scores.values, color=colors[:len(cluster_scores)])
axes[1].set_xlabel("Composite Score Moyen")
axes[1].set_title("Score Composite Moyen par Cluster")
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cluster_distribution.png"), dpi=300)
plt.close()
print("Saved: cluster_distribution.png")

# Print cluster stats
print("\nCluster Statistics:")
for label in LABELS.values():
    sub = df[df["cluster_label"] == label]
    print(f"  {label}: {len(sub)} branches, avg composite={sub['composite_score'].mean():.3f}")

# ---- 3. BCG Matrix ----
den = df[["name","latitude","longitude","composite_score","monthly_revenue_mad"]].dropna()
from scipy.spatial import cKDTree

# Spatial features for XGBoost prediction
X_bcg = StandardScaler().fit_transform(den[["composite_score"]].values)
y_bcg = den["monthly_revenue_mad"].values

model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_bcg, y_bcg)
den["predicted_revenue"] = model.predict(X_bcg)

med_pred = den["predicted_revenue"].median()
med_real = den["monthly_revenue_mad"].median()

def classify_bcg(row):
    pot = row["predicted_revenue"] >= med_pred
    perf = row["monthly_revenue_mad"] >= med_real
    if pot and perf: return "FLAGSHIP"
    elif pot and not perf: return "REVIEW / INVEST"
    elif not pot and perf: return "MONITOR"
    else: return "CLOSE"

den["bcg_quadrant"] = den.apply(classify_bcg, axis=1)

fig, ax = plt.subplots(figsize=(12, 9))
bcg_colors = {"FLAGSHIP": "#22C55E", "REVIEW / INVEST": "#F59E0B",
              "MONITOR": "#3B82F6", "CLOSE": "#EF4444"}
for quad, color in bcg_colors.items():
    sub = den[den["bcg_quadrant"] == quad]
    ax.scatter(sub["predicted_revenue"], sub["monthly_revenue_mad"],
               c=color, label=quad, alpha=0.65, s=35, edgecolors='white', linewidth=0.5)

# Median lines
ax.axvline(med_pred, color='gray', linestyle='--', alpha=0.6, label=f'Médiane Potentiel ({med_pred:,.0f})')
ax.axhline(med_real, color='gray', linestyle='-.', alpha=0.6, label=f'Médiane Revenu ({med_real:,.0f})')

# Diagonal y=x
lim_min = min(den["predicted_revenue"].min(), den["monthly_revenue_mad"].min())
lim_max = max(den["predicted_revenue"].max(), den["monthly_revenue_mad"].max())
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.4, linewidth=1.5, label="Ligne d'Espérance (y=x)")

ax.set_xlabel("Potentiel Spatial Prédit (XGBoost) — MAD", fontsize=11)
ax.set_ylabel("Revenu Mensuel Réel — MAD", fontsize=11)
ax.set_title("Matrice BCG Spatiale : Potentiel vs Performance", fontsize=13)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bcg_matrix.png"), dpi=300)
plt.close()
print("Saved: bcg_matrix.png")

# Print BCG stats
print("\nBCG Matrix Distribution:")
for quad in ["FLAGSHIP", "REVIEW / INVEST", "MONITOR", "CLOSE"]:
    count = len(den[den["bcg_quadrant"] == quad])
    print(f"  {quad}: {count} branches ({count/len(den)*100:.1f}%)")

# ---- 4. Composite Score Distribution ----
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["composite_score"], bins=40, color='#3B82F6', edgecolor='white', alpha=0.85)
ax.axvline(df["composite_score"].mean(), color='red', linestyle='--', label=f'Moyenne ({df["composite_score"].mean():.3f})')
ax.axvline(df["composite_score"].median(), color='orange', linestyle='--', label=f'Médiane ({df["composite_score"].median():.3f})')
ax.set_xlabel("Composite Score")
ax.set_ylabel("Fréquence")
ax.set_title("Distribution du Composite Score des Agences")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "composite_score_dist.png"), dpi=300)
plt.close()
print("Saved: composite_score_dist.png")

# ---- 5. Revenue Distribution ----
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["monthly_revenue_mad"], bins=40, color='#A855F7', edgecolor='white', alpha=0.85)
ax.axvline(df["monthly_revenue_mad"].mean(), color='red', linestyle='--', label=f'Moyenne ({df["monthly_revenue_mad"].mean():,.0f} MAD)')
ax.axvline(df["monthly_revenue_mad"].median(), color='orange', linestyle='--', label=f'Médiane ({df["monthly_revenue_mad"].median():,.0f} MAD)')
ax.set_xlabel("Revenu Mensuel (MAD)")
ax.set_ylabel("Fréquence")
ax.set_title("Distribution de la Variable Cible : Revenu Mensuel des Agences")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "revenue_distribution.png"), dpi=300)
plt.close()
print("Saved: revenue_distribution.png")

# ---- 6. Correlation Heatmap ----
corr_cols = features + ["composite_score", "monthly_revenue_mad"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            xticklabels=[c.replace('score_','').replace('_',' ').title() for c in corr_cols],
            yticklabels=[c.replace('score_','').replace('_',' ').title() for c in corr_cols])
ax.set_title("Matrice de Corrélation des Variables Spatiales et Financières")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
plt.close()
print("Saved: correlation_heatmap.png")

# ---- 7. Radar chart per cluster ----
from math import pi
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), subplot_kw=dict(polar=True))
feat_labels = ["Commercial", "Office", "Residential", "Access.", "Compet.", "Cannibal."]
angles = [n / float(len(feat_labels)) * 2 * pi for n in range(len(feat_labels))]
angles += angles[:1]

for idx, (label, color) in enumerate(zip(LABELS.values(), colors)):
    sub = df[df["cluster_label"] == label]
    vals = sub[features].mean().values.tolist()
    vals += vals[:1]
    axes[idx].fill(angles, vals, color=color, alpha=0.25)
    axes[idx].plot(angles, vals, color=color, linewidth=2)
    axes[idx].set_xticks(angles[:-1])
    axes[idx].set_xticklabels(feat_labels, size=7)
    axes[idx].set_title(label, size=9, pad=15)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cluster_radar.png"), dpi=300)
plt.close()
print("Saved: cluster_radar.png")

# ---- Save extended metrics ----
ext_metrics = {
    "dataset_size": len(df),
    "n_features": len(features),
    "composite_score_mean": round(df["composite_score"].mean(), 4),
    "composite_score_median": round(df["composite_score"].median(), 4),
    "composite_score_std": round(df["composite_score"].std(), 4),
    "revenue_mean": round(df["monthly_revenue_mad"].mean(), 2),
    "revenue_median": round(df["monthly_revenue_mad"].median(), 2),
    "pca_var_pc1": round(pca.explained_variance_ratio_[0]*100, 1),
    "pca_var_pc2": round(pca.explained_variance_ratio_[1]*100, 1),
    "bcg_flagship": len(den[den["bcg_quadrant"]=="FLAGSHIP"]),
    "bcg_review": len(den[den["bcg_quadrant"]=="REVIEW / INVEST"]),
    "bcg_monitor": len(den[den["bcg_quadrant"]=="MONITOR"]),
    "bcg_close": len(den[den["bcg_quadrant"]=="CLOSE"]),
}

with open(r"final_model\scripts\metrics_extended.json", "w") as f:
    json.dump(ext_metrics, f, indent=4)

print("\nExtended metrics saved.")
print(json.dumps(ext_metrics, indent=2))
print("\nAll figures generated successfully!")
