import json

text_intro = """# ATTIJARIWAFA — SITE SELECTION WITH POPULATION DENSITY + MODEL BENCHMARKING

This notebook benchmarks clustering, regression, and classification algorithms on the bank branch dataset, incorporating the new **Population Density** metric.

Our goals:
1. **Clustering Benchmarking**: Compare K-Means, K-Medoids, Agglomerative, and GMM.
2. **Regression Benchmarking**: Predict `monthly_revenue_mad` using KNN, Ridge, Random Forest, Gradient Boosting, and XGBoost.
3. **Classification Benchmarking**: Predict if a branch is a 'High Performer' using Logistic Regression, Random Forest Classifier, and XGBoost Classifier to output a Confusion Matrix and Accuracy.
"""

code_imports = """import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             accuracy_score, confusion_matrix, classification_report)
from scipy.spatial import cKDTree
from xgboost import XGBRegressor, XGBClassifier
"""

text_data = """## 1. Data Loading & Preparation
We will load `branches_scored.csv`, `branch_performance.csv`, and `density.csv`.
"""

code_data = """# Load branch data
scores = pd.read_csv("../data/inputs/branches_scored.csv")
perf = pd.read_csv("../data/inputs/branch_performance.csv")
scores.columns = scores.columns.str.strip().str.lower()
perf.columns = perf.columns.str.strip().str.lower()

overlap = [c for c in perf.columns if c in scores.columns and c != "name"]
if overlap:
    perf = perf.drop(columns=overlap)

df = scores.merge(perf, on="name", how="inner")
df = df.drop_duplicates(subset=["name"]).reset_index(drop=True)

target_revenue = "monthly_revenue_mad"
score_cols = ["score_commercial", "score_office", "score_residential",
              "score_accessibility", "score_competition", "score_cannibalization"]

for col in score_cols:
    if col not in df.columns:
        df[col] = 0

# Load density data
try:
    pop_raw = pd.read_csv("../data/inputs/density.csv")
    pop_raw = pop_raw[["city", "lat", "lng", "population"]].copy()
    pop_raw.columns = ["city", "lat", "lon", "population"]
    pop_raw["population"] = pd.to_numeric(pop_raw["population"], errors="coerce").fillna(0).clip(lower=0)
    
    pop_coords = pop_raw[["lat", "lon"]].values
    pop_tree = cKDTree(pop_coords)
    pop_z = pop_raw["population"].values

    def get_population(lat, lon):
        _, idx = pop_tree.query([lat, lon])
        return float(pop_z[idx])

    df["population_density"] = [get_population(r["latitude"], r["longitude"]) for _, r in df.iterrows()]
except Exception as e:
    print("Warning: density.csv not found or unreadable. Using zeros.", e)
    df["population_density"] = 0.0

numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
df_clean = df[numeric_cols].copy()
df_clean["name"] = df["name"]
df_clean = df_clean.dropna()

print(f"Data shape after cleaning: {df_clean.shape}")
"""

text_clustering = """## 2. Clustering Benchmarking
We compare clustering techniques to segment branches into 8 clusters.
"""

code_clustering = """X_cluster = df_clean[score_cols].values
X_cluster_sc = StandardScaler().fit_transform(X_cluster)
N_CLUSTERS = 6

algorithms = {
    "K-Means": KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit_predict(X_cluster_sc),
    "Agglomerative": AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward").fit_predict(X_cluster_sc),
    "GMM": GaussianMixture(n_components=N_CLUSTERS, random_state=42, n_init=5).fit_predict(X_cluster_sc),
}

print(f"{'Algorithm':<15} | {'Silhouette':>10} | {'Davies-Bouldin':>15} | {'Calinski-Harabasz':>20}")
print("-" * 70)
for name, labels in algorithms.items():
    sil = silhouette_score(X_cluster_sc, labels)
    db = davies_bouldin_score(X_cluster_sc, labels)
    ch = calinski_harabasz_score(X_cluster_sc, labels)
    print(f"{name:<15} | {sil:>10.4f} | {db:>15.4f} | {ch:>20.2f}")

best_labels = algorithms["K-Means"]
df_clean["location_cluster"] = best_labels
"""

text_regression = """## 3. Regression Benchmarking (Revenue Prediction)
We predict `monthly_revenue_mad` based on score features + population density.
"""

code_regression = """features = score_cols + ["population_density"]
X_reg = df_clean[features].values
y_reg = df_clean[target_revenue].values

# Remove extreme outliers for training
q_lo, q_hi = np.percentile(y_reg, [1, 99])
keep = (y_reg >= q_lo) & (y_reg <= q_hi)
X_ml = X_reg[keep]
y_ml = y_reg[keep]

scaler = StandardScaler()
X_sc_ml = scaler.fit_transform(X_ml)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models_reg = {
    "Ridge": RidgeCV(alphas=[0.1, 1.0, 10.0]),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1, verbosity=0)
}

print(f"{'Model':<18} | {'CV R2':>8} | {'RMSE':>10} | {'MAE':>10}")
print("-" * 55)

best_r2 = -float("inf")
best_model_name = None

for name, model in models_reg.items():
    cv_r2 = cross_val_score(model, X_sc_ml, y_ml, cv=kf, scoring="r2").mean()
    cv_rmse = np.sqrt(-cross_val_score(model, X_sc_ml, y_ml, cv=kf, scoring="neg_mean_squared_error").mean())
    cv_mae = -cross_val_score(model, X_sc_ml, y_ml, cv=kf, scoring="neg_mean_absolute_error").mean()
    print(f"{name:<18} | {cv_r2:>8.4f} | {cv_rmse:>10.0f} | {cv_mae:>10.0f}")
    if cv_r2 > best_r2:
        best_r2 = cv_r2
        best_model_name = name

print(f"\\nBest Model: {best_model_name} with R2: {best_r2:.4f}")
"""

text_classification = """## 4. Classification Benchmarking
To provide a confusion matrix, we define a classification task: Is the branch a 'High Performer' (Revenue > Median)?
"""

code_classification = """# Classification: use mean revenue as threshold for balanced classes
threshold_rev = np.mean(y_reg)
y_cls = (y_reg > threshold_rev).astype(int)
X_cls = scaler.transform(X_reg)

print(f"Classification target: High Performer (revenue > {threshold_rev:,.0f} MAD)")
print(f"Class distribution: {dict(zip(*np.unique(y_cls, return_counts=True)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

models_cls = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost Classifier": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

for name, model in models_cls.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(classification_report(y_test, y_pred, target_names=["Low Performer", "High Performer"]))
"""

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(True)
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True)
    }

nb = {
    "cells": [
        md_cell(text_intro),
        code_cell(code_imports),
        md_cell(text_data),
        code_cell(code_data),
        md_cell(text_clustering),
        code_cell(code_clustering),
        md_cell(text_regression),
        code_cell(code_regression),
        md_cell(text_classification),
        code_cell(code_classification)
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('../notebooks/model_benchmarking.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
print("Jupyter Notebook '../notebooks/model_benchmarking.ipynb' generated successfully.")

