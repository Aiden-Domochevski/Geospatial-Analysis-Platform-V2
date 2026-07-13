import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Setup directories
data_path_scored = "../data/inputs/branches_scored.csv"
data_path_perf = "../data/inputs/branch_performance.csv"
output_dir = "../documentation/figures/"
os.makedirs(output_dir, exist_ok=True)

# Load Data
df_scored = pd.read_csv(data_path_scored)
df_perf = pd.read_csv(data_path_perf)

# Merge datasets on 'name' to get features and targets together
df = pd.merge(df_scored, df_perf, on="name", how="inner")

# Rename 'monthly_revenue_mad' to 'revenue_actual' for consistency
df.rename(columns={'monthly_revenue_mad': 'revenue_actual'}, inplace=True)


# Ensure columns are numeric
features = ["score_commercial", "score_office", "score_residential", 
            "score_accessibility", "score_competition", "score_cannibalization"]
for c in features + ["revenue_actual"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=features + ["revenue_actual"])

X = df[features]
y_reg = df["revenue_actual"]

# Create a simplified Classification Target (Action) based on revenue thresholds
def create_class(row):
    r = row["revenue_actual"]
    if r > 11000000:
        return "KEEP"
    elif r < 6000000:
        return "CLOSE"
    else:
        return "REVIEW"

df["action_class"] = df.apply(create_class, axis=1)
y_clf = df["action_class"]

# Train/Test Split & Scaling
X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

metrics_dict = {"regression": {}, "classification": {}}

# ==========================================
# 1. REGRESSION BENCHMARKS
# ==========================================
reg_models = {
    "OLS": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
}

plt.figure(figsize=(15, 10))
for i, (name, model) in enumerate(reg_models.items(), 1):
    model.fit(X_train_s, yr_train)
    preds = model.predict(X_test_s)
    
    rmse = np.sqrt(mean_squared_error(yr_test, preds))
    mae = mean_absolute_error(yr_test, preds)
    r2 = r2_score(yr_test, preds)
    
    metrics_dict["regression"][name] = {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4)
    }
    
    # Plot Actual vs Predicted
    plt.subplot(2, 2, i)
    plt.scatter(yr_test, preds, alpha=0.6, color='blue')
    plt.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], 'r--', lw=2)
    plt.title(f"{name} (R2={r2:.2f})")
    plt.xlabel("Revenu Réel")
    plt.ylabel("Revenu Prédit")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "regression_benchmark.png"), dpi=300)
plt.close()

# ==========================================
# 2. CLASSIFICATION BENCHMARKS
# ==========================================
# Convert string labels to integers for XGBoost
label_map = {"CLOSE": 0, "KEEP": 1, "REVIEW": 2}
inv_label_map = {v: k for k, v in label_map.items()}
yc_train_enc = yc_train.map(label_map)
yc_test_enc = yc_test.map(label_map)

clf_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBClassifier": xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
}

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, (name, model) in enumerate(clf_models.items()):
    model.fit(X_train_s, yc_train_enc)
    preds_enc = model.predict(X_test_s)
    
    acc = accuracy_score(yc_test_enc, preds_enc)
    cm = confusion_matrix(yc_test_enc, preds_enc)
    
    metrics_dict["classification"][name] = {
        "Accuracy": round(acc, 4)
    }
    
    # Plot Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=["CLOSE", "KEEP", "REVIEW"],
                yticklabels=["CLOSE", "KEEP", "REVIEW"])
    axes[i].set_title(f"{name} Confusion Matrix (Acc={acc:.2f})")
    axes[i].set_xlabel("Prédit")
    axes[i].set_ylabel("Réel")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "classification_benchmark.png"), dpi=300)
plt.close()

# Feature Importance Plot (XGBoost)
plt.figure(figsize=(10, 6))
xgb_model = reg_models["XGBoost"]
importance = xgb_model.feature_importances_
sns.barplot(x=importance, y=features, palette="viridis")
plt.title("XGBoost - Importance des Variables Spatiales")
plt.xlabel("Score d'Importance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "xgboost_importance.png"), dpi=300)
plt.close()

# Save metrics
with open("metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)

print("Benchmarks completed. Plots saved to figures/ and metrics saved to metrics.json.")
