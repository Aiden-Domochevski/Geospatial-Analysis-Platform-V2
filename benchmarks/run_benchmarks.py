import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, silhouette_score, davies_bouldin_score, calinski_harabasz_score, accuracy_score, confusion_matrix, classification_report
from scipy.spatial import cKDTree
from xgboost import XGBRegressor, XGBClassifier
print('=' * 60)
print('STEP 1 — Loading data')
print('=' * 60)
scores = pd.read_csv('branches_scored.csv')
perf = pd.read_csv('branch_performance.csv')
scores.columns = scores.columns.str.strip().str.lower()
perf.columns = perf.columns.str.strip().str.lower()
overlap = [c for c in perf.columns if c in scores.columns and c != 'name']
if overlap:
    perf = perf.drop(columns=overlap)
df = scores.merge(perf, on='name', how='inner')
df = df.drop_duplicates(subset=['name']).reset_index(drop=True)
target_revenue = 'monthly_revenue_mad'
score_cols = ['score_commercial', 'score_office', 'score_residential', 'score_accessibility', 'score_competition', 'score_cannibalization']
for col in score_cols:
    if col not in df.columns:
        df[col] = 0
pop_raw = pd.read_csv('density.csv')
pop_raw = pop_raw[['city', 'lat', 'lng', 'population']].copy()
pop_raw.columns = ['city', 'lat', 'lon', 'population']
pop_raw['population'] = pd.to_numeric(pop_raw['population'], errors='coerce').fillna(0).clip(lower=0)
pop_coords = pop_raw[['lat', 'lon']].values
pop_tree = cKDTree(pop_coords)
pop_z = pop_raw['population'].values

def get_population(lat, lon):
    _, idx = pop_tree.query([lat, lon])
    return float(pop_z[idx])
df['population_density'] = [get_population(r['latitude'], r['longitude']) for _, r in df.iterrows()]
print(f'  Branches loaded: {len(df)}')
print(f"  Population density — mean: {df['population_density'].mean():,.0f}")
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
df_clean = df[numeric_cols].copy()
df_clean['name'] = df['name']
df_clean = df_clean.dropna()
print(f'  Clean data shape: {df_clean.shape}')
print('\n' + '=' * 60)
print('STEP 2 — Clustering benchmarking')
print('=' * 60)
X_cluster = df_clean[score_cols].values
X_cluster_sc = StandardScaler().fit_transform(X_cluster)
N_CLUSTERS = 8
clustering_algorithms = {'K-Means': KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit_predict(X_cluster_sc), 'Agglomerative': AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward').fit_predict(X_cluster_sc), 'GMM': GaussianMixture(n_components=N_CLUSTERS, random_state=42, n_init=5).fit_predict(X_cluster_sc)}
clustering_results = []
print(f"  {'Algorithm':<18} {'Silhouette':>12} {'Davies-Bouldin':>16} {'Calinski-Harabasz':>18}")
print('  ' + '-' * 68)
for algo_name, labels in clustering_algorithms.items():
    sil = silhouette_score(X_cluster_sc, labels)
    db = davies_bouldin_score(X_cluster_sc, labels)
    ch = calinski_harabasz_score(X_cluster_sc, labels)
    clustering_results.append({'algorithm': algo_name, 'silhouette_score': round(sil, 4), 'davies_bouldin': round(db, 4), 'calinski_harabasz': round(ch, 2)})
    print(f'  {algo_name:<18} {sil:>12.4f} {db:>16.4f} {ch:>18.2f}')
sorted_sil = sorted(clustering_results, key=lambda x: x['silhouette_score'], reverse=True)
sorted_db = sorted(clustering_results, key=lambda x: x['davies_bouldin'], reverse=False)
sorted_ch = sorted(clustering_results, key=lambda x: x['calinski_harabasz'], reverse=True)
for rank, r in enumerate(sorted_sil):
    r['rank_sil'] = rank
for rank, r in enumerate(sorted_db):
    r['rank_db'] = rank
for rank, r in enumerate(sorted_ch):
    r['rank_ch'] = rank
for r in clustering_results:
    r['total_rank'] = r.pop('rank_sil') + r.pop('rank_db') + r.pop('rank_ch')
best_clustering = min(clustering_results, key=lambda x: x['total_rank'])
best_cluster_name = best_clustering['algorithm']
for r in clustering_results:
    r['is_best'] = r['algorithm'] == best_cluster_name
print(f"\n  BEST CLUSTERING: {best_cluster_name} (total rank = {best_clustering['total_rank']})")
df_clean['location_cluster'] = clustering_algorithms[best_cluster_name]
unique_labels = np.unique(df_clean['location_cluster'])
benchmark_rows = []
for cluster_id in unique_labels:
    cluster_df = df_clean[df_clean['location_cluster'] == cluster_id]
    if len(cluster_df) == 0:
        continue
    n_top = max(1, int(len(cluster_df) * 0.1))
    top_branches = cluster_df.nlargest(n_top, target_revenue)
    for _, row in top_branches.iterrows():
        entry = {'cluster': cluster_id, 'revenue': row[target_revenue]}
        for col in score_cols:
            entry[col] = row[col]
        benchmark_rows.append(entry)
benchmark_df = pd.DataFrame(benchmark_rows)
print(f'  Benchmark set: {len(benchmark_df)} branches (top 10% per cluster)')
print('\n' + '=' * 60)
print('STEP 4 — Regression benchmarking (5-fold CV)')
print('=' * 60)
features = score_cols + ['population_density']
X_reg = df_clean[features].values
y_reg = df_clean[target_revenue].values
q_lo, q_hi = np.percentile(y_reg, [1, 99])
keep = (y_reg >= q_lo) & (y_reg <= q_hi)
X_ml = X_reg[keep]
y_ml = y_reg[keep]
scaler = StandardScaler()
X_sc_ml = scaler.fit_transform(X_ml)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def eval_model(name, model, X, y, kfold):
    cv_r2 = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    cv_mae = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error'))
    mape_scores = []
    for tr, va in kfold.split(X):
        model.fit(X[tr], y[tr])
        mape_scores.append(mape(y[va], model.predict(X[va])))
    model.fit(X, y)
    return {'model': name, 'cv_r2_mean': round(float(cv_r2.mean()), 4), 'cv_r2_std': round(float(cv_r2.std()), 4), 'cv_mae_mean': round(float(cv_mae.mean()), 0), 'cv_mae_std': round(float(cv_mae.std()), 0), 'cv_rmse_mean': round(float(cv_rmse.mean()), 0), 'cv_rmse_std': round(float(cv_rmse.std()), 0), 'cv_mape_mean': round(float(np.mean(mape_scores)), 2), 'cv_mape_std': round(float(np.std(mape_scores)), 2)}

def cosine_knn_predict(Xtr, ytr, Xte, k=5):
    ntr = np.linalg.norm(Xtr, axis=1, keepdims=True)
    ntr = np.where(ntr < 1e-10, 1.0, ntr)
    Xtr_n = Xtr / ntr
    nte = np.linalg.norm(Xte, axis=1, keepdims=True)
    nte = np.where(nte < 1e-10, 1.0, nte)
    Xte_n = Xte / nte
    sims = np.clip(np.dot(Xte_n, Xtr_n.T), 0, 1)
    top = np.argpartition(sims, -k, axis=1)[:, -k:]
    tsim = np.take_along_axis(sims, top, axis=1)
    trev = ytr[top]
    w = tsim ** 2
    ws = np.where(w.sum(axis=1, keepdims=True) > 1e-10, w.sum(axis=1, keepdims=True), 1.0)
    return (w * trev).sum(axis=1) / ws.flatten()
knn_r2, knn_mae, knn_rmse, knn_mape = ([], [], [], [])
for tr, va in kf.split(X_sc_ml):
    p = cosine_knn_predict(X_sc_ml[tr], y_ml[tr], X_sc_ml[va])
    knn_r2.append(r2_score(y_ml[va], p))
    knn_mae.append(mean_absolute_error(y_ml[va], p))
    knn_rmse.append(np.sqrt(mean_squared_error(y_ml[va], p)))
    knn_mape.append(mape(y_ml[va], p))
knn_result = {'model': 'KNN Cosine (baseline)', 'cv_r2_mean': round(float(np.mean(knn_r2)), 4), 'cv_r2_std': round(float(np.std(knn_r2)), 4), 'cv_mae_mean': round(float(np.mean(knn_mae)), 0), 'cv_mae_std': round(float(np.std(knn_mae)), 0), 'cv_rmse_mean': round(float(np.mean(knn_rmse)), 0), 'cv_rmse_std': round(float(np.std(knn_rmse)), 0), 'cv_mape_mean': round(float(np.mean(knn_mape)), 2), 'cv_mape_std': round(float(np.std(knn_mape)), 2)}
sklearn_models = [('Ridge Regression', RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=3)), ('Random Forest', RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=8, max_features=0.7, random_state=42, n_jobs=-1)), ('Gradient Boosting', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, min_samples_leaf=8, subsample=0.8, random_state=42)), ('XGBoost', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=-1))]
reg_results = [knn_result]
print(f"  {'Model':<26} {'CV-R2':>8} {'(+/-)':>7} {'MAE':>12} {'RMSE':>12} {'MAPE (%)':>10}")
print('  ' + '-' * 80)
for mname, model in sklearn_models:
    print(f'  Training [{mname}]...')
    res = eval_model(mname, model, X_sc_ml, y_ml, kf)
    reg_results.append(res)
    print(f"    CV-R2 : {res['cv_r2_mean']:.4f} +/- {res['cv_r2_std']:.4f}")
    print(f"    MAE   : {res['cv_mae_mean']:,.0f}  |  RMSE: {res['cv_rmse_mean']:,.0f}  |  MAPE: {res['cv_mape_mean']:.2f}%")
best_reg = max(reg_results, key=lambda r: r['cv_r2_mean'])
best_model_name = best_reg['model']
for r in reg_results:
    r['is_best'] = r['model'] == best_model_name
print(f"\n  BEST REGRESSION MODEL: {best_model_name} (CV-R2 = {best_reg['cv_r2_mean']:.4f})")
print(f"\n  {'Model':<26} {'CV-R2':>8} {'(+/-)':>7} {'MAE':>12} {'RMSE':>12} {'MAPE':>10}")
print('  ' + '-' * 80)
for r in reg_results:
    tag = ' <-- BEST' if r['is_best'] else ''
    print(f"  {r['model']:<26} {r['cv_r2_mean']:>8.4f} {r['cv_r2_std']:>7.4f} {r['cv_mae_mean']:>12,.0f} {r['cv_rmse_mean']:>12,.0f} {r['cv_mape_mean']:>9.2f}%{tag}")
print('\n' + '=' * 60)
print('STEP 5 — Classification benchmarking')
print('=' * 60)
threshold_rev = np.mean(y_reg)
y_cls = (y_reg > threshold_rev).astype(int)
X_cls = scaler.transform(X_reg)
print(f'  Threshold: High Performer = revenue > {threshold_rev:,.0f} MAD')
print(f'  Class distribution: {dict(zip(*np.unique(y_cls, return_counts=True)))}')
cls_metrics = []
if len(np.unique(y_cls)) < 2:
    print('  ERROR: Only one class — skipping classification.')
else:
    X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    models_cls = [('Logistic Regression', LogisticRegression(max_iter=1000)), ('Random Forest Classifier', RandomForestClassifier(n_estimators=100, random_state=42)), ('XGBoost Classifier', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'))]
    for name, model in models_cls:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Low Performer', 'High Performer'], output_dict=True)
        print(f'\n  --- {name} ---')
        print(f'  Accuracy: {acc:.4f}')
        print(f'  Confusion Matrix: {cm.tolist()}')
        print(classification_report(y_test, y_pred, target_names=['Low Performer', 'High Performer']))
        cls_metrics.append({'model': name, 'accuracy': round(acc, 4), 'confusion_matrix': cm.tolist(), 'precision_high': round(report['High Performer']['precision'], 4), 'recall_high': round(report['High Performer']['recall'], 4), 'f1_high': round(report['High Performer']['f1-score'], 4), 'precision_low': round(report['Low Performer']['precision'], 4), 'recall_low': round(report['Low Performer']['recall'], 4), 'f1_low': round(report['Low Performer']['f1-score'], 4)})
print('\n' + '=' * 60)
print('STEP 6 — Saving benchmarking_results.json')
print('=' * 60)
results = {'method': 'Cluster-Based Benchmark KNN + Model Benchmarking', 'best_revenue_model': best_model_name, 'best_clustering': best_cluster_name, 'features_used': features, 'clusters_used': N_CLUSTERS, 'benchmarks_used': len(benchmark_df), 'total_branches': len(df_clean), 'min_poi_required': 5, 'min_population': 10.0, 'score_weights': {'poi': 0.35, 'similarity': 0.35, 'population': 0.3}, 'clustering_benchmarking': clustering_results, 'regression_benchmarking': reg_results, 'classification_benchmarking': {'threshold_mad': round(float(threshold_rev), 0), 'models': cls_metrics}}
with open('benchmarking_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'  Saved to benchmarking_results.json')
print(f'  Best clustering : {best_cluster_name}')
print(f"  Best regression : {best_model_name} (R2={best_reg['cv_r2_mean']:.4f})")
print(f'  Done!')