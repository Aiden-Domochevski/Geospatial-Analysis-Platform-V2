import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
AGENCIES_FILE = 'agencies_final.csv'
POI_FILE = 'merged_data_clean.csv'
OUTPUT_FILE = 'branches_scored.csv'
R_500M = 0.0045
R_1KM = 0.009
R_2KM = 0.018
R_3KM = 0.027
N_CLUSTERS = 5
COMMERCIAL_CATS = {'restaurant', 'cafe', 'coffee_shop', 'fast_food_restaurant', 'bakery', 'pizzeria', 'boulangerie', 'patisserie', 'sandwich_shop', 'casual_eatery', 'bar', 'pub', 'bar_and_grill', 'buffet_restaurant', 'breakfast_restaurant', 'juice_bar', 'food_or_beverage_store', 'eating_drinking_location', 'shop', 'retail_location', 'specialty_store', 'clothing_store', 'fashion_or_apparel_store', 'jewelry_store', 'shoe_store', 'gift_shop', 'electronics_store', 'boutique', 'magasin', 'souk', 'marche', 'mall', 'shopping_mall', 'supermarket', 'grocery_store', 'convenience_store', 'food_bank', 'wholesale', 'wholesaler', 'department_store', 'hardware_store', 'furniture', 'home_improvement_center'}
OFFICE_CATS = {'office', 'corporate_or_business_office', 'bureau', 'b2b_service', 'b2b_supplier_distributor', 'financial_service', 'insurance_agency', 'real_estate_agency', 'real_estate_service', 'attorney_or_law_firm', 'legal_service', 'accountant_or_bookkeeper', 'engineering_service', 'software_development', 'information_technology_service', 'business_management_service', 'consulting', 'company', 'ngo', 'advertising_agency', 'human_resource_service', 'travel_service', 'travel_agency', 'import_export_company', 'telecommunications_service', 'government_office', 'diplomatic', 'association', 'construction_company'}
RESIDENTIAL_CATS = {'mosque', 'muslim_place_of_worship', 'place_of_worship', 'church', 'pharmacy', 'school', 'elementary_school', 'high_school', 'middle_school', 'college_university', 'preschool', 'kindergarten', 'university', 'hospital', 'clinic_or_treatment_center', 'doctors_office', 'healthcare_location', 'dental_care', 'eye_care', 'supermarket', 'grocery_store', 'epicerie', 'hammam', 'salon', 'hair_salon', 'beauty_salon', 'barber_shop', 'coiffeur', 'laundromat', 'dry_cleaning_service', 'post_office', 'community_center', 'social_or_community_service', 'neighbourhood', 'suburb', 'residential'}
TRANSIT_CATS = {'bus_stop', 'bus_station', 'train_station', 'stop_position', 'taxi', 'taxi_service', 'taxi_or_ride_share_service', 'ferry_terminal', 'airport', 'aerodrome', 'transportation_location', 'parking', 'station', 'public_transport', 'tram_stop'}
COMPETITOR_BANK_KEYWORDS = ['bmce', 'bmci', 'cih', 'crédit agricole', 'credit agricole', 'banque populaire', 'société générale', 'societe generale', 'al barid', 'barid bank', 'cdm', 'crédor', 'wafacash', 'cash plus', 'western union', 'money transfer', 'atm', 'bureau_de_change', 'credit_union', 'loan_provider']
OWN_BANK_KEYWORDS = ['attijariwafa', 'attijari wafa', 'wafa', 'attijariwafa bank']
print('\n' + '═' * 60)
print('  STEP 1 — Loading data')
print('═' * 60)
agencies = pd.read_csv(AGENCIES_FILE)
agencies.columns = agencies.columns.str.strip().str.lower()
lat_col = next((c for c in agencies.columns if 'lat' in c), None)
lon_col = next((c for c in agencies.columns if 'lon' in c), None)
name_col = next((c for c in agencies.columns if 'name' in c), None)
agencies = agencies.rename(columns={lat_col: 'lat', lon_col: 'lon', name_col: 'name'})
agencies['lat'] = pd.to_numeric(agencies['lat'], errors='coerce')
agencies['lon'] = pd.to_numeric(agencies['lon'], errors='coerce')
agencies = agencies.dropna(subset=['lat', 'lon'])
print(f'   Agencies loaded: {len(agencies):,} Morocco branches')
poi = pd.read_csv(POI_FILE, low_memory=False)
poi.columns = poi.columns.str.strip().str.lower()
poi_lat = next((c for c in poi.columns if 'lat' in c), None)
poi_lon = next((c for c in poi.columns if 'lon' in c), None)
poi_name = next((c for c in poi.columns if 'name' in c), None)
poi_cat = next((c for c in poi.columns if 'cat' in c), None)
poi = poi.rename(columns={poi_lat: 'lat', poi_lon: 'lon', poi_name: 'name', poi_cat: 'category'})
poi['lat'] = pd.to_numeric(poi['lat'], errors='coerce')
poi['lon'] = pd.to_numeric(poi['lon'], errors='coerce')
poi['category'] = poi['category'].fillna('').astype(str).str.lower().str.strip()
poi = poi.dropna(subset=['lat', 'lon']).reset_index(drop=True)
print(f'   POIs loaded: {len(poi):,} points of interest')
print('\n' + '═' * 60)
print('  STEP 2 — Building spatial indices')
print('═' * 60)
poi_coords = poi[['lat', 'lon']].values
agency_coords = agencies[['lat', 'lon']].values

def mask_for_cats(cat_set):
    return poi['category'].isin(cat_set)

def mask_for_keywords(keywords):
    pattern = '|'.join(keywords)
    return poi['category'].str.contains(pattern, na=False) | poi['name'].str.lower().str.contains(pattern, na=False)
commercial_poi = poi[mask_for_cats(COMMERCIAL_CATS)][['lat', 'lon']].values
office_poi = poi[mask_for_cats(OFFICE_CATS)][['lat', 'lon']].values
residential_poi = poi[mask_for_cats(RESIDENTIAL_CATS)][['lat', 'lon']].values
transit_poi = poi[mask_for_cats(TRANSIT_CATS)][['lat', 'lon']].values
competitor_poi = poi[mask_for_keywords(COMPETITOR_BANK_KEYWORDS)][['lat', 'lon']].values
all_poi_coords = poi_coords
tree_all = cKDTree(all_poi_coords) if len(all_poi_coords) > 0 else None
tree_commercial = cKDTree(commercial_poi) if len(commercial_poi) > 0 else None
tree_office = cKDTree(office_poi) if len(office_poi) > 0 else None
tree_residential = cKDTree(residential_poi) if len(residential_poi) > 0 else None
tree_transit = cKDTree(transit_poi) if len(transit_poi) > 0 else None
tree_competitor = cKDTree(competitor_poi) if len(competitor_poi) > 0 else None
tree_agency = cKDTree(agency_coords)
print(f'  Commercial POIs : {len(commercial_poi):,}')
print(f'  Office POIs     : {len(office_poi):,}')
print(f'  Residential POIs: {len(residential_poi):,}')
print(f'  Transit POIs    : {len(transit_poi):,}')
print(f'  Competitor POIs : {len(competitor_poi):,}')

def count_within(tree, point, radius):
    if tree is None:
        return 0
    return len(tree.query_ball_point(point, radius))
print('\n' + '═' * 60)
print('  STEP 3 — Scoring all branches')
print('═' * 60)
scores = []
total = len(agencies)
for i, row in agencies.iterrows():
    if i % 100 == 0:
        print(f'  Processing {i}/{total}...', end='\r')
    pt = [row['lat'], row['lon']]
    commercial_1km = count_within(tree_commercial, pt, R_1KM)
    commercial_500m = count_within(tree_commercial, pt, R_500M)
    office_1km = count_within(tree_office, pt, R_1KM)
    residential_1km = count_within(tree_residential, pt, R_1KM)
    transit_500m = count_within(tree_transit, pt, R_500M)
    transit_1km = count_within(tree_transit, pt, R_1KM)
    competitor_1km = count_within(tree_competitor, pt, R_1KM)
    competitor_500m = count_within(tree_competitor, pt, R_500M)
    total_poi_1km = count_within(tree_all, pt, R_1KM)
    total_poi_500m = count_within(tree_all, pt, R_500M)
    own_2km = count_within(tree_agency, pt, R_2KM) - 1
    own_1km = count_within(tree_agency, pt, R_1KM) - 1
    own_2km = max(own_2km, 0)
    own_1km = max(own_1km, 0)
    commercial_density = min(commercial_1km / 50, 1.0)
    office_density = min(office_1km / 30, 1.0)
    residential_density = min(residential_1km / 40, 1.0)
    accessibility = min(transit_1km / 10, 1.0)
    competition = min(competitor_1km / 8, 1.0)
    cannibalization = min(own_2km / 3, 1.0)
    composite = 0.3 * commercial_density + 0.2 * office_density + 0.2 * residential_density + 0.15 * accessibility - 0.1 * competition - 0.05 * cannibalization
    composite = round(max(composite, 0.0), 4)
    scores.append({'name': row['name'], 'latitude': row['lat'], 'longitude': row['lon'], 'commercial_500m': commercial_500m, 'commercial_1km': commercial_1km, 'office_1km': office_1km, 'residential_1km': residential_1km, 'transit_500m': transit_500m, 'transit_1km': transit_1km, 'competitor_500m': competitor_500m, 'competitor_1km': competitor_1km, 'total_poi_500m': total_poi_500m, 'total_poi_1km': total_poi_1km, 'own_branches_1km': own_1km, 'own_branches_2km': own_2km, 'score_commercial': round(commercial_density, 4), 'score_office': round(office_density, 4), 'score_residential': round(residential_density, 4), 'score_accessibility': round(accessibility, 4), 'score_competition': round(competition, 4), 'score_cannibalization': round(cannibalization, 4), 'composite_score': composite})
df = pd.DataFrame(scores)
print(f'\n   Scored {len(df):,} branches')
print('\n' + '═' * 60)
print('  STEP 4 — KMeans clustering (profiling branch types)')
print('═' * 60)
feature_cols = ['score_commercial', 'score_office', 'score_residential', 'score_accessibility', 'score_competition', 'score_cannibalization']
X = df[feature_cols].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)
cluster_means = df.groupby('kmeans_cluster')['composite_score'].mean().sort_values(ascending=False)
rank_map = {cluster_id: rank for rank, cluster_id in enumerate(cluster_means.index)}
CLUSTER_LABELS = {0: '🟢 High Performer', 1: '🔵 Commercial Hub', 2: '🟡 Residential Anchor', 3: '🟠 Underperformer', 4: '🔴 Redundant / At Risk'}
df['cluster_rank'] = df['kmeans_cluster'].map(rank_map)
df['cluster_label'] = df['cluster_rank'].map(CLUSTER_LABELS)
print('  Cluster distribution:')
print(df['cluster_label'].value_counts().to_string())
print('\n' + '═' * 60)
print('  STEP 5 — DBSCAN (detecting overcrowded branch zones)')
print('═' * 60)
geo_coords = df[['latitude', 'longitude']].values
dbscan = DBSCAN(eps=R_2KM, min_samples=2, algorithm='ball_tree', metric='euclidean')
df['dbscan_zone'] = dbscan.fit_predict(geo_coords)
n_overcrowded_zones = df[df['dbscan_zone'] >= 0]['dbscan_zone'].nunique()
n_isolated = (df['dbscan_zone'] == -1).sum()
n_in_zones = (df['dbscan_zone'] >= 0).sum()
print(f'  Isolated branches (healthy spacing): {n_isolated}')
print(f'  Overcrowded zones detected:          {n_overcrowded_zones}')
print(f'  Branches inside overcrowded zones:   {n_in_zones}')
df['is_overcrowded'] = df['dbscan_zone'] >= 0
print('\n' + '═' * 60)
print('  STEP 6 — Final action classification')
print('═' * 60)

def classify(row):
    score = row['composite_score']
    crowd = row['is_overcrowded']
    canib = row['score_cannibalization']
    comp = row['score_competition']
    if score >= 0.35 and (not crowd):
        return 'KEEP — Invest'
    elif score >= 0.35 and crowd:
        return 'KEEP — Flagship (merge nearby)'
    elif score >= 0.2 and (not crowd):
        return 'MONITOR — Growth Potential'
    elif score >= 0.2 and crowd:
        return 'REVIEW — Merger Candidate'
    elif score < 0.2 and crowd:
        return 'CLOSE — Redundant'
    elif score < 0.2 and comp >= 0.5:
        return 'REPOSITION — High Competition'
    else:
        return 'CLOSE — Underperformer'
df['action'] = df.apply(classify, axis=1)
print('  Action classification:')
print(df['action'].value_counts().to_string())
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(X_scaled)
df['pca_x'] = pca_result[:, 0].round(4)
df['pca_y'] = pca_result[:, 1].round(4)
print('\n' + '═' * 60)
print('  STEP 7 — Saving output')
print('═' * 60)
df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
df['rank'] = df.index + 1
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"\n{'═' * 60}\n   PIPELINE COMPLETE\n{'═' * 60}\n\n  Total branches scored    : {len(df):,}\n  Output file              : {OUTPUT_FILE}\n\n  COMPOSITE SCORE STATS:\n    Mean   : {df['composite_score'].mean():.3f}\n    Max    : {df['composite_score'].max():.3f}\n    Min    : {df['composite_score'].min():.3f}\n\n  TOP 5 BRANCHES:\n{df[['rank', 'name', 'composite_score', 'action']].head(5).to_string(index=False)}\n\n  BOTTOM 5 BRANCHES:\n{df[['rank', 'name', 'composite_score', 'action']].tail(5).to_string(index=False)}\n\n  ACTION SUMMARY:\n{df['action'].value_counts().to_string()}\n\n{'═' * 60}\n  Ready for dashboard → load branches_scored.csv\n{'═' * 60}\n")