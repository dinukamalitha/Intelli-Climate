import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# ── 1. LOAD ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
# Try local folder if DATA_DIR doesn't exist
DATA_FILE = os.path.join(BASE_DIR, 'SriLanka_Weather_Dataset.csv')
if not os.path.exists(DATA_FILE):
    DATA_FILE = os.path.join(BASE_DIR, 'data', 'SriLanka_Weather_Dataset.csv')

os.makedirs(MODELS_DIR, exist_ok=True)

print("\n[1/7] Loading dataset...")
if not os.path.exists(DATA_FILE):
    print(f"❌ Error: {DATA_FILE} not found.")
    exit(1)

df = pd.read_csv(DATA_FILE, parse_dates=['time'])
print(f"      {df.shape[0]:,} rows · {df['city'].nunique()} cities · {df['time'].min().date()} → {df['time'].max().date()}")

# ── 2. PREPROCESS ──────────────────────────
print("\n[2/7] Preprocessing...")
df.dropna(inplace=True)
df['is_rainy']   = (df['precipitation_sum'] > 1.0).astype(int)
df['month']      = df['time'].dt.month
df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
print(f"      Rainy days: {df['is_rainy'].mean():.1%} of all records")

# ── 3. STATION PROFILES & MONSOON FEATURES ────────────────────────
print("\n[3/7] Building station profiles & monsoon features...")

# Basic aggregation
station_profile = df.groupby('city').agg({
    'temperature_2m_mean': 'mean',
    'temperature_2m_max':  'mean',
    'temperature_2m_min':  'mean',
    'precipitation_sum':   'mean',
    'rain_sum':            'mean',
    'windspeed_10m_max':   'mean',
    'shortwave_radiation_sum': 'mean',
    'latitude':            'first',
    'longitude':           'first',
    'elevation':           'first',
}).reset_index()

# Monsoon feature engineering
monthly_rain = df.groupby(['city', 'month'])['precipitation_sum'].mean().unstack()
station_profile = station_profile.merge(
    monthly_rain.add_prefix('m'),
    left_on='city', right_index=True
)

# SW Monsoon = May–Sep  |  NE Monsoon = Oct–Feb
station_profile['sw_rain'] = station_profile[['m5','m6','m7','m8','m9']].mean(axis=1)
station_profile['ne_rain'] = station_profile[['m10','m11','m12','m1','m2']].mean(axis=1)
station_profile['monsoon_ratio'] = station_profile['sw_rain'] / (station_profile['ne_rain'] + 0.001)

# ── 4. DBSCAN ─────────────────────────────────────────────────────
print("\n[4/7] DBSCAN clustering (eps=2.0, min_samples=3)...")

CLUSTER_FEATS = [
    'temperature_2m_mean',
    'precipitation_sum',
    'windspeed_10m_max',
    'shortwave_radiation_sum',
    'elevation',
    'sw_rain',
    'ne_rain',
    'monsoon_ratio',
]

scaler_cluster = StandardScaler()
X_station = scaler_cluster.fit_transform(station_profile[CLUSTER_FEATS])

dbscan = DBSCAN(eps=2.0, min_samples=3, metric='euclidean')
labels = dbscan.fit_predict(X_station)
station_profile['climate_zone'] = labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = list(labels).count(-1)
print(f"      Clusters found: {n_clusters} | Noise (isolated): {n_noise}")

mask = labels != -1
if n_clusters > 1:
    sil = silhouette_score(X_station[mask], labels[mask])
    db  = davies_bouldin_score(X_station[mask], labels[mask])
    print(f"      Silhouette: {sil:.4f} | Davies-Bouldin: {db:.4f}")
else:
    sil, db = 0, 0
    print("      Not enough clusters to compute silhouette/DB scores.")

# ── 5. CLASSIFICATION DATA ────────────────────────────────────────
print("\n[5/7] Naive Bayes classification setup...")
zone_map = station_profile.set_index('city')['climate_zone'].to_dict()
df['climate_zone'] = df['city'].map(zone_map).fillna(-1).astype(int)

CLS_FEATS = [
     'temperature_2m_mean',
    'temp_range',
    'windspeed_10m_max',
    'shortwave_radiation_sum',
    'et0_fao_evapotranspiration',
    'month',
    'climate_zone'
]
X = df[CLS_FEATS]
y = df['is_rainy']
groups = df['city']

X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv)

scaler_cls = StandardScaler()
X_train_sc = scaler_cls.fit_transform(X_train)
X_val_sc   = scaler_cls.transform(X_val)
X_test_sc  = scaler_cls.transform(X_test)

# ── 6. GAUSSIAN NAIVE BAYES ───────────────────────────────────────
print("\n[6/7] Training Gaussian Naive Bayes...")
best_acc, best_vs, best_model = 0, 1e-9, None
for vs in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
    gnb = GaussianNB(var_smoothing=vs)
    gnb.fit(X_train_sc, y_train)
    acc = gnb.score(X_val_sc, y_val)
    print(f"      var_smoothing={vs:.0e} → Val Acc: {acc:.4f}")
    if acc > best_acc: best_acc, best_vs, best_model = acc, vs, gnb

y_pred  = best_model.predict(X_test_sc)
y_proba = best_model.predict_proba(X_test_sc)[:,1]
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"\n      TEST RESULTS: Acc: {acc:.4f} | F1: {f1:.4f} | AUC-ROC: {auc:.4f}")

# ── 7. SURROGATE FOR SHAP + SAVE ─────────────────────────────────
print("\n[7/7] Training SHAP surrogate + saving all models...")
mask_sp = station_profile['climate_zone'] != -1
X_surr  = pd.DataFrame(X_station[mask_sp.values], columns=CLUSTER_FEATS)
y_surr  = station_profile.loc[mask_sp, 'climate_zone'].values
rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=6)
rf.fit(X_surr, y_surr)

# Build ZONE_NAMES dynamically based on found labels
# Apply final prescribed zone names
ZONE_NAMES = {
     0: 'Wet Zone',
     1: 'Central Highlands',
     2: 'Southern Coast',
     3: 'Dry Zone',
     4: 'Semi Arid Zone',
    -1: 'Isolated / Unique Microclimate',
}

# Ensure any unexpected clusters (if any) get a name early
for lbl in sorted(station_profile['climate_zone'].unique()):
    if lbl not in ZONE_NAMES:
        ZONE_NAMES[lbl] = f'Climate Region {lbl}'

station_profile['zone_name'] = station_profile['climate_zone'].map(ZONE_NAMES)
station_profile['zone_color'] = station_profile['climate_zone'].map({
     0: '#2d6abf',
     1: '#7ab648',
     2: '#c47d2d',
     3: '#9b59b6',
     4: '#e74c3c',
    -1: '#888888',
})

to_save = {
    'gnb.pkl':           best_model,
    'scaler_cls.pkl':    scaler_cls,
    'scaler_cluster.pkl':scaler_cluster,
    'station_profile.pkl':station_profile,
    'X_train_sc.pkl':    X_train_sc,
    'cls_feats.pkl':     CLS_FEATS,
    'cluster_feats.pkl': CLUSTER_FEATS,
    'rf_surrogate.pkl':  rf,
    'zone_names.pkl':    ZONE_NAMES,
    'X_test_sc.pkl':     X_test_sc,
    'y_test.pkl':        y_test,
    'y_pred.pkl':        y_pred,
    'y_proba.pkl':       y_proba,
    'metrics.pkl':       {'accuracy':acc,'f1':f1,'auc':auc,'best_vs':best_vs,
                          'sil_score':sil,'db_score':db},
}

for fname, obj in to_save.items():
    path = os.path.join(MODELS_DIR, fname)
    with open(path, 'wb') as f: 
        pickle.dump(obj, f)

# Save FULL df for explorer (including -1 zones), but use df_cls for training
df.to_csv(os.path.join(MODELS_DIR, 'df_processed.csv'), index=False)
print("\n✅  All models and full dataset saved in /models/")
