"""
train.py — Run this ONCE before launching app.py
Trains DBSCAN + Gaussian Naive Bayes from SriLanka_Weather_Dataset.csv
Saves all models to /models/
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# ── 1. LOAD ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

print("\n[1/7] Loading dataset...")
df = pd.read_csv(os.path.join(DATA_DIR, 'SriLanka_Weather_Dataset.csv'), parse_dates=['time'])
print(f"      {df.shape[0]:,} rows · {df['city'].nunique()} cities · {df['time'].min().date()} → {df['time'].max().date()}")

# ── 2. PREPROCESS (exact notebook steps) ──────────────────────────
print("\n[2/7] Preprocessing...")
df.dropna(inplace=True)
df['is_rainy']   = (df['precipitation_sum'] > 1.0).astype(int)
df['month']      = df['time'].dt.month
df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
df['season']     = df['month'].map({12:'NE-Monsoon',1:'NE-Monsoon',2:'Inter',
                                     3:'Inter',4:'Inter',5:'SW-Monsoon',6:'SW-Monsoon',
                                     7:'SW-Monsoon',8:'SW-Monsoon',9:'SW-Monsoon',
                                     10:'NE-Monsoon',11:'NE-Monsoon'})
print(f"      Rainy days: {df['is_rainy'].mean():.1%} of all records")

# ── 3. STATION PROFILES ───────────────────────────────────────────
print("\n[3/7] Building station profiles...")
station_profile = df.groupby('city').agg({
    'temperature_2m_mean':'mean','temperature_2m_max':'mean','temperature_2m_min':'mean',
    'precipitation_sum':'mean','rain_sum':'mean','windspeed_10m_max':'mean',
    'shortwave_radiation_sum':'mean','latitude':'first','longitude':'first',
}).reset_index()

# ── 4. DBSCAN (exact notebook settings) ───────────────────────────
print("\n[4/7] DBSCAN clustering (eps=1.5, min_samples=3)...")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

CLUSTER_FEATS = ['temperature_2m_mean','precipitation_sum','rain_sum',
                 'windspeed_10m_max','shortwave_radiation_sum','latitude','longitude']
scaler_cluster = StandardScaler()
X_station = scaler_cluster.fit_transform(station_profile[CLUSTER_FEATS])

dbscan = DBSCAN(eps=1.5, min_samples=3, metric='euclidean')
labels = dbscan.fit_predict(X_station)
station_profile['climate_zone'] = labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = list(labels).count(-1)
print(f"      Clusters found: {n_clusters} | Noise (isolated): {n_noise}")

mask = labels != -1
sil = silhouette_score(X_station[mask], labels[mask])
db  = davies_bouldin_score(X_station[mask], labels[mask])
print(f"      Silhouette: {sil:.4f} | Davies-Bouldin: {db:.4f}")

km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
km_labels = km.fit_predict(X_station)
km_sil = silhouette_score(X_station, km_labels)
print(f"      K-Means Silhouette (for comparison): {km_sil:.4f}")
print(station_profile[['city','climate_zone']].to_string(index=False))

# ── 5. CLASSIFICATION DATA ────────────────────────────────────────
print("\n[5/7] Naive Bayes classification setup...")
zone_map = station_profile.set_index('city')['climate_zone'].to_dict()
df['climate_zone'] = df['city'].map(zone_map)
df_cls = df[df['climate_zone'] != -1].copy()

CLS_FEATS = ['temperature_2m_mean','temperature_2m_max','temperature_2m_min',
             'windspeed_10m_max','shortwave_radiation_sum','et0_fao_evapotranspiration',
             'temp_range','month','climate_zone']
X = df_cls[CLS_FEATS]; y = df_cls['is_rainy']
print(f"      Class balance: {y.value_counts().to_dict()}")

from sklearn.model_selection import train_test_split
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv)
print(f"      Train:{len(X_train):,} | Val:{len(X_val):,} | Test:{len(X_test):,}")

scaler_cls = StandardScaler()
X_train_sc = scaler_cls.fit_transform(X_train)
X_val_sc   = scaler_cls.transform(X_val)
X_test_sc  = scaler_cls.transform(X_test)

# ── 6. GAUSSIAN NAIVE BAYES ───────────────────────────────────────
print("\n[6/7] Training Gaussian Naive Bayes...")
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

best_acc, best_vs, best_model = 0, 1e-9, None
for vs in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
    gnb = GaussianNB(var_smoothing=vs)
    gnb.fit(X_train_sc, y_train)
    acc = gnb.score(X_val_sc, y_val)
    print(f"      var_smoothing={vs:.0e} → Val Acc: {acc:.4f}")
    if acc > best_acc: best_acc, best_vs, best_model = acc, vs, gnb

print(f"\n      Best var_smoothing: {best_vs}")
y_pred  = best_model.predict(X_test_sc)
y_proba = best_model.predict_proba(X_test_sc)[:,1]
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"\n      TEST RESULTS:")
print(f"      Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC-ROC: {auc:.4f}")
print(classification_report(y_test, y_pred, target_names=['Not Rainy','Rainy'], digits=4))

# ── 7. SURROGATE FOR SHAP + SAVE ─────────────────────────────────
print("\n[7/7] Training SHAP surrogate + saving all models...")
from sklearn.ensemble import RandomForestClassifier
mask_sp = station_profile['climate_zone'] != -1
X_surr  = pd.DataFrame(X_station[mask_sp.values], columns=CLUSTER_FEATS)
y_surr  = station_profile.loc[mask_sp, 'climate_zone'].values
rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=6)
rf.fit(X_surr, y_surr)

ZONE_NAMES = {-1:'Isolated/Unique', 0:'Wet Zone (Western)', 1:'Southern Coast', 2:'Central Highlands'}

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
                          'sil_score':sil,'db_score':db,'kmeans_sil':km_sil},
}
for fname, obj in to_save.items():
    path = os.path.join(MODELS_DIR, fname)
    with open(path, 'wb') as f: 
        pickle.dump(obj, f)

df_cls.to_csv(os.path.join(MODELS_DIR, 'df_processed.csv'), index=False)

print("\n✅  All models saved to /models/")
print(f"   Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
print("="*55)
print("   Next step:  streamlit run app.py")
print("="*55)
