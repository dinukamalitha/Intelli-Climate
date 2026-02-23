import os
import pickle
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def get_models_mtime():
    """Returns the latest modification time of the models directory."""
    if not os.path.exists(MODELS_DIR): return 0
    files = [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR)]
    if not files: return 0
    return max(os.path.getmtime(f) for f in files)

@st.cache_resource
def load_models_and_data(mtime):
    def _p(f): 
        path = os.path.join(MODELS_DIR, f)
        return pickle.load(open(path, 'rb'))
    
    try:
        data = {
            'gnb':           _p('gnb.pkl'),
            'scaler_cls':    _p('scaler_cls.pkl'),
            'scaler_cluster':_p('scaler_cluster.pkl'),
            'station':       _p('station_profile.pkl'),
            'X_train_sc':    _p('X_train_sc.pkl'),
            'cls_feats':     _p('cls_feats.pkl'),
            'cluster_feats': _p('cluster_feats.pkl'),
            'rf':            _p('rf_surrogate.pkl'),
            'metrics':       _p('metrics.pkl'),
            'zone_names':    _p('zone_names.pkl'),
            'X_test_sc':     _p('X_test_sc.pkl'),
            'y_test':        _p('y_test.pkl'),
            'y_pred':        _p('y_pred.pkl'),
            'y_proba':       _p('y_proba.pkl'),
            'df':            pd.read_csv(os.path.join(MODELS_DIR, 'df_processed.csv'), parse_dates=['time']),
        }
        return data
    except FileNotFoundError:
        st.error("⚠️  Model files not found. Run `python train.py` first.")
        st.stop()

def get_helper_variables(M):
    """Returns derived variables and maps from the loaded model data."""
    station = M['station']
    
    helpers = {
        'all_cities': sorted(station['city'].tolist()),
        'in_model':   sorted(station[station['climate_zone'] != -1]['city'].tolist()),
        'zone_map':   station.set_index('city')['climate_zone'].to_dict(),
        'coords':     station.set_index('city')[['latitude','longitude']].to_dict(orient='index'),
    }
    return helpers
