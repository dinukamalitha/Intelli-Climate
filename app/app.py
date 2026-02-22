"""
Sri Lanka Climate & Rainfall Intelligence App
Built from: ML_Assignment.ipynb + SriLanka_Weather_Dataset.csv
Run: streamlit run app.py
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sri Lanka Climate Intelligence",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Instrument Sans', sans-serif; }

.stApp {
    background: #f5f2ee;
    color: #1a1a1a;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1c2b1e !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #c8d9c0 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e8f0e0 !important; }

/* Inputs in sidebar */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: #e8f0e0 !important;
}
[data-testid="stSidebar"] .stSlider [role="slider"] { background: #7ab648 !important; }
[data-testid="stSidebar"] label { color: #8aaa78 !important; font-size:0.75rem !important;
    text-transform:uppercase; letter-spacing:0.1em; font-weight:600 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #ede9e3;
    border-radius: 0;
    padding: 0;
    border-bottom: 2px solid #d4cfc8;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #7a7068 !important;
    border-radius: 0 !important;
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 14px 28px !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #2d5a1b !important;
    border-bottom: 3px solid #2d5a1b !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 28px !important; background: #f5f2ee; }

/* Cards */
.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 28px 30px;
    border: 1px solid #e4dfd8;
    margin-bottom: 16px;
}
.card-dark {
    background: #1c2b1e;
    border-radius: 16px;
    padding: 28px 30px;
    border: none;
    margin-bottom: 16px;
    color: #e8f0e0;
}
.card-accent {
    background: linear-gradient(135deg, #2d5a1b 0%, #4a8a2e 100%);
    border-radius: 16px;
    padding: 28px 30px;
    border: none;
    margin-bottom: 16px;
    color: white;
}

/* Metric pill */
.metric-row { display:flex; gap:12px; margin-bottom:20px; flex-wrap:wrap; }
.metric-pill {
    background:#fff; border:1px solid #e4dfd8; border-radius:12px;
    padding:16px 20px; flex:1; min-width:110px; text-align:center;
}
.metric-pill .val { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:700; color:#2d5a1b; }
.metric-pill .lbl { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; color:#9a9088; margin-top:2px; }

/* Prediction result */
.result-rainy {
    background: linear-gradient(135deg, #1a3d8f 0%, #2d6abf 100%);
    border-radius: 20px; padding:36px; text-align:center; color:white;
}
.result-dry {
    background: linear-gradient(135deg, #8f5a1a 0%, #c47d2d 100%);
    border-radius: 20px; padding:36px; text-align:center; color:white;
}
.result-icon { font-size:4rem; line-height:1; margin-bottom:12px; }
.result-title { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; margin-bottom:6px; }
.result-conf { font-size:1rem; opacity:0.85; }

/* Section title */
.sec-title {
    font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700;
    color:#1a1a1a; margin:0 0 6px; line-height:1.2;
}
.sec-sub { font-size:0.88rem; color:#7a7068; margin-bottom:24px; line-height:1.6; }

/* Zone badge */
.zbadge {
    display:inline-block; padding:5px 14px; border-radius:20px;
    font-size:0.78rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase;
}

/* Buttons */
.stButton > button {
    background: #2d5a1b !important; color: #fff !important;
    border: none !important; border-radius: 12px !important;
    font-family:'Syne',sans-serif !important; font-weight:700 !important;
    font-size:0.95rem !important; letter-spacing:0.05em !important;
    padding: 14px 32px !important; width:100% !important;
    text-transform: uppercase !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #3d7825 !important;
    box-shadow: 0 6px 24px rgba(45,90,27,0.35) !important;
    transform: translateY(-1px) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid #e4dfd8; }

/* Scrollbar */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#f0ece6; }
::-webkit-scrollbar-thumb { background:#c4bdb5; border-radius:3px; }

/* Sliders */
.stSlider [data-baseweb="slider"] [role="slider"] { background:#2d5a1b !important; }

/* Alert */
.stAlert { border-radius:12px !important; }

hr { border-color:#e4dfd8 !important; }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODELS ────────────────────────────────────────────────────────────────
@st.cache_resource
def load():
    def _p(f): return pickle.load(open(f'models/{f}','rb'))
    return {
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
        'df':            pd.read_csv('models/df_processed.csv', parse_dates=['time']),
    }

try:
    M = load()
except FileNotFoundError:
    st.error("⚠️  Model files not found. Run `python train.py` first.")
    st.stop()

# Helpers
gnb         = M['gnb']
scaler_cls  = M['scaler_cls']
station     = M['station']
CLS_FEATS   = M['cls_feats']
ZONE_NAMES  = M['zone_names']
df          = M['df']
metrics     = M['metrics']

ZONE_COLORS  = {-1:'#9a9088', 0:'#2d6abf', 1:'#c47d2d', 2:'#7ab648'}
ZONE_BG      = {-1:'#f0ece6', 0:'#e8f0fb', 1:'#fdf0e0', 2:'#edf6e0'}
ZONE_TEXT    = {-1:'#5a5048', 0:'#1a3d8f', 1:'#8f5a1a', 2:'#2d5a1b'}

ALL_CITIES   = sorted(station['city'].tolist())
IN_MODEL     = sorted(station[station['climate_zone'] != -1]['city'].tolist())
ZONE_MAP     = station.set_index('city')['climate_zone'].to_dict()
COORDS       = station.set_index('city')[['latitude','longitude']].to_dict(orient='index')
MONTH_NAMES  = ['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

WMO_CODES = {
    0:'Clear sky ☀️', 1:'Mainly clear 🌤️', 2:'Partly cloudy ⛅', 3:'Overcast ☁️',
    51:'Light drizzle 🌦️', 53:'Moderate drizzle 🌦️', 55:'Dense drizzle 🌧️',
    61:'Slight rain 🌧️', 63:'Moderate rain 🌧️', 65:'Heavy rain ⛈️'
}

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 28px;text-align:center'>
        <div style='font-size:2.8rem'>🌿</div>
        <div style='font-family:Syne,sans-serif;font-size:1.25rem;font-weight:800;
                    color:#e8f0e0;margin-top:10px'>Sri Lanka</div>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:600;
                    color:#7ab648;letter-spacing:0.08em'>Climate Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#5a7a50;margin-bottom:10px'>Model Performance</div>", unsafe_allow_html=True)
    for lbl, val, fmt in [("Accuracy", metrics['accuracy'], "{:.1%}"),
                            ("F1 Score", metrics['f1'],       "{:.4f}"),
                            ("AUC-ROC",  metrics['auc'],      "{:.4f}")]:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
                    padding:9px 0;border-bottom:1px solid rgba(255,255,255,0.07)'>
            <span style='font-size:0.82rem;color:#8aaa78'>{lbl}</span>
            <span style='font-size:1rem;color:#7ab648;font-weight:700;font-family:Syne,sans-serif'>{fmt.format(val)}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#5a7a50;margin-bottom:10px'>Dataset</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.83rem;color:#8aaa78;line-height:2'>
        📍 30 Cities across Sri Lanka<br>
        📅 Jan 2010 – Jun 2023<br>
        📊 {len(df):,} daily records<br>
        🌧️ {df['is_rainy'].mean():.1%} rainy days<br>
        🔬 Source: Open-Meteo / Kaggle
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#5a7a50;margin-bottom:10px'>Climate Zones</div>", unsafe_allow_html=True)
    for zone_id, zone_name in sorted(ZONE_NAMES.items()):
        cities_in = station[station['climate_zone']==zone_id]['city'].tolist()
        if not cities_in: continue
        color = ZONE_COLORS.get(zone_id, '#9a9088')
        st.markdown(f"""
        <div style='margin-bottom:10px;padding:10px 12px;
                    background:rgba(255,255,255,0.05);border-radius:8px;
                    border-left:3px solid {color}'>
            <div style='color:{color};font-size:0.8rem;font-weight:700;
                        text-transform:uppercase;letter-spacing:0.05em'>{zone_name}</div>
            <div style='color:#6a8a60;font-size:0.72rem;margin-top:3px'>{len(cities_in)} stations</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#5a7a50;margin-bottom:10px'>Algorithms</div>", unsafe_allow_html=True)
    for a, p in [("DBSCAN","Climate clustering"),("Gaussian NB","Rain prediction"),("SHAP","Feature importance"),("LIME","Per-prediction XAI")]:
        st.markdown(f"<div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05)'><span style='color:#7ab648;font-size:0.8rem;font-weight:600'>{a}</span><span style='color:#6a8a60;font-size:0.76rem'>{p}</span></div>", unsafe_allow_html=True)


# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:40px 0 8px'>
    <div style='font-family:Syne,sans-serif;font-size:2.8rem;font-weight:800;
                color:#1a1a1a;line-height:1.1;margin-bottom:8px'>
        Sri Lanka Climate &amp;<br>Rainfall Intelligence
    </div>
    <div style='font-size:1rem;color:#7a7068;max-width:620px;line-height:1.6'>
        DBSCAN climate zone clustering · Gaussian Naive Bayes rainfall prediction ·
        SHAP &amp; LIME explainability — built from 13 years of real Sri Lanka weather data
    </div>
</div>
<hr style='margin:24px 0 0'>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️  Climate Zones",
    "🌧️  Predict Rainfall",
    "📈  City Explorer",
    "🔍  Model Explainability",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLIMATE ZONES
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-title">DBSCAN Climate Zone Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Your model automatically discovered 3 distinct climate zones from 30 Sri Lankan cities — without being told how many zones to find. 9 cities were classified as "Isolated" due to unique microclimates.</div>', unsafe_allow_html=True)

    map_col, detail_col = st.columns([2, 1], gap="large")

    with map_col:
        fig = go.Figure()
        # Sri Lanka outline via a dummy background
        for zone_id in sorted(station['climate_zone'].unique()):
            sub = station[station['climate_zone'] == zone_id]
            zname = ZONE_NAMES.get(zone_id, f'Zone {zone_id}')
            color = ZONE_COLORS.get(zone_id, '#9a9088')
            hover = sub.apply(lambda r:
                f"<b>{r['city']}</b><br>"
                f"Zone: {zname}<br>"
                f"Avg Temp: {r['temperature_2m_mean']:.1f}°C<br>"
                f"Avg Rain: {r['precipitation_sum']:.1f} mm/day<br>"
                f"Avg Wind: {r['windspeed_10m_max']:.1f} km/h", axis=1)
            fig.add_trace(go.Scattergeo(
                lon=sub['longitude'], lat=sub['latitude'],
                mode='markers+text',
                name=zname,
                text=sub['city'],
                textposition='top center',
                textfont=dict(size=8.5, color='#333'),
                hovertemplate=hover + '<extra></extra>',
                marker=dict(
                    size=20 if zone_id != -1 else 14,
                    color=color,
                    symbol='circle',
                    line=dict(width=2.5, color='white'),
                    opacity=0.92
                )
            ))
        fig.update_layout(
            geo=dict(
                scope='asia', showland=True, landcolor='#e8e2d9',
                showocean=True, oceancolor='#d0e8f5',
                showcountries=True, countrycolor='#b0a898',
                showcoastlines=True, coastlinecolor='#8a7a6a',
                showlakes=True, lakecolor='#d0e8f5',
                center=dict(lat=7.85, lon=80.7),
                projection_scale=11,
                lataxis_range=[5.5, 10.3],
                lonaxis_range=[79.2, 82.3],
                bgcolor='#f5f2ee',
            ),
            paper_bgcolor='#ffffff',
            plot_bgcolor='#f5f2ee',
            margin=dict(l=0,r=0,t=0,b=0), height=520,
            legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#e4dfd8',
                        borderwidth=1, font=dict(size=12, color='#333'),
                        x=0.01, y=0.99),
            font=dict(color='#1a1a1a', family='Instrument Sans')
        )
        st.plotly_chart(fig, use_container_width=True)

    with detail_col:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        for zone_id in sorted(ZONE_NAMES.keys()):
            cities_in = station[station['climate_zone']==zone_id]
            if len(cities_in) == 0: continue
            zname = ZONE_NAMES[zone_id]
            color = ZONE_COLORS.get(zone_id, '#9a9088')
            bg    = ZONE_BG.get(zone_id, '#f5f2ee')
            avg_r = cities_in['precipitation_sum'].mean()
            avg_t = cities_in['temperature_2m_mean'].mean()
            avg_w = cities_in['windspeed_10m_max'].mean()
            city_list = ', '.join(cities_in['city'].tolist())
            st.markdown(f"""
            <div style='background:{bg};border-radius:14px;padding:16px 18px;
                        margin-bottom:14px;border-left:4px solid {color}'>
                <div style='color:{color};font-family:Syne,sans-serif;font-size:0.95rem;
                            font-weight:700;text-transform:uppercase;letter-spacing:0.05em'>
                    {zname}
                </div>
                <div style='display:flex;gap:16px;margin:8px 0;flex-wrap:wrap'>
                    <span style='font-size:0.78rem;color:#5a5048'>
                        🌡️ {avg_t:.1f}°C
                    </span>
                    <span style='font-size:0.78rem;color:#5a5048'>
                        🌧️ {avg_r:.1f}mm/day
                    </span>
                    <span style='font-size:0.78rem;color:#5a5048'>
                        💨 {avg_w:.1f}km/h
                    </span>
                </div>
                <div style='font-size:0.76rem;color:#7a7068;line-height:1.6'>
                    {city_list}
                </div>
            </div>""", unsafe_allow_html=True)

    # Stats table
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title" style="font-size:1.2rem">Zone Comparison</div>', unsafe_allow_html=True)
    zone_stats = []
    for zid in sorted(ZONE_NAMES.keys()):
        sub = station[station['climate_zone']==zid]
        if len(sub)==0: continue
        zone_stats.append({
            'Zone': ZONE_NAMES[zid],
            '# Cities': len(sub),
            'Avg Temp (°C)': f"{sub['temperature_2m_mean'].mean():.1f}",
            'Avg Rain (mm/day)': f"{sub['precipitation_sum'].mean():.2f}",
            'Avg Wind (km/h)': f"{sub['windspeed_10m_max'].mean():.1f}",
            'Avg Radiation (MJ/m²)': f"{sub['shortwave_radiation_sum'].mean():.1f}",
        })
    st.dataframe(pd.DataFrame(zone_stats), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT RAINFALL
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-title">Rainfall Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Select a city and configure today\'s weather parameters. The Gaussian Naive Bayes model (80.85% accuracy) will predict whether it will rain and explain <i>why</i> using LIME.</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### 📍 Location & Date")
        city_opt = IN_MODEL  # only cities the model was trained on
        city = st.selectbox("City", city_opt,
                            index=city_opt.index('Colombo') if 'Colombo' in city_opt else 0)
        zone = ZONE_MAP.get(city, 0)
        zname = ZONE_NAMES.get(zone, '')
        color = ZONE_COLORS.get(zone, '#9a9088')
        bg    = ZONE_BG.get(zone, '#f5f2ee')
        st.markdown(f"""
        <div style='background:{bg};border-left:4px solid {color};border-radius:0 10px 10px 0;
                    padding:10px 16px;margin-bottom:16px'>
            <span style='color:{color};font-weight:700;font-size:0.85rem'>{zname}</span>
            <span style='color:#9a9088;font-size:0.78rem;margin-left:8px'>
                {COORDS[city]['latitude']:.2f}°N · {COORDS[city]['longitude']:.2f}°E
            </span>
        </div>""", unsafe_allow_html=True)

        month = st.select_slider("Month", options=list(range(1,13)),
                                  format_func=lambda x: MONTH_NAMES[x], value=6)

        st.markdown("#### 🌡️ Temperature")
        temp_mean = st.slider("Mean Temperature (°C)", 14.0, 34.0, 26.5, 0.5)
        c1, c2 = st.columns(2)
        with c1:
            temp_max = st.slider("Max (°C)", float(temp_mean), 40.0, min(temp_mean+4,40.0), 0.5)
        with c2:
            temp_min = st.slider("Min (°C)", 10.0, float(temp_mean), max(temp_mean-5,10.0), 0.5)

        st.markdown("#### 🌬️ Atmospheric")
        wind      = st.slider("Max Wind Speed (km/h)", 0.0, 50.0, 14.0, 0.5)
        radiation = st.slider("Solar Radiation (MJ/m²)", 4.0, 30.0, 17.0, 0.5)
        et0       = st.slider("Evapotranspiration (mm)", 1.0, 9.0, 4.2, 0.1)

        temp_range = round(temp_max - temp_min, 2)
        st.caption(f"Computed temp range: **{temp_range:.1f}°C** (max − min)")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        predict = st.button("⚡  Run Prediction", type="primary")

    with right:
        if predict:
            feats = np.array([[temp_mean, temp_max, temp_min,
                               wind, radiation, et0, temp_range, month, zone]])
            feats_sc = scaler_cls.transform(feats)
            pred  = gnb.predict(feats_sc)[0]
            proba = gnb.predict_proba(feats_sc)[0]
            p_rain = proba[1]
            p_dry  = proba[0]

            # Result box
            if pred == 1:
                st.markdown(f"""
                <div class='result-rainy'>
                    <div class='result-icon'>🌧️</div>
                    <div class='result-title'>RAINY DAY</div>
                    <div class='result-conf'>Rain probability: <b>{p_rain*100:.1f}%</b> · Confidence: {p_rain*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-dry'>
                    <div class='result-icon'>☀️</div>
                    <div class='result-title'>DRY DAY</div>
                    <div class='result-conf'>No-rain probability: <b>{p_dry*100:.1f}%</b> · Rain chance: {p_rain*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            # Probability bar
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            fig_prob = go.Figure(go.Bar(
                x=[p_dry*100, p_rain*100],
                y=['☀️ Dry', '🌧️ Rainy'],
                orientation='h',
                marker_color=['#c47d2d','#2d6abf'],
                text=[f'{p_dry*100:.1f}%', f'{p_rain*100:.1f}%'],
                textposition='inside',
                textfont=dict(color='white', size=13, family='Syne'),
            ))
            fig_prob.update_layout(
                height=110, margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor='white', plot_bgcolor='white',
                xaxis=dict(range=[0,100], showgrid=False, visible=False),
                yaxis=dict(showgrid=False),
                font=dict(color='#1a1a1a', family='Instrument Sans'),
                showlegend=False,
                bargap=0.3,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # LIME explanation
            st.markdown("#### 🔬 Why this prediction? (LIME)")
            try:
                import lime.lime_tabular
                lime_exp = lime.lime_tabular.LimeTabularExplainer(
                    M['X_train_sc'], feature_names=CLS_FEATS,
                    class_names=['Not Rainy','Rainy'], mode='classification', random_state=42
                )
                exp = lime_exp.explain_instance(feats_sc[0], gnb.predict_proba, num_features=9)
                exp_list = sorted(exp.as_list(), key=lambda x: abs(x[1]), reverse=True)

                feat_labels = [e[0] for e in exp_list]
                feat_vals   = [e[1] for e in exp_list]
                bar_colors  = ['#2d6abf' if v > 0 else '#c47d2d' for v in feat_vals]

                fig_lime, ax = plt.subplots(figsize=(7, 4.2))
                fig_lime.patch.set_facecolor('white')
                ax.set_facecolor('#fafaf8')
                idx = np.argsort(np.abs(feat_vals))
                ax.barh([feat_labels[i] for i in idx], [feat_vals[i] for i in idx],
                        color=[bar_colors[i] for i in idx], height=0.65, edgecolor='none')
                ax.axvline(0, color='#c8c0b8', linewidth=1.2)
                ax.set_title(f'LIME — {city}, {MONTH_NAMES[month]}',
                             color='#1a1a1a', fontsize=11, fontweight='bold', pad=10)
                ax.set_xlabel('← Pushes toward Dry   |   Pushes toward Rainy →',
                              color='#7a7068', fontsize=8.5)
                ax.tick_params(colors='#5a5048', labelsize=8.5)
                for sp in ax.spines.values(): sp.set_edgecolor('#e4dfd8')
                plt.tight_layout()
                st.pyplot(fig_lime, use_container_width=True)
                plt.close()
            except Exception as e:
                st.info(f"Install lime: `pip install lime` ({e})")

            # Historical context for the city+month
            st.markdown(f"#### 📊 Historical Context: {city} in {MONTH_NAMES[month]}")
            hist = df[(df['city']==city) & (df['month']==month)]
            if len(hist):
                h_rain = hist['is_rainy'].mean()
                h_avg  = hist['precipitation_sum'].mean()
                c1,c2,c3 = st.columns(3)
                c1.metric("Historical Rain Days", f"{h_rain:.0%}")
                c2.metric("Avg Daily Rain", f"{h_avg:.1f} mm")
                c3.metric("Your Predicted", "🌧️ Rain" if pred==1 else "☀️ Dry")
        else:
            st.markdown("""
            <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
                        height:500px;border:2px dashed #d4cfc8;border-radius:20px;
                        background:#faf8f5;text-align:center;padding:40px'>
                <div style='font-size:4rem;margin-bottom:16px'>⚡</div>
                <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:700;
                            color:#1a1a1a;margin-bottom:8px'>Configure & Predict</div>
                <div style='font-size:0.88rem;color:#9a9088;max-width:280px;line-height:1.6'>
                    Select a city, set the weather parameters on the left, then
                    click <b>Run Prediction</b> for a rainfall forecast + LIME explanation
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CITY EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-title">City Weather Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Explore 13 years of historical weather patterns for any city. Discover seasonal cycles, year-on-year trends, and cross-city comparisons.</div>', unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
    with ctrl1:
        sel_city = st.selectbox("City", ALL_CITIES, key='explorer_city',
                                 index=ALL_CITIES.index('Colombo') if 'Colombo' in ALL_CITIES else 0)
    with ctrl2:
        chart_type = st.selectbox("Chart", [
            "Monthly Rainfall Pattern",
            "Temperature Range by Month",
            "Rainy Day Frequency",
            "Annual Rainfall Trend",
            "Weather Code Distribution",
            "All-City Rainfall Heatmap",
        ])
    with ctrl3:
        yr_min, yr_max = st.select_slider("Year Range", options=list(range(2010,2024)),
                                            value=(2010, 2023))

    city_df = df[(df['city']==sel_city) &
                 (df['time'].dt.year.between(yr_min, yr_max))].copy()
    city_df['year'] = city_df['time'].dt.year

    MNAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    fig_style = dict(paper_bgcolor='white', plot_bgcolor='#fafaf8',
                     font=dict(color='#1a1a1a', family='Instrument Sans'),
                     margin=dict(l=60,r=20,t=50,b=50))

    if chart_type == "Monthly Rainfall Pattern":
        monthly = city_df.groupby('month')['precipitation_sum'].mean().reset_index()
        avg = monthly['precipitation_sum'].mean()
        bar_c = ['#2d6abf' if v >= avg else '#7ab6df' for v in monthly['precipitation_sum']]
        fig_c = go.Figure([
            go.Bar(x=[MNAMES[m-1] for m in monthly['month']],
                   y=monthly['precipitation_sum'], marker_color=bar_c,
                   name='Avg Daily Rain', hovertemplate='%{x}: %{y:.2f} mm<extra></extra>'),
            go.Scatter(x=MNAMES, y=[avg]*12, mode='lines',
                       line=dict(color='#c47d2d', dash='dash', width=2),
                       name=f'Annual avg ({avg:.2f}mm)')
        ])
        fig_c.update_layout(title=f'Average Daily Rainfall by Month — {sel_city}',
                             yaxis_title='Avg Daily Precipitation (mm)',
                             legend=dict(bgcolor='rgba(255,255,255,0.8)'), **fig_style)

    elif chart_type == "Temperature Range by Month":
        monthly = city_df.groupby('month').agg(
            max_t=('temperature_2m_max','mean'),
            mean_t=('temperature_2m_mean','mean'),
            min_t=('temperature_2m_min','mean')).reset_index()
        mx = [MNAMES[m-1] for m in monthly['month']]
        fig_c = go.Figure([
            go.Scatter(x=mx, y=monthly['max_t'], mode='lines+markers',
                       name='Max', line=dict(color='#c47d2d',width=2.5), marker=dict(size=6)),
            go.Scatter(x=mx+mx[::-1],
                       y=list(monthly['max_t'])+list(monthly['min_t'])[::-1],
                       fill='toself', fillcolor='rgba(45,106,191,0.1)',
                       line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'),
            go.Scatter(x=mx, y=monthly['mean_t'], mode='lines+markers',
                       name='Mean', line=dict(color='#2d6abf',width=2.5), marker=dict(size=6)),
            go.Scatter(x=mx, y=monthly['min_t'], mode='lines+markers',
                       name='Min', line=dict(color='#7ab648',width=2), marker=dict(size=6)),
        ])
        fig_c.update_layout(title=f'Monthly Temperature Range — {sel_city}',
                             yaxis_title='Temperature (°C)', **fig_style)

    elif chart_type == "Rainy Day Frequency":
        monthly = city_df.groupby('month')['is_rainy'].mean().reset_index()
        bar_c2 = [f'rgba(45,106,191,{0.4+v*0.6:.2f})' for v in monthly['is_rainy']]
        fig_c = go.Figure(go.Bar(
            x=[MNAMES[m-1] for m in monthly['month']],
            y=monthly['is_rainy']*100,
            marker_color=bar_c2, text=[f"{v*100:.0f}%" for v in monthly['is_rainy']],
            textposition='outside', textfont=dict(size=10),
            hovertemplate='%{x}: %{y:.1f}% rainy days<extra></extra>'
        ))
        fig_c.update_layout(title=f'Rainy Day Probability by Month — {sel_city}',
                             yaxis_title='% Days with Rain (>1mm)', yaxis_range=[0,110], **fig_style)

    elif chart_type == "Annual Rainfall Trend":
        annual = city_df.groupby('year')['precipitation_sum'].sum().reset_index()
        z = np.polyfit(annual['year'], annual['precipitation_sum'], 1)
        trend = np.poly1d(z)(annual['year'])
        fig_c = go.Figure([
            go.Scatter(x=annual['year']+annual['year'][::-1].tolist(),
                       y=list(annual['precipitation_sum'])+[annual['precipitation_sum'].mean()]*len(annual),
                       fill='toself', fillcolor='rgba(45,106,191,0.1)',
                       line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'),
            go.Scatter(x=annual['year'], y=annual['precipitation_sum'], mode='lines+markers',
                       name='Annual total', line=dict(color='#2d6abf',width=2.5), marker=dict(size=7)),
            go.Scatter(x=annual['year'], y=trend, mode='lines',
                       name='Trend', line=dict(color='#c47d2d',dash='dash',width=2)),
        ])
        fig_c.update_layout(title=f'Annual Total Rainfall — {sel_city}',
                             yaxis_title='Total Precipitation (mm/year)',
                             xaxis=dict(tickmode='linear', dtick=2), **fig_style)

    elif chart_type == "Weather Code Distribution":
        wc = city_df['weathercode'].value_counts().reset_index()
        wc.columns = ['code','count']
        wc['label'] = wc['code'].map(WMO_CODES).fillna(wc['code'].astype(str))
        wc['pct'] = wc['count']/wc['count'].sum()*100
        fig_c = go.Figure(go.Bar(
            x=wc['label'], y=wc['pct'],
            marker_color='#2d6abf',
            text=[f"{v:.1f}%" for v in wc['pct']],
            textposition='outside', textfont=dict(size=10),
            hovertemplate='%{x}<br>%{y:.1f}% of days<extra></extra>'
        ))
        fig_c.update_layout(title=f'Weather Condition Distribution — {sel_city}',
                             yaxis_title='% of Days', xaxis_tickangle=-25, **fig_style)

    elif chart_type == "All-City Rainfall Heatmap":
        dfm = df.copy()
        dfm['month'] = dfm['time'].dt.month
        pivot = dfm.groupby(['city','month'])['precipitation_sum'].mean().unstack()
        pivot.columns = MNAMES
        pivot = pivot.sort_values('Jun', ascending=False)
        fig_c = go.Figure(go.Heatmap(
            z=pivot.values, x=MNAMES, y=pivot.index.tolist(),
            colorscale='Blues', hovertemplate='%{y} · %{x}: %{z:.2f} mm<extra></extra>',
            colorbar=dict(title='mm/day', thickness=14, len=0.8)
        ))
        fig_c.update_layout(title='Average Daily Precipitation — All 30 Cities × Month',
                             height=700, **fig_style)

    if chart_type != "All-City Rainfall Heatmap":
        fig_c.update_layout(height=420)
    st.plotly_chart(fig_c, use_container_width=True)

    # Quick city summary stats below chart
    if chart_type != "All-City Rainfall Heatmap":
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Rainy Days", f"{city_df['is_rainy'].mean():.1%}")
        s2.metric("Avg Daily Rain", f"{city_df['precipitation_sum'].mean():.2f} mm")
        s3.metric("Avg Temp", f"{city_df['temperature_2m_mean'].mean():.1f}°C")
        s4.metric("Avg Max Wind", f"{city_df['windspeed_10m_max'].mean():.1f} km/h")
        s5.metric("Records", f"{len(city_df):,}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-title">Model Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Understand what your model learned — globally with SHAP and per-prediction with confusion matrix and ROC analysis.</div>', unsafe_allow_html=True)

    xai_l, xai_r = st.columns(2, gap="large")

    with xai_l:
        st.markdown("#### SHAP — What defines a Climate Zone?")
        st.caption("A surrogate Random Forest was trained on DBSCAN cluster labels, then SHAP extracted global feature importance.")
        try:
            import shap
            rf = M['rf']
            cluster_feats = M['cluster_feats']
            scaler_cluster = M['scaler_cluster']
            sp = M['station']
            mask = sp['climate_zone'] != -1
            X_surr = pd.DataFrame(
                scaler_cluster.transform(sp.loc[mask, cluster_feats]),
                columns=cluster_feats
            )
            explainer = shap.TreeExplainer(rf)
            shap_vals = explainer.shap_values(X_surr)

            if isinstance(shap_vals, list):
                mean_shap = np.mean([np.abs(sv).mean(0) for sv in shap_vals], 0)
            else:
                mean_shap = np.abs(shap_vals).mean(0)

            feat_clean = ['Mean Temp','Precipitation','Rain Sum','Wind Speed','Solar Radiation','Latitude','Longitude']
            sidx = np.argsort(mean_shap)
            bar_c = ['#2d5a1b' if i==sidx[-1] else '#7ab648' if i==sidx[-2] else '#b8d9a0' for i in range(len(mean_shap))]

            fig_s, ax_s = plt.subplots(figsize=(7, 4))
            fig_s.patch.set_facecolor('white')
            ax_s.set_facecolor('#fafaf8')
            ax_s.barh([feat_clean[i] for i in sidx], mean_shap[sidx],
                      color=[bar_c[i] for i in sidx], height=0.62, edgecolor='none')
            ax_s.set_title('SHAP Feature Importance\n(Climate Zone Assignment)', fontsize=11,
                           fontweight='bold', color='#1a1a1a', pad=10)
            ax_s.set_xlabel('Mean |SHAP value|', color='#7a7068', fontsize=9)
            ax_s.tick_params(colors='#5a5048', labelsize=9)
            for sp2 in ax_s.spines.values(): sp2.set_edgecolor('#e4dfd8')
            plt.tight_layout()
            st.pyplot(fig_s, use_container_width=True); plt.close()

            st.markdown("""
            <div style='background:#f0f7e8;border-left:4px solid #2d5a1b;border-radius:0 10px 10px 0;padding:14px 16px;margin-top:8px'>
                <b style='color:#2d5a1b;font-size:0.85rem'>Key Findings</b><br>
                <ul style='color:#3a5a30;font-size:0.82rem;margin:6px 0 0;padding-left:18px;line-height:1.8'>
                    <li><b>Precipitation</b> is the strongest driver of zone separation</li>
                    <li><b>Latitude</b> captures the North-South climate gradient (Jaffna vs Galle)</li>
                    <li><b>Mean Temperature</b> separates Highland (cooler) from Coastal zones</li>
                    <li><b>Solar Radiation</b> differentiates the Dry Zone from Wet Zone</li>
                </ul>
            </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"SHAP not available: `pip install shap` ({e})")

    with xai_r:
        st.markdown("#### Model Evaluation")
        view = st.radio("", ["Confusion Matrix","ROC Curve","Class Means (NB)"], horizontal=True, label_visibility='collapsed')

        if view == "Confusion Matrix":
            from sklearn.metrics import confusion_matrix
            y_test = M['y_test']; y_pred = M['y_pred']
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.5))
            fig_cm.patch.set_facecolor('white')
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                        xticklabels=['Not Rainy','Rainy'], yticklabels=['Not Rainy','Rainy'],
                        linewidths=0.5, linecolor='white', annot_kws={'size':14,'weight':'bold'})
            ax_cm.set_title('Confusion Matrix — Gaussian Naive Bayes', fontsize=11,
                            fontweight='bold', color='#1a1a1a', pad=10)
            ax_cm.set_ylabel('Actual', color='#7a7068', fontsize=10)
            ax_cm.set_xlabel('Predicted', color='#7a7068', fontsize=10)
            ax_cm.tick_params(colors='#5a5048')
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True); plt.close()

            tn,fp,fn,tp = cm.ravel()
            c1,c2 = st.columns(2)
            c1.metric("True Positives (Rain ✓)", f"{tp:,}")
            c2.metric("True Negatives (Dry ✓)", f"{tn:,}")
            c1.metric("False Positives (Dry predicted Rain)", f"{fp:,}")
            c2.metric("False Negatives (Rain predicted Dry)", f"{fn:,}")

        elif view == "ROC Curve":
            from sklearn.metrics import roc_curve
            y_test = M['y_test']; y_proba = M['y_proba']
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = metrics['auc']
            fig_roc = go.Figure([
                go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {auc:.4f}',
                           line=dict(color='#2d6abf', width=2.5),
                           fill='tozeroy', fillcolor='rgba(45,106,191,0.12)'),
                go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random',
                           line=dict(color='#c8c0b8', dash='dash', width=1.5)),
            ])
            fig_roc.update_layout(
                title=f'ROC Curve — Rainfall Classification (AUC = {auc:.4f})',
                xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                xaxis=dict(range=[0,1]), yaxis=dict(range=[0,1.02]),
                legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#e4dfd8', borderwidth=1),
                height=380, paper_bgcolor='white', plot_bgcolor='#fafaf8',
                font=dict(color='#1a1a1a', family='Instrument Sans'),
                margin=dict(l=60,r=20,t=50,b=50)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.markdown(f"""
            <div style='background:#f0f7e8;border-left:4px solid #2d5a1b;border-radius:0 10px 10px 0;padding:12px 16px'>
                <span style='color:#2d5a1b;font-size:0.85rem'>An AUC of <b>{auc:.4f}</b> means the model correctly ranks a randomly chosen rainy day above a dry day <b>{auc*100:.1f}%</b> of the time — well above random (50%).</span>
            </div>""", unsafe_allow_html=True)

        elif view == "Class Means (NB)":
            # GaussianNB theta_ = class means per feature
            theta_diff = np.abs(gnb.theta_[1] - gnb.theta_[0])
            feat_clean = ['Mean Temp','Max Temp','Min Temp','Wind Speed','Radiation','ET0','Temp Range','Month','Zone']
            sidx2 = np.argsort(theta_diff)
            fig_nb, ax_nb = plt.subplots(figsize=(7, 4.5))
            fig_nb.patch.set_facecolor('white')
            ax_nb.set_facecolor('#fafaf8')
            ax_nb.barh([feat_clean[i] for i in sidx2], theta_diff[sidx2],
                       color=['#2d5a1b' if i==sidx2[-1] else '#7ab648' if i==sidx2[-2]
                              else '#b8d9a0' for i in range(len(theta_diff))],
                       height=0.62, edgecolor='none')
            ax_nb.set_title('Feature Separation Between Classes\n|μ_rainy − μ_dry| per feature', fontsize=11,
                            fontweight='bold', color='#1a1a1a', pad=10)
            ax_nb.set_xlabel('Absolute difference in class means', color='#7a7068', fontsize=9)
            ax_nb.tick_params(colors='#5a5048', labelsize=9)
            for sp3 in ax_nb.spines.values(): sp3.set_edgecolor('#e4dfd8')
            plt.tight_layout()
            st.pyplot(fig_nb, use_container_width=True); plt.close()
            st.caption("Features with higher values are most distinguishable between rainy and dry days in the Naive Bayes model.")

    # Bottom — algorithm explanation
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### How the Algorithms Work")
    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown("""
        <div class='card' style='border-left:5px solid #2d6abf'>
            <div style='font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:#2d6abf;margin-bottom:4px'>DBSCAN</div>
            <div style='font-size:0.75rem;color:#9a9088;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px'>Density-Based Spatial Clustering</div>
            <p style='font-size:0.85rem;color:#5a5048;line-height:1.7'>Identifies clusters as dense regions of points separated by sparse space. Uses two parameters: <b>ε=1.5</b> (neighbourhood radius) and <b>MinPts=3</b>.</p>
            <p style='font-size:0.85rem;color:#5a5048;line-height:1.7'><b>Why not K-Means (taught)?</b> K-Means needs k specified upfront and assumes spherical clusters. DBSCAN discovered 3 zones automatically and correctly labelled 9 cities as "Isolated" — geographically impossible with K-Means.</p>
        </div>""", unsafe_allow_html=True)
    with a2:
        st.markdown(f"""
        <div class='card' style='border-left:5px solid #2d5a1b'>
            <div style='font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:#2d5a1b;margin-bottom:4px'>Gaussian Naive Bayes</div>
            <div style='font-size:0.75rem;color:#9a9088;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px'>Probabilistic Classification</div>
            <p style='font-size:0.85rem;color:#5a5048;line-height:1.7'>Applies Bayes' theorem assuming conditional independence: P(Y|X) ∝ P(X|Y)·P(Y). Models each feature as a Gaussian distribution per class.</p>
            <p style='font-size:0.85rem;color:#5a5048;line-height:1.7'><b>Trained on your data:</b> Accuracy <b>{metrics['accuracy']:.2%}</b> · F1 <b>{metrics['f1']:.4f}</b> · AUC <b>{metrics['auc']:.4f}</b> · var_smoothing = {metrics['best_vs']:.0e}</p>
        </div>""", unsafe_allow_html=True)
