import streamlit as st

def apply_custom_css():
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
