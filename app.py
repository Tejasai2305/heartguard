"""
HeartGuard — Heart Health Checker
Simple, friendly heart health app for everyone.
No medical knowledge needed!
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix)
import warnings, datetime
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="HeartGuard — Free Heart Health Checker",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# CSS — Pure Blue & White | Maximum Readability
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Poppins:wght@600;700;800;900&display=swap');

/* ════════════════════════════════════
   COLOUR TOKENS — Blue & White Only
   ════════════════════════════════════ */
:root {
    --blue:        #1d4ed8;   /* primary blue      */
    --blue-dark:   #1e3a8a;   /* deep navy         */
    --blue-mid:    #3b82f6;   /* medium blue       */
    --blue-light:  #dbeafe;   /* very light blue   */
    --blue-pale:   #eff6ff;   /* near white-blue   */
    --white:       #ffffff;
    --off-white:   #f8faff;
    --text-black:  #0f172a;   /* near black        */
    --text-dark:   #1e293b;   /* dark slate        */
    --text-body:   #1e3a5f;   /* dark navy body    */
    --text-label:  #0f2e5e;   /* form labels       */
    --border:      #93c5fd;   /* blue-300          */
    --border-lt:   #bfdbfe;   /* blue-200          */
    --green:       #15803d;   /* success green     */
    --green-lt:    #dcfce7;
    --green-bdr:   #4ade80;
    --amber:       #b45309;   /* warning amber     */
    --amber-lt:    #fef3c7;
    --amber-bdr:   #fbbf24;
    --red:         #b91c1c;   /* danger red        */
    --red-lt:      #fff1f2;
    --red-bdr:     #fca5a5;
    --orange:      #c2410c;
    --orange-lt:   #fff7ed;
}

/* ════════════════════════════════════
   RESET & BASE
   ════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-black) !important;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--off-white) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* ════════════════════════════════════
   ALL TEXT — FORCE HIGH CONTRAST
   ════════════════════════════════════ */
p, li, span, div, td, th  { color: var(--text-dark); }
strong, b                  { color: var(--text-black) !important; font-weight: 800 !important; }
h1, h2, h3, h4, h5, h6    { color: var(--blue-dark) !important; font-family: 'Poppins', sans-serif !important; font-weight: 800 !important; }
.stMarkdown p              { color: var(--text-body) !important; font-size: 0.97rem !important; line-height: 1.75 !important; }
.stMarkdown li             { color: var(--text-body) !important; font-size: 0.97rem !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: var(--blue-dark) !important; }

/* ════════════════════════════════════
   FORM LABELS — DARK & BOLD
   ════════════════════════════════════ */
label                           { color: var(--text-black) !important; font-weight: 700 !important; font-size: 0.97rem !important; }
.stSlider      > label          { color: var(--text-black) !important; font-weight: 700 !important; font-size: 0.97rem !important; }
.stSelectbox   > label          { color: var(--text-black) !important; font-weight: 700 !important; font-size: 0.97rem !important; }
.stNumberInput > label          { color: var(--text-black) !important; font-weight: 700 !important; font-size: 0.97rem !important; }
.stRadio       > label          { color: var(--text-black) !important; font-weight: 700 !important; font-size: 0.97rem !important; }
.stCheckbox    > label          { color: var(--text-black) !important; font-weight: 700 !important; font-size: 0.97rem !important; }

/* Radio / checkbox options */
.stRadio    div[role="radiogroup"] label { color: var(--text-dark) !important; font-weight: 600 !important; font-size: 0.95rem !important; }
.stCheckbox div[role="checkbox"]   label { color: var(--text-dark) !important; font-weight: 600 !important; font-size: 0.95rem !important; }

/* ════════════════════════════════════
   INPUT WIDGETS
   ════════════════════════════════════ */
[data-baseweb="select"] > div {
    background: var(--white) !important;
    border: 2px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-black) !important;
}
[data-baseweb="select"] span  { color: var(--text-black) !important; font-weight: 600 !important; font-size: 0.95rem !important; }
[data-baseweb="input"]  > div { background: var(--white) !important; border: 2px solid var(--border) !important; border-radius: 12px !important; }
[data-baseweb="input"]  input { color: var(--text-black) !important; font-weight: 600 !important; font-size: 0.97rem !important; }
[role="listbox"]              { background: var(--white) !important; border: 2px solid var(--border) !important; border-radius: 12px !important; }
[role="option"]               { color: var(--text-dark) !important; font-weight: 600 !important; }
[role="option"]:hover         { background: var(--blue-light) !important; color: var(--blue-dark) !important; }
.stSlider [data-baseweb="thumb"]::after { background: var(--blue) !important; }

/* ════════════════════════════════════
   TABS
   ════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--blue-light) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: var(--blue-dark) !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    padding: 0.55rem 1.1rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--white) !important;
    color: var(--blue) !important;
    box-shadow: 0 2px 10px rgba(29,78,216,0.18) !important;
    font-weight: 800 !important;
}

/* ════════════════════════════════════
   PROGRESS BAR
   ════════════════════════════════════ */
.stProgress > div > div {
    background: var(--blue-light) !important;
    border-radius: 999px !important;
    height: 16px !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--blue-dark), var(--blue-mid)) !important;
    border-radius: 999px !important;
}

/* ════════════════════════════════════
   METRICS
   ════════════════════════════════════ */
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 900 !important;
    color: var(--blue-dark) !important;
    font-family: 'Poppins', sans-serif !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.77rem !important;
    font-weight: 700 !important;
    color: var(--text-body) !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
[data-testid="metric-container"] {
    background: var(--white);
    border: 2px solid var(--border-lt);
    border-radius: 18px;
    padding: 1.2rem 1.4rem !important;
    box-shadow: 0 2px 12px rgba(29,78,216,0.09);
    transition: all .2s;
}
[data-testid="metric-container"]:hover {
    border-color: var(--border);
    box-shadow: 0 4px 20px rgba(29,78,216,0.16);
    transform: translateY(-2px);
}

/* ════════════════════════════════════
   ALERTS
   ════════════════════════════════════ */
.stSuccess { background: #f0fdf4 !important; border-color: #4ade80 !important; }
.stSuccess p { color: #14532d !important; font-weight: 700 !important; }
.stInfo    { background: var(--blue-pale) !important; border-color: var(--border) !important; }
.stInfo    p { color: var(--blue-dark) !important; font-weight: 700 !important; }
.stWarning { background: #fffbeb !important; border-color: #fbbf24 !important; }
.stWarning p { color: #78350f !important; font-weight: 700 !important; }
.stError   { background: var(--red-lt) !important; border-color: var(--red-bdr) !important; }
.stError   p { color: var(--red) !important; font-weight: 700 !important; }

/* ════════════════════════════════════
   EXPANDER
   ════════════════════════════════════ */
[data-testid="stExpander"] {
    background: var(--white) !important;
    border: 2px solid var(--border-lt) !important;
    border-radius: 16px !important;
    margin-bottom: 0.65rem !important;
}
[data-testid="stExpander"]:hover { border-color: var(--border) !important; }
[data-testid="stExpander"] summary p {
    color: var(--blue-dark) !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

/* ════════════════════════════════════
   SIDEBAR — Deep Navy Blue
   ════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e3a8a 50%, #1d4ed8 100%) !important;
    border-right: none;
    box-shadow: 5px 0 30px rgba(29,78,216,0.25);
}
/* ALL sidebar text — bright white */
[data-testid="stSidebar"] *                            { color: #ffffff !important; }
[data-testid="stSidebar"] .stRadio > label             { display: none; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(255,255,255,0.1) !important;
    border: 1.5px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    padding: 0.6rem 1rem !important;
    margin-bottom: 0.32rem !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    cursor: pointer;
    transition: all .18s;
    display: block;
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.22) !important;
    border-color: rgba(255,255,255,0.4) !important;
    color: #ffffff !important;
}

/* ════════════════════════════════════
   CARDS — White with blue accents
   ════════════════════════════════════ */
.card {
    background: var(--white);
    border: 2px solid var(--border-lt);
    border-radius: 20px;
    padding: 1.6rem 1.9rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 2px 16px rgba(29,78,216,0.07), 0 1px 4px rgba(0,0,0,0.04);
    transition: all .2s;
    position: relative;
    overflow: hidden;
}
.card::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--blue-dark), var(--blue-mid));
    border-radius: 20px 20px 0 0;
}
.card:hover {
    border-color: var(--border);
    box-shadow: 0 6px 28px rgba(29,78,216,0.14);
    transform: translateY(-2px);
}
.card-sm {
    background: var(--white);
    border: 2px solid var(--border-lt);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: all .2s;
}
.card-sm:hover { border-color: var(--border); box-shadow: 0 4px 16px rgba(29,78,216,0.1); }
.card-title {
    font-size: 0.73rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--blue) !important;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ════════════════════════════════════
   HEADER BANNER — Blue gradient
   ════════════════════════════════════ */
.header-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 40%, #1d4ed8 75%, #2563eb 100%);
    border-radius: 24px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.8rem;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 36px rgba(29,78,216,0.3);
}
.header-banner::before {
    content: '';
    position: absolute; top: -100px; right: -100px;
    width: 300px; height: 300px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.1), transparent 70%);
}
.header-banner::after {
    content: '';
    position: absolute; bottom: -70px; left: 20%;
    width: 220px; height: 220px; border-radius: 50%;
    background: radial-gradient(circle, rgba(147,197,253,0.15), transparent 70%);
}
.header-banner h1 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 2.1rem !important;
    font-weight: 900 !important;
    margin: 0 0 0.45rem 0 !important;
    color: #ffffff !important;
    letter-spacing: -0.02em;
    line-height: 1.2;
    position: relative; z-index: 1;
    text-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.header-banner p {
    font-size: 1.05rem;
    color: #bfdbfe;
    margin: 0;
    font-weight: 600;
    line-height: 1.6;
    position: relative; z-index: 1;
}
.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1.5px solid rgba(255,255,255,0.4);
    border-radius: 999px;
    padding: 0.25rem 0.95rem;
    font-size: 0.73rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    position: relative; z-index: 1;
}

/* ════════════════════════════════════
   RISK BADGES
   ════════════════════════════════════ */
.risk-low    { background: var(--green-lt); color: var(--green);    border: 2px solid var(--green-bdr); }
.risk-medium { background: var(--amber-lt); color: var(--amber);    border: 2px solid var(--amber-bdr); }
.risk-high   { background: var(--red-lt);   color: var(--red);      border: 2px solid var(--red-bdr);   }
.risk-badge  {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.55rem 1.4rem; border-radius: 999px;
    font-weight: 800; font-size: 1.08rem;
}

/* ════════════════════════════════════
   BUTTONS — Blue
   ════════════════════════════════════ */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--blue-dark) 0%, var(--blue) 100%);
    color: #ffffff !important;
    border: none;
    padding: 0.78rem 2rem;
    font-size: 1.02rem;
    font-weight: 800;
    border-radius: 14px;
    cursor: pointer;
    transition: all .2s;
    width: 100%;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 4px 18px rgba(29,78,216,0.35);
    letter-spacing: 0.02em;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 32px rgba(29,78,216,0.5);
    background: linear-gradient(135deg, #172554 0%, var(--blue-dark) 100%);
}

/* ════════════════════════════════════
   CONTENT TEXT BOXES
   ════════════════════════════════════ */
.tip-box {
    background: var(--green-lt);
    border: 2px solid var(--green-bdr);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    font-size: 0.93rem;
    color: var(--green) !important;
    line-height: 1.75;
    margin-bottom: 0.9rem;
    font-weight: 700;
}
.tip-box b, .tip-box strong { color: #14532d !important; }

.warn-box {
    background: var(--red-lt);
    border: 2px solid var(--red-bdr);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    font-size: 0.93rem;
    color: var(--red) !important;
    line-height: 1.75;
    margin-bottom: 0.9rem;
    font-weight: 700;
}
.warn-box b, .warn-box strong { color: #7f1d1d !important; }

.info-box {
    background: var(--blue-pale);
    border: 2px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    font-size: 0.93rem;
    color: var(--blue-dark) !important;
    line-height: 1.75;
    margin-bottom: 0.9rem;
    font-weight: 700;
}
.info-box b, .info-box strong { color: #172554 !important; }

.disclaimer {
    background: var(--amber-lt);
    border-left: 5px solid var(--amber-bdr);
    padding: 1rem 1.25rem;
    border-radius: 0 14px 14px 0;
    font-size: 0.9rem;
    color: var(--amber) !important;
    margin-top: 0.9rem;
    line-height: 1.75;
    font-weight: 700;
}
.disclaimer b, .disclaimer strong { color: #78350f !important; }

/* ════════════════════════════════════
   LIST & ROW ITEMS
   ════════════════════════════════════ */
.rec-item {
    padding: 0.6rem 0;
    border-bottom: 1.5px solid var(--blue-light);
    font-size: 0.92rem;
    line-height: 1.72;
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    font-weight: 600;
    color: var(--text-body) !important;
}
.rec-item:last-child { border-bottom: none; }

.hist-row {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 0.75rem 0.5rem;
    border-bottom: 1.5px solid var(--blue-light);
    font-size: 0.9rem; gap: 1rem;
    font-weight: 600;
    color: var(--text-dark) !important;
    border-radius: 8px;
    transition: background .15s;
}
.hist-row:hover    { background: var(--blue-pale); }
.hist-row:last-child { border-bottom: none; }

/* ════════════════════════════════════
   TAGS & PILLS
   ════════════════════════════════════ */
.feat-tag {
    display: inline-block;
    background: var(--blue-light);
    color: var(--blue-dark) !important;
    border: 1.5px solid var(--border);
    border-radius: 8px;
    padding: 0.18rem 0.65rem;
    font-size: 0.8rem;
    font-weight: 700;
}
.stat-pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: var(--blue-pale);
    border: 2px solid var(--border-lt);
    border-radius: 12px;
    padding: 0.38rem 0.85rem;
    font-size: 0.84rem;
    font-weight: 700;
    color: var(--blue-dark) !important;
    margin: 0.2rem;
}

/* ════════════════════════════════════
   STEP CIRCLE
   ════════════════════════════════════ */
.step-circle {
    width: 42px; height: 42px; border-radius: 50%;
    background: linear-gradient(135deg, var(--blue-dark), var(--blue-mid));
    color: #ffffff !important; font-weight: 900; font-size: 1.05rem;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 0.65rem;
    box-shadow: 0 6px 18px rgba(29,78,216,0.4);
}

/* ════════════════════════════════════
   NAV SECTION LABELS (sidebar)
   ════════════════════════════════════ */
.nav-section {
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #93c5fd !important;
    padding: 0.9rem 0 0.35rem 0.1rem;
}

/* ════════════════════════════════════
   QUIZ & ANSWER CARDS
   ════════════════════════════════════ */
.answer-card {
    background: var(--white);
    border: 2px solid var(--border-lt);
    border-radius: 16px;
    padding: 0.88rem 1.15rem;
    margin-bottom: 0.55rem;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-dark) !important;
    transition: border-color .18s;
}
.answer-card:hover { border-color: var(--border); }
.answer-ok  { background: var(--green-lt) !important; border-color: var(--green-bdr) !important; color: var(--green) !important; }
.answer-err { background: var(--red-lt)   !important; border-color: var(--red-bdr)   !important; color: var(--red)   !important; }

/* ════════════════════════════════════
   PLAN CARDS
   ════════════════════════════════════ */
.plan-week {
    background: var(--white);
    border: 2px solid var(--border-lt);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
}
.plan-week::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 6px; height: 100%;
    background: linear-gradient(180deg, var(--blue-dark), var(--blue-mid));
}
.plan-day {
    display: flex; gap: 0.8rem; padding: 0.46rem 0;
    border-bottom: 1.5px solid var(--blue-light);
    font-size: 0.9rem; align-items: flex-start;
    font-weight: 600; color: var(--text-body) !important;
}
.plan-day:last-child { border-bottom: none; }

/* ════════════════════════════════════
   DIVIDER
   ════════════════════════════════════ */
.divider {
    border: none;
    border-top: 2px solid var(--blue-light);
    margin: 1.5rem 0;
}

/* ════════════════════════════════════
   SCROLLBAR
   ════════════════════════════════════ */
::-webkit-scrollbar               { width: 7px; height: 7px; }
::-webkit-scrollbar-track         { background: var(--blue-pale); border-radius: 4px; }
::-webkit-scrollbar-thumb         { background: var(--border);    border-radius: 4px; }
::-webkit-scrollbar-thumb:hover   { background: var(--blue);      }

/* ════════════════════════════════════
   SELECTION
   ════════════════════════════════════ */
::selection { background: var(--blue-light); color: var(--blue-dark); }

/* ════════════════════════════════════
   HEART SCORE BAR
   ════════════════════════════════════ */
.heart-score-bar {
    height: 22px; border-radius: 999px;
    background: linear-gradient(90deg, #16a34a, #d97706, #b91c1c);
    margin: 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
_DEFAULTS = {
    "history":        [],
    "last_result":    None,
    "quiz_state":     {"current": 0, "answers": {}, "done": False},
    "active_page":    "Home",
    "prev_core":      "Home",
    "prev_adv":       "BMI & Weight Check",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────────────────────────
# ML ENGINE
# ──────────────────────────────────────────────
FEATURES = [
    "Age", "Sex", "Chest Discomfort", "Resting Blood Pressure",
    "Cholesterol", "Blood Sugar After Fasting", "Heart Scan (ECG)",
    "Peak Heart Rate", "Chest Pain on Walking", "ST Drop (mm)", "ST Slope",
]

@st.cache_resource(show_spinner=False)
def train_models():
    np.random.seed(42)
    n = 2000

    age = np.random.randint(29, 78, n)
    sex = np.random.randint(0, 2, n)
    chest_pain = np.random.randint(0, 4, n)
    resting_bp = np.random.randint(90, 200, n)
    cholesterol = np.random.randint(130, 570, n)
    fasting_bs = np.random.randint(0, 2, n)
    rest_ecg = np.random.randint(0, 3, n)
    max_hr = np.random.randint(70, 210, n)
    exer_angina = np.random.randint(0, 2, n)
    oldpeak = np.round(np.random.uniform(0, 6, n), 1)
    st_slope = np.random.randint(0, 3, n)

    score = (
        0.045 * age + 0.35 * sex + 0.28 * chest_pain
        + 0.012 * resting_bp + 0.003 * cholesterol
        + 0.22  * fasting_bs - 0.012 * max_hr
        + 0.32  * exer_angina + 0.28 * oldpeak - 0.22 * st_slope
        + np.random.normal(0, 0.5, n)
    )
    target = (score > score.mean()).astype(int)

    X = np.column_stack([age, sex, chest_pain, resting_bp, cholesterol,
                         fasting_bs, rest_ecg, max_hr, exer_angina, oldpeak, st_slope])
    X_tr, X_te, y_tr, y_te = train_test_split(X, target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)

    models = {
        "Smart Check (Recommended)": RandomForestClassifier(n_estimators=150, random_state=42),
        "Deep Check":                GradientBoostingClassifier(n_estimators=150, random_state=42),
        "Quick Check":               LogisticRegression(max_iter=1000, random_state=42),
    }
    for m in models.values(): m.fit(Xtr, y_tr)

    def mets(m):
        yp = m.predict(Xte)
        return {
            "Accuracy":  round(accuracy_score(y_te, yp)  * 100, 1),
            "Precision": round(precision_score(y_te, yp) * 100, 1),
            "Recall":    round(recall_score(y_te, yp)    * 100, 1),
            "F1":        round(f1_score(y_te, yp)        * 100, 1),
        }

    metrics = {n: mets(m) for n, m in models.items()}
    rf = models["Smart Check (Recommended)"]
    imp = rf.feature_importances_
    roc_data = {}
    for name, m in models.items():
        pp = m.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, pp)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": round(auc(fpr, tpr), 3)}
    cm = confusion_matrix(y_te, rf.predict(Xte))

    return {
        "models":  models,
        "scaler":  scaler,
        "metrics": metrics,
        "roc":     roc_data,
        "cm":      cm,
        "imp":     imp,
        "X_full":  X,
        "y_full":  target,
    }


def predict(M, name, inputs):
    arr = np.array(inputs, dtype=float).reshape(1, -1)
    return M["models"][name].predict_proba(M["scaler"].transform(arr))[0][1]


def risk_label(prob):
    if prob < 0.35:  return "Low Risk 😊",      "risk-low",    "💚", "#16a34a"
    if prob < 0.65:  return "Medium Risk 😐",   "risk-medium", "💛", "#ca8a04"
    return               "High Risk ⚠️",          "risk-high",   "❤️", "#e11d48"


def get_tips(prob):
    if prob >= 0.65:
        L = [
            "Please visit a doctor or heart specialist soon",
            "Check your blood pressure every day",
            "Eat less salt, oil, and fried food",
            "Stop smoking if you smoke — this is the most important step",
            "Walk slowly for 10 minutes every day and slowly increase",
            "Try to sleep 7–8 hours every night",
            "Reduce stress — try deep breathing or prayer/meditation",
        ]
        M = [
            "Your doctor may prescribe blood pressure medicines",
            "Cholesterol-lowering medicines (statins) may be needed",
            "Blood thinners may be recommended by your doctor",
        ]
    elif prob >= 0.35:
        L = [
            "Visit your doctor for a check-up in the next month",
            "Eat more fruits, vegetables, and less oily food",
            "Walk 30 minutes every day, 5 days a week",
            "Drink less sugary drinks and more water",
            "If you smoke, try to reduce or quit",
            "Reduce stress with hobbies, family time, or prayer",
        ]
        M = [
            "Your doctor may suggest fish oil tablets for heart health",
            "Ask about whether you need cholesterol checks",
        ]
    else:
        L = [
            "Keep up your great healthy habits!",
            "Continue exercising regularly",
            "Eat balanced meals with plenty of vegetables",
            "Get a heart check-up once a year",
            "Keep your weight in a healthy range",
        ]
        M = ["No medicines needed right now — keep living healthy!"]
    return L, M


# ──────────────────────────────────────────────
# CHART HELPERS
# ──────────────────────────────────────────────
def _ax(ax):
    ax.set_facecolor("#f0f9ff")
    fig = ax.get_figure()
    fig.patch.set_facecolor("#ffffff")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#bae6fd")
    ax.spines["bottom"].set_color("#bae6fd")
    ax.tick_params(colors="#64748b", labelsize=8.5)


def ch_gauge(prob):
    fig, ax = plt.subplots(figsize=(4.5, 3.2), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    t = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(t), np.sin(t), lw=26, color="#dbeafe", solid_capstyle="round")
    for lo, hi, col in [(0, .35, "#dcfce7"), (.35, .65, "#fef3c7"), (.65, 1, "#fff1f2")]:
        seg = np.linspace(np.pi - lo * np.pi, np.pi - hi * np.pi, 100)
        ax.plot(np.cos(seg), np.sin(seg), lw=26, color=col, solid_capstyle="butt", alpha=.9)
    fc = "#16a34a" if prob < .35 else ("#d97706" if prob < .65 else "#e11d48")
    ft = np.linspace(np.pi, np.pi - prob * np.pi, 300)
    ax.plot(np.cos(ft), np.sin(ft), lw=26, color=fc, solid_capstyle="round", alpha=.95)
    ang = np.pi - prob * np.pi
    ax.annotate("", xy=(.72 * np.cos(ang), .72 * np.sin(ang)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#0c1a2e", lw=2.5, mutation_scale=15))
    ax.plot(0, 0, "o", color="#0c1a2e", ms=9, zorder=5)
    ax.text(0, .28, f"{prob*100:.0f}%", ha="center", va="center",
            fontsize=24, fontweight="900", color="#0c1a2e")
    ax.text(0, .08, "Your Heart Risk", ha="center", va="center",
            fontsize=9, color="#64748b", fontweight="700")
    for v, lbl, c in [(0, "Safe", "#16a34a"), (.5, "Watch", "#d97706"), (1, "Risk", "#e11d48")]:
        a2 = np.pi - v * np.pi
        ax.text(1.22 * np.cos(a2), 1.22 * np.sin(a2), lbl,
                ha="center", va="center", fontsize=8, color=c, fontweight="800")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-.35, 1.4); ax.axis("off")
    plt.tight_layout(pad=.3); return fig


def ch_top_factors(imp):
    idx  = np.argsort(imp)[::-1][:6]
    vals = imp[idx]
    labels_friendly = {
        "Age":                     "Your Age",
        "Sex":                     "Gender",
        "Chest Discomfort":        "Chest Discomfort",
        "Resting Blood Pressure":  "Blood Pressure",
        "Cholesterol":             "Cholesterol Level",
        "Blood Sugar After Fasting":"Blood Sugar",
        "Heart Scan (ECG)":        "Heart Scan Result",
        "Peak Heart Rate":         "Heart Rate",
        "Chest Pain on Walking":   "Pain on Walking",
        "ST Drop (mm)":            "Heart Signal Drop",
        "ST Slope":                "Heart Signal Slope",
    }
    labs = [labels_friendly.get(FEATURES[i], FEATURES[i]) for i in idx]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    fig.patch.set_facecolor("white"); _ax(ax)
    colors = ["#1d4ed8" if v == max(vals) else
              ("#3b82f6" if v >= np.percentile(vals, 66) else "#93c5fd") for v in vals]
    ax.barh(labs[::-1], vals[::-1], color=colors[::-1], height=.6, edgecolor="none")
    for i, v in enumerate(vals[::-1]):
        ax.text(v + .002, i, f"{v*100:.0f}%", va="center",
                fontsize=9, color="#64748b", fontweight="700")
    ax.set_xlabel("How much this affects your result", fontsize=9, color="#64748b")
    ax.set_title("What Matters Most For Your Heart", fontsize=11,
                 fontweight="700", color="#0c1a2e", pad=10)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0, colors="#334155")
    plt.tight_layout(); return fig


def ch_history(history):
    dates = [h["time"] for h in history]
    probs = [h["prob"] * 100 for h in history]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("white"); _ax(ax)
    ax.fill_between(range(len(probs)), probs, alpha=.12, color="#1d4ed8")
    ax.plot(range(len(probs)), probs, "o-", color="#1d4ed8",
            lw=2.8, ms=9, markerfacecolor="white", markeredgewidth=2.8,
            markeredgecolor="#1d4ed8")
    for y, c, lbl in [(35, "#16a34a", "Safe Zone"), (65, "#e11d48", "Risky Zone")]:
        ax.axhline(y, color=c, lw=1.5, ls="--", alpha=.6)
        ax.text(len(probs) - .1, y + 1.5, lbl, fontsize=8.5, color=c,
                ha="right", fontweight="800")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d.strftime("%b %d\n%H:%M") for d in dates],
                       fontsize=8.5, rotation=30, ha="right", color="#64748b")
    ax.set_ylabel("Your Risk %", fontsize=9, color="#64748b")
    ax.set_ylim(0, 108)
    ax.set_title("Your Risk Score Over Time", fontsize=11,
                 fontweight="700", color="#0c1a2e", pad=10)
    plt.tight_layout(); return fig


def ch_bmi(bmi):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white"); ax.axis("off")
    zones = [
        (10, 18.5, "#bfdbfe", "Too Thin"),
        (18.5, 24.9, "#99f6e4", "Healthy ✓"),
        (24.9, 29.9, "#fef08a", "A Bit Heavy"),
        (29.9, 45,   "#fecaca", "Too Heavy"),
    ]
    for lo, hi, col, lbl in zones:
        w = (hi - lo) / 35; s = (lo - 10) / 35
        ax.barh(0, w, left=s, height=.55, color=col, edgecolor="white", linewidth=2.5)
        ax.text(s + w / 2, -.42, lbl, ha="center", fontsize=8.5,
                color="#334155", fontweight="700")
    pos = min(max((bmi - 10) / 35, 0), 1)
    ax.annotate("", xy=(pos, .28), xytext=(pos, .6),
                arrowprops=dict(arrowstyle="-|>", color="#1d4ed8", lw=2.5, mutation_scale=14))
    ax.text(pos, .76, f"Your BMI: {bmi:.1f}", ha="center",
            fontsize=10.5, fontweight="900", color="#1d4ed8")
    ax.set_xlim(0, 1); ax.set_ylim(-.7, 1)
    plt.tight_layout(); return fig


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
def sidebar_nav():
    with st.sidebar:
        st.markdown("""
        <div style='padding:.3rem 0 .6rem 0;'>
            <div style='font-size:1.6rem;font-weight:900;color:white;
                        font-family:Sora,sans-serif;letter-spacing:-.02em;'>
                💙 HeartGuard
            </div>
            <div style='font-size:.75rem;color:#bae6fd;margin-top:.2rem;font-weight:600;
                        letter-spacing:.03em;'>
                Free Heart Health Checker — For Everyone
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1e3a8a;margin:.8rem 0'>",
                    unsafe_allow_html=True)

        st.markdown("<div class='nav-section'>🏠 Main</div>", unsafe_allow_html=True)
        core = st.radio("_core", [
            "🏠  Home",
            "❤️  Check My Heart",
            "📋  My Results History",
            "ℹ️  How It Works",
        ], label_visibility="collapsed", key="_core_r")

        st.markdown("<div class='nav-section'>🛠️ Health Tools</div>",
                    unsafe_allow_html=True)
        adv = st.radio("_adv", [
            "⚖️  BMI & Weight Check",
            "💓  Heart Rate Zones",
            "🩸  Blood Pressure Guide",
            "🍎  Healthy Eating Tips",
            "🚶  Exercise for Beginners",
            "😴  Sleep & Stress Guide",
            "🧠  Heart Health Quiz",
            "📅  My Health Plan",
            "❓  Common Questions",
        ], label_visibility="collapsed", key="_adv_r")

        n = len(st.session_state.history)
        st.markdown("<hr style='border-color:#1e3a8a;margin:1rem 0'>",
                    unsafe_allow_html=True)
        if n > 0:
            st.markdown(
                f"<div style='font-size:.78rem;color:#ffffff;margin-bottom:.4rem;font-weight:700;'>"
                f"✅ {n} check(s) completed this session</div>",
                unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:.75rem;color:#dbeafe;line-height:1.9;font-weight:600;'>
        100% Free · No Sign-up · Private<br>
        Works on Phone & Computer
        </div>
        <div style='background:rgba(255,255,255,0.12);border-left:3px solid #93c5fd;
                    border-radius:0 8px 8px 0;padding:.6rem .8rem;margin-top:.8rem;
                    font-size:.72rem;color:#ffffff;font-weight:600;'>
        ⚠️ This is NOT a replacement for a real doctor visit.
        </div>
        """, unsafe_allow_html=True)

    cn = core.split("  ")[-1].strip()
    an = adv.split("  ")[-1].strip()

    if cn != st.session_state.prev_core:
        st.session_state.prev_core   = cn
        st.session_state.active_page = cn
    elif an != st.session_state.prev_adv:
        st.session_state.prev_adv    = an
        st.session_state.active_page = an

    return st.session_state.active_page


# ══════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div class='header-banner'>
        <div class='header-badge'>100% Free · No Sign-up Required · Safe & Private</div>
        <h1>❤️ HeartGuard — Check Your Heart Health</h1>
        <p>Answer a few simple questions and find out how healthy your heart is.
           Takes only 2 minutes. No medical knowledge needed!</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("✅ Free",        "Always",   help="No payment ever")
    with c2: st.metric("⏱️ Time Needed", "2 Minutes")
    with c3: st.metric("🔒 Private",     "100%",     help="Nothing is stored")
    with c4: st.metric("📱 Works On",    "Any Device")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # How to use
    st.markdown("### 👋 How to Use HeartGuard")
    s1, s2, s3, s4 = st.columns(4)
    for col, num, icon, title, desc in [
        (s1, "1", "📝", "Answer Questions", "Tell us basic things about yourself like age, weight, and if you feel chest pain"),
        (s2, "2", "🤖", "AI Checks",        "Our computer programme looks at your answers and checks for warning signs"),
        (s3, "3", "📊", "See Your Result",  "You get a simple score — Green (safe), Yellow (be careful), or Red (see doctor)"),
        (s4, "4", "💡", "Get Advice",       "We give you easy tips on what to eat, do, and whether to see a doctor"),
    ]:
        with col:
            st.markdown(f"""
            <div class='card' style='text-align:center;min-height:160px;'>
                <div class='step-circle'>{num}</div>
                <div style='font-size:1.5rem;margin-bottom:.3rem;'>{icon}</div>
                <div style='font-weight:800;font-size:.95rem;color:#0c1a2e;margin-bottom:.3rem;'>{title}</div>
                <div style='font-size:.82rem;color:#334155;line-height:1.55;font-weight:600;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Who can use it
    st.markdown("### 🧑‍🤝‍🧑 Who Can Use This?")
    people = [
        ("👴", "Older Adults", "Check your heart health as you age"),
        ("👨‍👩‍👧", "Families",    "Check on your parents or loved ones"),
        ("🏢", "Office Workers","Long sitting hours affect heart health"),
        ("🏃", "Fitness Lovers","Make sure your heart keeps up with exercise"),
        ("🤰", "Anyone Really", "Heart health matters for everyone"),
        ("🏥", "Caregivers",   "Help patients understand their risk"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(people):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='card-sm' style='display:flex;gap:.75rem;align-items:flex-start;'>
                <span style='font-size:1.6rem;'>{icon}</span>
                <div>
                    <div style='font-weight:800;font-size:.92rem;color:#0c1a2e;'>{title}</div>
                    <div style='font-size:.82rem;color:#334155;margin-top:.12rem;font-weight:600;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Warning signs section
    st.markdown("### ⚠️ Warning Signs of Heart Problems")
    st.markdown("""
    <div class='warn-box'>
    🚨 <b>Go to Emergency (999/112) RIGHT NOW if you have:</b><br>
    • Chest pain, pressure, or tightness that doesn't go away<br>
    • Pain spreading to your left arm, jaw, neck, or back<br>
    • Sudden breathlessness at rest<br>
    • Feeling faint, dizzy, or cold sweat with chest pain<br>
    • Heart beating very fast and irregularly
    </div>
    """, unsafe_allow_html=True)

    signs_col1, signs_col2 = st.columns(2)
    with signs_col1:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>🔴 See a Doctor Soon (Within 1 Week)</div>
            <div style='font-size:.88rem;line-height:1.8;color:#1e3a5f;font-weight:600;'>
            • Chest discomfort when climbing stairs or walking<br>
            • Gets tired very easily doing simple tasks<br>
            • Feet or ankles are swollen<br>
            • Breathless when lying flat at night<br>
            • Heart skipping beats regularly
            </div>
        </div>
        """, unsafe_allow_html=True)
    with signs_col2:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>🟡 Worth Mentioning at Your Next Check-up</div>
            <div style='font-size:.88rem;line-height:1.8;color:#1e3a5f;font-weight:600;'>
            • Mild tiredness more than usual<br>
            • Occasional mild chest tightness that passes quickly<br>
            • Blood pressure reading above 130/80 at home<br>
            • Family member had a heart attack before age 60<br>
            • You smoke or recently quit smoking
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='disclaimer'>
    ⚠️ <b>Important:</b> HeartGuard is an educational tool only. It is NOT a doctor and
    cannot diagnose any disease. Always visit a real doctor for health concerns.
    This tool is completely free and stores nothing about you.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: CHECK MY HEART
# ══════════════════════════════════════════════
def page_check(M):
    st.markdown("""
    <div class='header-banner'>
        <h1>❤️ Check My Heart Health</h1>
        <p>Answer these simple questions honestly. There are no wrong answers — just be truthful!</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    💡 <b>Don't know a value?</b> That's okay! Use the default value shown.
    You can ask your doctor or chemist to check your blood pressure and cholesterol for free.
    </div>
    """, unsafe_allow_html=True)

    # Check type
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>⚙️ Choose How Thorough You Want the Check</div>",
                unsafe_allow_html=True)
    check_type = st.radio(
        "check",
        ["🚀 Quick Check (5 questions — takes 1 minute)",
         "🔍 Full Check (all questions — takes 2 minutes, more accurate)"],
        label_visibility="collapsed"
    )
    full_check = "Full" in check_type
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        # Basic Info
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>👤 About You</div>", unsafe_allow_html=True)

        age = st.slider(
            "How old are you?",
            20, 80, 45,
            help="Your age in years"
        )
        sex = st.selectbox(
            "What is your biological sex?",
            ["Male", "Female"],
            help="This affects heart disease risk patterns"
        )
        sv = 1 if sex == "Male" else 0

        chest_opts = {
            "No chest pain or discomfort": 3,
            "Mild chest tightness sometimes": 2,
            "Chest pain when I exercise or stress": 1,
            "Regular chest pain (typical angina)": 0,
        }
        cp_lbl = st.selectbox(
            "Do you ever feel chest pain or discomfort?",
            list(chest_opts.keys()),
            help="Be honest — this is very important"
        )
        cp = chest_opts[cp_lbl]

        ea_lbl = st.selectbox(
            "Do you feel chest pain or tightness when you walk fast or climb stairs?",
            ["No, I feel fine", "Yes, sometimes"],
            help="This can be a sign of reduced blood flow to the heart"
        )
        ea = 1 if "Yes" in ea_lbl else 0
        st.markdown("</div>", unsafe_allow_html=True)

        # Heart measures
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>💓 Your Heart Numbers</div>", unsafe_allow_html=True)

        rbp = st.slider(
            "What is your blood pressure? (top number, in mmHg)",
            80, 200, 120,
            help="Normal is below 120. Check with a chemist if unsure."
        )

        if rbp < 120:
            st.markdown("<div class='tip-box'>✅ Great! Your blood pressure looks normal.</div>",
                        unsafe_allow_html=True)
        elif rbp < 140:
            st.markdown("<div class='warn-box'>⚠️ Slightly high. Try reducing salt in your food.</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='warn-box'>🔴 High blood pressure — please see a doctor.</div>",
                        unsafe_allow_html=True)

        chol = st.slider(
            "What is your cholesterol level? (mg/dL)",
            100, 600, 200,
            help="Normal is below 200. Your doctor or chemist can test this."
        )

        if full_check:
            hr = st.slider(
                "What is the highest heart rate you reach when exercising? (beats per minute)",
                60, 220, 150,
                help="You can check this with a fitness band or smartwatch"
            )
            op = st.slider(
                "Have you had an ECG (heart scan)? What was the ST drop reading? (mm)",
                0.0, 6.5, 1.0, step=0.1,
                help="Leave at 1.0 if you don't know — your doctor will have this"
            )
        else:
            hr = 150
            op = 1.0
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if full_check:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>🩺 A Few More Questions</div>",
                        unsafe_allow_html=True)

            fbs_lbl = st.selectbox(
                "Was your blood sugar high on your last fasting test? (above 7 mmol/L or 126 mg/dL)",
                ["No / I don't know", "Yes — my blood sugar was high"],
                help="This checks for diabetes risk"
            )
            fbs = 1 if "Yes" in fbs_lbl else 0

            ecg_opts = {
                "Normal / Never had one": 0,
                "Doctor said there was an abnormal wave (ST-T change)": 1,
                "Doctor said my heart was enlarged (LV Hypertrophy)": 2,
            }
            ecg_lbl = st.selectbox(
                "What did your heart scan (ECG) show? (if you had one)",
                list(ecg_opts.keys()),
                help="If you've never had an ECG, choose Normal"
            )
            ecg = ecg_opts[ecg_lbl]

            slope_opts = {
                "Normal / Don't know": 0,
                "Flat line during exercise": 1,
                "Going down during exercise (downsloping)": 2,
            }
            slope_lbl = st.selectbox(
                "During an exercise test, what did your heart signal (ST slope) do?",
                list(slope_opts.keys()),
                help="Leave as Normal if you haven't had this test"
            )
            sl = slope_opts[slope_lbl]
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            fbs = 0; ecg = 0; sl = 0

        # Summary
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📝 Your Summary</div>", unsafe_allow_html=True)
        pairs = [
            ("Age",             f"{age} years"),
            ("Sex",             sex),
            ("Chest Pain",      cp_lbl.split("(")[0][:30]),
            ("Blood Pressure",  f"{rbp} mmHg"),
            ("Cholesterol",     f"{chol} mg/dL"),
            ("Pain on Walking", "Yes" if ea == 1 else "No"),
        ]
        for k, v in pairs:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:.28rem 0;border-bottom:1px solid #fde8e8;font-size:.87rem;'>"
                f"<span style='color:#334155;font-weight:600;'>{k}</span>"
                f"<span style='font-weight:800;color:#0c1a2e;'>{v}</span></div>",
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='tip-box'>
        💡 <b>Tip:</b> The more accurately you answer, the more useful your result will be!
        </div>
        """, unsafe_allow_html=True)

    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn, _ = st.columns([2, 1.2, 2])
    with btn:
        clicked = st.button("🔍 Check My Heart Now!", use_container_width=True)

    if clicked:
        inputs = [age, sv, cp, rbp, chol, fbs, ecg, hr, ea, op, sl]
        with st.spinner("🤖 Checking your heart health... please wait a moment..."):
            import time; time.sleep(0.5)
            prob             = predict(M, "Smart Check (Recommended)", inputs)
            label, css, em, rc = risk_label(prob)
            life, meds       = get_tips(prob)

        entry = {
            "prob": prob, "label": label, "inputs": inputs,
            "lifestyle": life, "meds": meds,
            "time": datetime.datetime.now(),
            "input_map": dict(zip(FEATURES, inputs)),
        }
        st.session_state.last_result = entry
        st.session_state.history.append(entry)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("## 📊 Your Heart Health Result")

        # Result cards
        r1, r2, r3 = st.columns(3)
        with r1: st.metric("Your Risk Score", f"{prob*100:.0f}%")
        with r2: st.metric("Risk Level",      label.split(" ")[0] + " " + label.split(" ")[1])
        with r3: st.metric("What To Do",
                           "See Doctor" if prob >= 0.65 else
                           "Stay Careful" if prob >= 0.35 else "Keep It Up!")

        rl, rr = st.columns([1.1, 1], gap="large")

        with rl:
            # Big result card
            if prob < 0.35:
                bg, border_c, msg = "#f0fdf4", "#86efac", "Your heart looks healthy! Keep up the good work and have yearly check-ups."
            elif prob < 0.65:
                bg, border_c, msg = "#fefce8", "#fde047", "Your heart needs some attention. Making lifestyle changes now can help a lot."
            else:
                bg, border_c, msg = "#fff1f2", "#fda4af", "Your heart is showing warning signs. Please visit a doctor as soon as possible."

            st.markdown(f"""
            <div style='background:{bg};border:3px solid {border_c};border-radius:20px;
                        padding:1.8rem;text-align:center;margin-bottom:1rem;'>
                <div style='font-size:2.5rem;margin-bottom:.4rem;'>{em}</div>
                <span class='risk-badge {css}'>{label}</span>
                <div style='margin-top:1rem;font-size:.95rem;color:#1e3a5f;
                            font-weight:700;line-height:1.6;'>{msg}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom:.5rem;font-size:.88rem;color:#334155;font-weight:700;'>Your risk meter:</div>",
                        unsafe_allow_html=True)
            st.progress(prob)

            # Simple colour explanation
            st.markdown("""
            <div class='card' style='margin-top:.8rem;'>
                <div class='card-title'>🎨 What the Colours Mean</div>
                <div style='display:flex;flex-direction:column;gap:.5rem;'>
                    <div style='display:flex;align-items:center;gap:.7rem;font-weight:700;font-size:.88rem;'>
                        <span style='background:#dcfce7;border-radius:8px;padding:.2rem .6rem;color:#14532d;border:2px solid #86efac;'>💚 Green (0–35%)</span>
                        <span style='color:#1e3a5f;'>Your heart looks safe — keep going!</span>
                    </div>
                    <div style='display:flex;align-items:center;gap:.7rem;font-weight:700;font-size:.88rem;'>
                        <span style='background:#fef9c3;border-radius:8px;padding:.2rem .6rem;color:#713f12;border:2px solid #fde047;'>💛 Yellow (35–65%)</span>
                        <span style='color:#1e3a5f;'>Be careful — make some changes</span>
                    </div>
                    <div style='display:flex;align-items:center;gap:.7rem;font-weight:700;font-size:.88rem;'>
                        <span style='background:#fff1f2;border-radius:8px;padding:.2rem .6rem;color:#9f1239;border:2px solid #fda4af;'>❤️ Red (65–100%)</span>
                        <span style='color:#1e3a5f;'>See a doctor soon</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with rr:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>🎯 Your Heart Risk Meter</div>",
                        unsafe_allow_html=True)
            st.pyplot(ch_gauge(prob), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>📊 What Affects Your Result Most</div>",
                        unsafe_allow_html=True)
            st.pyplot(ch_top_factors(M["imp"]), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Tips
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("## 💡 What You Should Do")

        t1, t2 = st.columns(2, gap="large")
        with t1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>🥗 Simple Lifestyle Tips for You</div>",
                        unsafe_allow_html=True)
            for tip in life:
                st.markdown(
                    f"<div class='rec-item'>"
                    f"<span style='font-size:1.1rem;'>✅</span>"
                    f"<span style='color:#1e3a5f;'>{tip}</span></div>",
                    unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with t2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>💊 What Your Doctor Might Suggest</div>",
                        unsafe_allow_html=True)
            for med in meds:
                st.markdown(
                    f"<div class='rec-item'>"
                    f"<span style='font-size:1.1rem;'>💊</span>"
                    f"<span style='color:#1e3a5f;'>{med}</span></div>",
                    unsafe_allow_html=True)
            st.markdown("""
            <div class='disclaimer'>
            ⚠️ <b>Important:</b> Never take medicines without asking a real doctor first.
            These are just general suggestions — everyone is different!
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='tip-box' style='text-align:center;margin-top:.5rem;'>
        💾 Your result has been saved to <b>My Results History</b> so you can track your progress over time!
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════════
def page_history():
    st.markdown("""
    <div class='header-banner'>
        <h1>📋 My Results History</h1>
        <p>See all your past heart health checks in one place and track your progress.</p>
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.history
    if not history:
        st.markdown("""
        <div class='card' style='text-align:center;padding:3rem;'>
            <div style='font-size:3rem;margin-bottom:.9rem;'>📭</div>
            <div style='font-size:1.1rem;font-weight:800;color:#0c1a2e;'>No checks done yet!</div>
            <div style='font-size:.92rem;color:#334155;margin-top:.4rem;font-weight:600;'>
                Go to <b>Check My Heart</b> and answer the questions to get your first result.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    probs = [h["prob"] for h in history]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Checks",    len(history))
    with c2: st.metric("Average Risk",    f"{np.mean(probs)*100:.0f}%")
    with c3: st.metric("Best Score",      f"{min(probs)*100:.0f}%", delta="Lowest risk")
    with c4: st.metric("Latest Score",    f"{probs[-1]*100:.0f}%")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if len(history) >= 2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📈 How Your Risk Has Changed Over Time</div>",
                    unsafe_allow_html=True)
        st.pyplot(ch_history(history), use_container_width=True)
        if probs[-1] < probs[0]:
            st.markdown("""<div class='tip-box'>🎉 Great news! Your risk has gone down since your first check. Keep it up!</div>""",
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>🗒️ All My Checks</div>", unsafe_allow_html=True)
    for i, h in enumerate(reversed(history), 1):
        _, css, em, col = risk_label(h["prob"])
        idx = len(history) - i + 1
        st.markdown(f"""
        <div class='hist-row'>
            <span style='font-weight:900;color:#ffffff;min-width:26px;'>#{idx}</span>
            <span style='color:#334155;font-size:.82rem;min-width:120px;'>
                {h["time"].strftime("%d %b %Y, %H:%M")}</span>
            <span style='flex:1;'></span>
            <span class='risk-badge {css}' style='font-size:.82rem;padding:.3rem .9rem;'>
                {em} {h["label"]}</span>
            <span style='font-weight:900;color:{col};min-width:55px;text-align:right;font-size:1rem;'>
                {h["prob"]*100:.0f}%</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    _, clr, _ = st.columns([3, 1, 3])
    with clr:
        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.history     = []
            st.session_state.last_result = None
            st.rerun()


# ══════════════════════════════════════════════
# PAGE: HOW IT WORKS
# ══════════════════════════════════════════════
def page_how():
    st.markdown("""
    <div class='header-banner'>
        <h1>ℹ️ How HeartGuard Works</h1>
        <p>Everything explained in simple words — no complicated medical terms!</p>
    </div>
    """, unsafe_allow_html=True)

    h1, h2 = st.columns(2, gap="large")
    with h1:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>🤖 What is AI / Machine Learning?</div>
            <div style='font-size:.9rem;line-height:1.85;color:#1e3a5f;font-weight:600;'>
            Think of it like this:<br><br>
            Imagine a very smart student who studied the medical records of
            <b>2,000 different patients</b>. They learned which combinations of
            blood pressure, cholesterol, age, and other factors are linked to
            heart disease.<br><br>
            When you enter your information, this "student" (the AI) compares
            your details to all the patients it studied and tells you how similar
            you are to people who had heart problems.<br><br>
            It gives you a <b>percentage score</b> — like 30% risk or 70% risk.
            </div>
        </div>
        <div class='card'>
            <div class='card-title'>📊 What Do the Questions Mean?</div>
            <div style='font-size:.88rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            <b>Blood Pressure</b> — The force of blood pushing against your artery walls.
            Normal is 120/80 or below.<br><br>
            <b>Cholesterol</b> — A fatty substance in your blood.
            Too much can block your arteries. Normal is below 200 mg/dL.<br><br>
            <b>Chest Pain on Walking</b> — If your heart doesn't get enough blood
            during exercise, it sends a pain signal.<br><br>
            <b>Heart Rate</b> — How fast your heart beats. A healthy heart can
            beat faster when you need it to.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>🎯 Is This Tool Accurate?</div>
            <div style='font-size:.9rem;line-height:1.85;color:#1e3a5f;font-weight:600;'>
            Our AI is about <b>80–84% accurate</b> in our tests.<br><br>
            That means it gets the right answer about 8 out of every 10 times —
            which is good, but NOT perfect.<br><br>
            <b>This is why:</b><br>
            ✅ Use it as a <b>starting point</b> to understand your risk<br>
            ✅ Use it to <b>motivate</b> you to see a doctor<br>
            ❌ Do NOT use it instead of a <b>real doctor's diagnosis</b><br>
            ❌ Do NOT make medicine decisions based on this alone
            </div>
        </div>
        <div class='card'>
            <div class='card-title'>🔒 Is My Information Safe?</div>
            <div style='font-size:.9rem;line-height:1.85;color:#1e3a5f;font-weight:600;'>
            <b>Yes, completely safe.</b><br><br>
            ✅ We do NOT save your answers anywhere<br>
            ✅ We do NOT ask for your name or email<br>
            ✅ Nobody else can see your results<br>
            ✅ Everything stays on your screen only<br>
            ✅ When you close the browser, everything is gone<br><br>
            Your privacy is completely protected.
            </div>
        </div>
        <div class='card'>
            <div class='card-title'>💰 Is It Really Free?</div>
            <div style='font-size:.9rem;line-height:1.85;color:#1e3a5f;font-weight:600;'>
            <b>Yes, 100% free — forever.</b><br><br>
            There are no hidden charges, no subscriptions,
            no premium plans. HeartGuard will always be
            free for everyone.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: BMI CHECK
# ══════════════════════════════════════════════
def page_bmi():
    st.markdown("""
    <div class='header-banner'>
        <h1>⚖️ BMI & Weight Check</h1>
        <p>Find out if your weight is healthy for your height. BMI stands for Body Mass Index.</p>
    </div>
    """, unsafe_allow_html=True)

    b1, b2 = st.columns(2, gap="large")
    with b1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>⚖️ Calculate Your BMI</div>",
                    unsafe_allow_html=True)
        unit = st.radio("Choose your units:", ["Metric (kg, cm)", "Imperial (lb, inches)"],
                        horizontal=True)

        if "Metric" in unit:
            weight = st.number_input("Your weight (kg)", 30.0, 250.0, 70.0, step=0.5)
            height = st.number_input("Your height (cm)", 100.0, 250.0, 165.0, step=0.5)
            bmi    = weight / (height / 100) ** 2
            iw_lo  = 18.5 * (height/100)**2; iw_hi = 24.9 * (height/100)**2
            u = "kg"
        else:
            weight = st.number_input("Your weight (lb)", 66.0, 550.0, 154.0, step=1.0)
            height = st.number_input("Your height (inches)", 39.0, 98.0, 65.0, step=0.5)
            bmi    = (weight / height**2) * 703
            iw_lo  = 18.5 * height**2 / 703; iw_hi = 24.9 * height**2 / 703
            u = "lb"

        if bmi < 18.5:    cat, tip_c = "Too Thin (Underweight)", "tip-box"
        elif bmi < 25:    cat, tip_c = "Healthy Weight ✅",       "tip-box"
        elif bmi < 30:    cat, tip_c = "A Bit Overweight",        "warn-box"
        else:             cat, tip_c = "Obese (Too Heavy)",        "warn-box"

        st.markdown(f"""
        <div style='text-align:center;margin:1rem 0;'>
            <div style='font-size:3.5rem;font-weight:900;color:#e11d48;'>{bmi:.1f}</div>
            <div style='font-size:1.1rem;font-weight:800;color:#0c1a2e;margin:.3rem 0;'>{cat}</div>
        </div>
        """, unsafe_allow_html=True)
        st.pyplot(ch_bmi(bmi), use_container_width=True)

        diff = (weight if "Metric" in unit else weight) - (iw_lo + iw_hi) / 2 * (1 if "Metric" in unit else 2.205)
        st.markdown(f"""
        <div class='{tip_c}'>
        🎯 <b>Your healthy weight range:</b> {iw_lo:.1f} – {iw_hi:.1f} {u}<br>
        {"✅ You are in the healthy range!" if 18.5 <= bmi < 25 else
         f"You are about {abs(diff):.1f} {u} {'above' if diff > 0 else 'below'} the middle of the healthy range."}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b2:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>📋 BMI Guide — What Does It Mean?</div>
            <div style='font-size:.88rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            <div style='background:#dbeafe;border-radius:10px;padding:.6rem .9rem;margin-bottom:.5rem;'>
                🔵 <b>Below 18.5 — Too Thin</b><br>
                You may need to eat more nutritious food. Being too thin also has health risks.
            </div>
            <div style='background:#dcfce7;border-radius:10px;padding:.6rem .9rem;margin-bottom:.5rem;'>
                💚 <b>18.5 to 24.9 — Healthy</b><br>
                This is the ideal range. Your weight is good for your height!
            </div>
            <div style='background:#fef9c3;border-radius:10px;padding:.6rem .9rem;margin-bottom:.5rem;'>
                💛 <b>25 to 29.9 — A Bit Overweight</b><br>
                Some extra weight. Try to eat a bit less and move more.
            </div>
            <div style='background:#fff1f2;border-radius:10px;padding:.6rem .9rem;margin-bottom:.5rem;'>
                ❤️ <b>30 and above — Obese</b><br>
                Being this heavy puts extra strain on your heart. Talk to a doctor.
            </div>
            </div>
        </div>
        <div class='card'>
            <div class='card-title'>💡 Easy Tips to Reach Healthy Weight</div>
            <div style='font-size:.88rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            ✅ Eat smaller portions — use a smaller plate<br>
            ✅ Drink water before meals to feel full faster<br>
            ✅ Walk 30 minutes every day — even split into 3 x 10 mins<br>
            ✅ Eat more vegetables and fruits, less oily food<br>
            ✅ Avoid sugary drinks like cola and juice<br>
            ✅ Sleep 7–8 hours — poor sleep causes weight gain<br>
            ✅ Eat slowly — it takes 20 minutes to feel full
            </div>
        </div>
        <div class='disclaimer'>
        ⚠️ BMI is a useful guide but not perfect. Athletes and very muscular people
        may have high BMI but be very healthy. Always combine with other health checks.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: HEART RATE ZONES
# ══════════════════════════════════════════════
def page_heartrate():
    st.markdown("""
    <div class='header-banner'>
        <h1>💓 Heart Rate Zones</h1>
        <p>Find out what heart rate is safe and healthy when you exercise.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    💡 <b>What is heart rate?</b> It's how many times your heart beats in one minute.
    You can check it by pressing two fingers on your wrist or neck and counting beats for 15 seconds,
    then multiply by 4. Or use a smartwatch or fitness band.
    </div>
    """, unsafe_allow_html=True)

    h1, h2 = st.columns(2, gap="large")
    with h1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>💓 Your Personal Heart Rate Zones</div>",
                    unsafe_allow_html=True)
        hr_age = st.slider("Your age", 18, 80, 40, key="hr_age")
        rest_hr = st.slider("Your resting heart rate (beats per minute)",
                            40, 100, 70, key="rhr",
                            help="Measure in the morning before getting up")
        mhr = 220 - hr_age; hrr = mhr - rest_hr

        st.markdown(f"""
        <div style='display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1rem;'>
            <span class='stat-pill'>Max heart rate: {mhr} bpm</span>
            <span class='stat-pill'>Heart rate reserve: {hrr} bpm</span>
        </div>
        """, unsafe_allow_html=True)

        zones = [
            ("🟦 Warm Up",        0.50, 0.60, "#bfdbfe", "#1d4ed8", "Light walk — good for beginners",        "5–10 min"),
            ("🟩 Fat Burning",    0.60, 0.70, "#bbf7d0", "#15803d", "Brisk walk — burns fat, good for health", "20–40 min"),
            ("🟨 Fitness",        0.70, 0.80, "#fef08a", "#713f12", "Jogging — builds heart strength",          "20–30 min"),
            ("🟧 Hard Work",      0.80, 0.90, "#fed7aa", "#7c2d12", "Running — for fit people only",            "10–15 min"),
            ("🟥 Max Effort",     0.90, 1.00, "#fecaca", "#991b1b", "Sprint — very short bursts only",          "< 5 min"),
        ]
        for name, lo, hi, bg, tc, desc, dur in zones:
            lo_b = int(rest_hr + hrr * lo); hi_b = int(rest_hr + hrr * hi)
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:.7rem;padding:.55rem 0;
                        border-bottom:1px solid #fde8e8;'>
                <div style='background:{bg};border-radius:8px;padding:.3rem .6rem;
                            font-weight:800;font-size:.82rem;color:{tc};min-width:100px;text-align:center;'>
                    {lo_b}–{hi_b} bpm</div>
                <div style='flex:1;'>
                    <div style='font-weight:800;font-size:.88rem;color:#0c1a2e;'>{name}</div>
                    <div style='font-size:.78rem;color:#334155;font-weight:600;'>{desc}</div>
                </div>
                <span style='background:#fde8e8;color:#e11d48;border-radius:6px;padding:.15rem .5rem;
                             font-size:.75rem;font-weight:800;'>{dur}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with h2:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>🚶 What Heart Rate is Safe for You?</div>
            <div style='font-size:.88rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            <b>If you are new to exercise:</b><br>
            Stay in the Blue Zone (Warm Up). Walk slowly and comfortably.
            If you can talk normally while walking, you're in the right zone.<br><br>
            <b>If you exercise regularly:</b><br>
            Aim for the Green Zone (Fat Burning) for most of your workout.
            This is best for heart health.<br><br>
            <b>If you have heart problems:</b><br>
            Always ask your doctor what heart rate is safe for you before exercising.
            Don't go into the Red or Orange zones without medical advice.
            </div>
        </div>
        <div class='card'>
            <div class='card-title'>⚠️ Stop Exercising Immediately If You Feel...</div>
            <div style='font-size:.88rem;line-height:1.9;color:#9f1239;font-weight:700;'>
            🛑 Chest pain or pressure<br>
            🛑 Dizziness or feeling faint<br>
            🛑 Very breathless and can't speak<br>
            🛑 Heart beating in an irregular way<br>
            🛑 Nausea or feeling very unwell<br><br>
            <span style='color:#1e3a5f;'>
            Sit down, rest, and call for help if it doesn't stop in 2–3 minutes.
            </span>
            </div>
        </div>
        <div class='tip-box'>
        💡 <b>The "Talk Test":</b> You should be able to speak in short sentences while exercising.
        If you can sing, you're going too easy. If you can't speak at all, slow down!
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: BLOOD PRESSURE GUIDE
# ══════════════════════════════════════════════
def page_bp():
    st.markdown("""
    <div class='header-banner'>
        <h1>🩸 Blood Pressure Guide</h1>
        <p>Understand your blood pressure numbers and what they mean for your health.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    💡 <b>What do the two numbers mean?</b><br>
    Blood pressure is written as two numbers like <b>120/80</b>.<br>
    The <b>top number (120)</b> = pressure when your heart beats (systolic)<br>
    The <b>bottom number (80)</b> = pressure when your heart rests between beats (diastolic)
    </div>
    """, unsafe_allow_html=True)

    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🩸 Check Your Blood Pressure</div>",
                    unsafe_allow_html=True)
        sys = st.slider("Top number (systolic)", 70, 200, 120)
        dia = st.slider("Bottom number (diastolic)", 40, 130, 80)

        if sys < 120 and dia < 80:
            cat, bg, tc, msg = "Normal ✅", "#dcfce7", "#14532d", "Your blood pressure is healthy! Keep doing what you're doing."
        elif sys < 130 and dia < 80:
            cat, bg, tc, msg = "Slightly High ⚠️", "#fef9c3", "#713f12", "A bit above normal. Reduce salt and try to relax more."
        elif sys < 140 or dia < 90:
            cat, bg, tc, msg = "High — Stage 1 🔴", "#fff1f2", "#9f1239", "High blood pressure. See your doctor within a week."
        elif sys >= 140 or dia >= 90:
            cat, bg, tc, msg = "High — Stage 2 🔴", "#fff1f2", "#9f1239", "Very high. See your doctor soon."
        else:
            cat, bg, tc, msg = "Emergency 🚨", "#fecaca", "#7f1d1d", "Extremely high. Go to hospital now!"

        st.markdown(f"""
        <div style='text-align:center;padding:1.5rem;background:{bg};border-radius:16px;
                    margin:1rem 0;border:2px solid {tc}20;'>
            <div style='font-size:2.8rem;font-weight:900;color:#0c1a2e;'>
                {sys}<span style='font-size:1.4rem;color:#334155;'>/</span>{dia}
            </div>
            <div style='font-size:.8rem;color:#334155;margin-bottom:.5rem;font-weight:600;'>mmHg</div>
            <div style='font-size:1.1rem;font-weight:800;color:{tc};'>{cat}</div>
            <div style='font-size:.88rem;color:#1e3a5f;margin-top:.4rem;font-weight:600;'>{msg}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>📊 Blood Pressure Categories</div>
        """, unsafe_allow_html=True)
        for cat, top, bot, bg, tc in [
            ("✅ Normal",            "Below 120",  "Below 80",  "#dcfce7", "#14532d"),
            ("⚠️ Slightly High",     "120–129",    "Below 80",  "#fef9c3", "#713f12"),
            ("🔴 High — Stage 1",    "130–139",    "80–89",     "#fff1f2", "#9f1239"),
            ("🔴 High — Stage 2",    "140 or more","90 or more","#fecaca", "#7f1d1d"),
            ("🚨 Emergency",         "180 or more","120 or more","#fecaca","#7f1d1d"),
        ]:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:.8rem;padding:.55rem .6rem;
                        border-radius:10px;background:{bg};margin-bottom:.4rem;border:1px solid {tc}30;'>
                <div style='flex:1.5;font-weight:800;font-size:.85rem;color:{tc};'>{cat}</div>
                <div style='font-size:.82rem;color:#1e3a5f;font-weight:700;'>{top} / {bot}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='card'>
            <div class='card-title'>🧂 How to Lower Blood Pressure Naturally</div>
            <div style='font-size:.88rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            ✅ Eat less salt — avoid adding salt to food<br>
            ✅ Walk 30 minutes most days<br>
            ✅ Eat more fruits and vegetables<br>
            ✅ Drink less alcohol<br>
            ✅ Stop smoking<br>
            ✅ Reduce stress — breathe slowly when stressed<br>
            ✅ Lose weight if you are overweight<br>
            ✅ Sleep 7–8 hours per night
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: HEALTHY EATING
# ══════════════════════════════════════════════
def page_eating():
    st.markdown("""
    <div class='header-banner'>
        <h1>🍎 Healthy Eating for Your Heart</h1>
        <p>Simple food tips that actually work — no fancy diets needed!</p>
    </div>
    """, unsafe_allow_html=True)

    e1, e2 = st.columns(2, gap="large")
    with e1:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>✅ Foods That Are GREAT for Your Heart</div>
            <div style='font-size:.9rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            🫐 <b>Fruits & Berries</b> — Eat daily. Any fruit is good!<br>
            🥦 <b>Vegetables</b> — More the better. All colours!<br>
            🐟 <b>Fish</b> — Especially salmon, sardines, mackerel (2x a week)<br>
            🫘 <b>Lentils & Beans</b> — Great protein, no bad fat<br>
            🥜 <b>Nuts</b> — A small handful of almonds or walnuts daily<br>
            🌾 <b>Brown Rice & Oats</b> — Better than white rice and bread<br>
            🫒 <b>Olive Oil</b> — Use instead of butter for cooking<br>
            🧄 <b>Garlic & Onion</b> — Natural blood pressure helpers<br>
            🍵 <b>Green Tea</b> — Good for heart health
            </div>
        </div>
        <div class='card'>
            <div class='card-title'>🌟 The Heart-Healthy Plate</div>
            <div style='font-size:.9rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            Imagine your plate divided like this:<br><br>
            🟢 <b>Half your plate</b> = Vegetables and salad<br>
            🟡 <b>One quarter</b> = Brown rice, oats, or roti<br>
            🔴 <b>One quarter</b> = Protein (fish, chicken, lentils, eggs)<br><br>
            Add a fruit on the side and drink water instead of cola!
            </div>
        </div>
        """, unsafe_allow_html=True)
    with e2:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>❌ Foods to EAT LESS (Not Banned — Just Less!)</div>
            <div style='font-size:.9rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            🧂 <b>Salt</b> — Too much raises blood pressure<br>
            🍟 <b>Fried Food</b> — Chips, fried chicken, samosas — limit these<br>
            🥩 <b>Red Meat</b> — Mutton, beef — have it less often<br>
            🧀 <b>Full-fat dairy</b> — Butter, cream, cheese — smaller amounts<br>
            🍰 <b>Sweet food</b> — Cakes, biscuits, sweets — occasional treat only<br>
            🥤 <b>Sugary drinks</b> — Cola, juice, energy drinks — choose water<br>
            🚬 <b>Smoking</b> — This is the biggest heart killer — please try to quit
            </div>
        </div>
        <div class='card'>
            <div class='card-title'>💡 Simple Swaps That Help</div>
            <div style='font-size:.9rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            Instead of... → Try this instead<br><br>
            🍟 Chips → 🥕 Carrot sticks or roasted nuts<br>
            🥤 Cola → 💧 Water with lemon<br>
            🍚 White rice → 🌾 Brown rice or oats<br>
            🧈 Butter → 🫒 Olive oil or avocado<br>
            🍩 Sweet biscuit → 🍎 An apple or banana<br>
            🧂 Salt → 🌿 Herbs and spices for flavour
            </div>
        </div>
        <div class='tip-box'>
        💡 <b>Remember:</b> You don't need to be perfect. Just make small improvements
        every week. Even one healthy swap per day adds up to big changes over time!
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: EXERCISE GUIDE
# ══════════════════════════════════════════════
def page_exercise():
    st.markdown("""
    <div class='header-banner'>
        <h1>🚶 Exercise Guide for Beginners</h1>
        <p>You don't need a gym! Simple movements can save your life.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='tip-box'>
    💡 <b>Did you know?</b> Just 30 minutes of brisk walking 5 days a week can reduce your
    heart disease risk by up to 35%! And you don't need any equipment.
    </div>
    """, unsafe_allow_html=True)

    ex1, ex2 = st.columns(2, gap="large")
    with ex1:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>🚶 Week 1–2: Just Start Moving</div>
            <div style='font-size:.9rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            Mon: Walk for 10 minutes after dinner<br>
            Tue: Rest<br>
            Wed: Walk for 10 minutes after lunch<br>
            Thu: Rest<br>
            Fri: Walk for 10 minutes in the morning<br>
            Sat: Walk 15 minutes at your own pace<br>
            Sun: Rest and relax<br><br>
            <b>Goal:</b> Just build the habit of moving!
            </div>
        </div>
        <div class='card'>
            <div class='card-title'>🏃 Week 3–4: Build Up Gradually</div>
            <div style='font-size:.9rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            Mon: Walk 20 minutes briskly<br>
            Tue: Stretching for 15 minutes at home<br>
            Wed: Walk 20 minutes + 5 min stretching<br>
            Thu: Rest or light walk<br>
            Fri: Walk 25 minutes<br>
            Sat: Longer walk — 30 minutes at your pace<br>
            Sun: Rest<br><br>
            <b>Goal:</b> 150 minutes of walking per week total!
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ex2:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>🏠 Simple Exercises You Can Do at Home</div>
            <div style='font-size:.9rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            <b>Standing exercises (no equipment):</b><br>
            • March on the spot for 2 minutes<br>
            • Sit to stand from a chair — 10 times<br>
            • Wall push-ups — 10 times<br>
            • Heel raises — stand and rise on tiptoes — 15 times<br><br>
            <b>Stretches:</b><br>
            • Touch your toes slowly and hold — 30 seconds<br>
            • Shoulder rolls — 10 forward, 10 backward<br>
            • Neck stretches — tilt head each side slowly
            </div>
        </div>
        <div class='card'>
            <div class='card-title'>⚠️ Exercise Safety Rules</div>
            <div style='font-size:.9rem;line-height:1.9;color:#9f1239;font-weight:700;'>
            🛑 Always warm up — walk slowly for 3 minutes first<br>
            🛑 Stop if you feel chest pain or dizziness<br>
            🛑 Drink water before and after<br>
            🛑 Don't exercise when very sick or very tired<br>
            🛑 Ask your doctor first if you have heart disease<br><br>
            <span style='color:#1e3a5f;'>
            💚 Start slow and build up gradually. Any movement is better than none!
            </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: SLEEP & STRESS
# ══════════════════════════════════════════════
def page_sleep():
    st.markdown("""
    <div class='header-banner'>
        <h1>😴 Sleep & Stress Guide</h1>
        <p>Poor sleep and high stress are silent killers for your heart. Here's how to fix them.</p>
    </div>
    """, unsafe_allow_html=True)

    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>😴 How Much Sleep Do You Need?</div>
            <div style='font-size:.9rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            <div style='background:#dbeafe;border-radius:10px;padding:.7rem;margin-bottom:.5rem;'>
                👶 Children (6–12): <b>9–12 hours</b>
            </div>
            <div style='background:#dcfce7;border-radius:10px;padding:.7rem;margin-bottom:.5rem;'>
                🧑 Teenagers (13–18): <b>8–10 hours</b>
            </div>
            <div style='background:#fef9c3;border-radius:10px;padding:.7rem;margin-bottom:.5rem;'>
                🧑‍💼 Adults (18–64): <b>7–9 hours</b>
            </div>
            <div style='background:#fff1f2;border-radius:10px;padding:.7rem;'>
                👴 Older Adults (65+): <b>7–8 hours</b>
            </div>
            <br>
            Sleeping less than 6 hours regularly increases heart disease risk by 48%.
            </div>
        </div>
        <div class='card'>
            <div class='card-title'>🌙 Tips for Better Sleep</div>
            <div style='font-size:.9rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            ✅ Sleep and wake up at the same time every day<br>
            ✅ No phone or TV 1 hour before bed<br>
            ✅ Keep your bedroom dark and cool<br>
            ✅ No coffee or tea after 3pm<br>
            ✅ A warm shower before bed helps you sleep<br>
            ✅ Read a book instead of scrolling your phone<br>
            ✅ Don't eat heavy meals just before bedtime
            </div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>😤 How Stress Hurts Your Heart</div>
            <div style='font-size:.9rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            When you're stressed, your body releases chemicals that:<br><br>
            ❌ Raise your blood pressure<br>
            ❌ Make your heart beat faster<br>
            ❌ Increase inflammation in your arteries<br>
            ❌ Make you eat more unhealthy food<br>
            ❌ Make you sleep less<br><br>
            People with very high stress have a <b>27% higher risk</b> of heart attack.
            </div>
        </div>
        <div class='card'>
            <div class='card-title'>🧘 Easy Ways to Reduce Stress</div>
            <div style='font-size:.9rem;line-height:1.9;color:#1e3a5f;font-weight:600;'>
            <b>Quick fixes (5 minutes):</b><br>
            • Deep breathing: Breathe in for 4 seconds, hold 4, out for 6<br>
            • Step outside for fresh air<br>
            • Drink a glass of cold water slowly<br><br>
            <b>Daily habits:</b><br>
            • Talk to a friend or family member<br>
            • Go for a walk<br>
            • Prayer, meditation, or quiet time<br>
            • Write down 3 things you are grateful for<br>
            • A hobby — cooking, gardening, music
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: QUIZ
# ══════════════════════════════════════════════
def page_quiz():
    st.markdown("""
    <div class='header-banner'>
        <h1>🧠 Heart Health Quiz</h1>
        <p>Test how much you know about heart health! 8 simple questions.</p>
    </div>
    """, unsafe_allow_html=True)

    QS = [
        {"q": "What is a normal blood pressure?",
         "opts": ["150/100", "120/80", "90/50", "180/90"],
         "ans": 1,
         "exp": "Normal blood pressure is 120/80 or below. If yours is higher, try eating less salt and exercising more."},
        {"q": "Which food is BEST for your heart?",
         "opts": ["Fried chicken", "White bread", "Oily fish like salmon", "Cola drink"],
         "ans": 2,
         "exp": "Oily fish like salmon is full of omega-3 fats that protect your heart. Try to eat fish twice a week!"},
        {"q": "How many minutes of walking per week do you need for a healthy heart?",
         "opts": ["30 minutes", "60 minutes", "150 minutes", "300 minutes"],
         "ans": 2,
         "exp": "Health guidelines recommend 150 minutes of moderate exercise per week — that's just 30 minutes a day, 5 days a week!"},
        {"q": "Which of these is a warning sign of a heart attack?",
         "opts": ["Headache", "Chest pain spreading to the left arm", "A runny nose", "Sore throat"],
         "ans": 1,
         "exp": "Chest pain that spreads to the arm, jaw, or back is a classic heart attack sign. Call emergency services immediately!"},
        {"q": "What does high cholesterol do to your heart?",
         "opts": ["Makes it beat faster", "Blocks blood vessels with fatty deposits", "Causes tiredness", "Nothing"],
         "ans": 1,
         "exp": "High cholesterol builds up fatty plaques in your arteries, blocking blood flow to the heart. Eat less fried food and more vegetables!"},
        {"q": "Smoking affects your heart by...",
         "opts": ["Making it stronger", "Damaging blood vessels and causing blockages", "Slowing it down", "No effect"],
         "ans": 1,
         "exp": "Smoking is one of the biggest causes of heart disease. Quitting reduces your risk by 50% within just one year!"},
        {"q": "How much sleep do adults need for a healthy heart?",
         "opts": ["4–5 hours", "6–7 hours", "7–9 hours", "10–12 hours"],
         "ans": 2,
         "exp": "Adults need 7–9 hours of sleep. People who sleep less than 6 hours have nearly 50% higher risk of heart disease!"},
        {"q": "Which is the BEST drink for heart health?",
         "opts": ["Cola drinks", "Energy drinks", "Water", "Fruit juice"],
         "ans": 2,
         "exp": "Plain water is the best drink! Sugary drinks raise blood sugar and cause weight gain, both bad for your heart."},
    ]

    state = st.session_state.quiz_state

    if not state["done"]:
        st.progress(state["current"] / len(QS))
        st.markdown(
            f"<div style='font-size:.9rem;color:#334155;font-weight:700;margin-bottom:1rem;'>"
            f"Question {state['current']+1} of {len(QS)}</div>",
            unsafe_allow_html=True)

        q = QS[state["current"]]
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>Question {state["current"]+1}</div>
            <div style='font-size:1.1rem;font-weight:800;color:#0c1a2e;line-height:1.5;margin-bottom:1rem;'>
                {q["q"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

        sel = st.radio("Your answer:", q["opts"],
                       key=f"q{state['current']}", label_visibility="collapsed")

        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            if st.button("✅ Submit Answer", use_container_width=True, key=f"s{state['current']}"):
                ci = q["opts"].index(sel)
                state["answers"][state["current"]] = ci
                ok = ci == q["ans"]
                if ok:
                    st.success(f"✅ Correct! {q['exp']}")
                else:
                    st.error(f"❌ Not quite. The answer is: **{q['opts'][q['ans']]}**\n\n{q['exp']}")
                import time; time.sleep(0.2)
                state["current"] += 1
                if state["current"] >= len(QS): state["done"] = True
                st.rerun()
        with c2:
            if st.button("⏭️ Skip", use_container_width=True, key=f"sk{state['current']}"):
                state["answers"][state["current"]] = -1
                state["current"] += 1
                if state["current"] >= len(QS): state["done"] = True
                st.rerun()
    else:
        correct = sum(1 for i, q in enumerate(QS) if state["answers"].get(i) == q["ans"])
        total   = len(QS); pct = correct / total * 100
        if pct >= 75:   gr, gc = "Heart Health Expert! 🏆", "#16a34a"
        elif pct >= 50: gr, gc = "Good Knowledge! 👍",      "#ca8a04"
        else:           gr, gc = "Keep Learning! 📖",        "#e11d48"

        st.markdown(f"""
        <div class='card' style='text-align:center;padding:2.5rem;'>
            <div style='font-size:4rem;font-weight:900;color:{gc};'>{correct}/{total}</div>
            <div style='font-size:1.3rem;font-weight:800;color:{gc};margin:.4rem 0;'>{gr}</div>
            <div style='font-size:.95rem;color:#334155;font-weight:600;'>{pct:.0f}% correct</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📝 Review Your Answers")
        for i, q in enumerate(QS):
            given = state["answers"].get(i, -1)
            ok    = given == q["ans"]
            icon  = "✅" if ok else ("⏭️" if given == -1 else "❌")
            css2  = "answer-ok" if ok else ("" if given == -1 else "answer-err")
            st.markdown(f"""
            <div class='answer-card {css2}'>
                <div style='font-weight:800;font-size:.92rem;margin-bottom:.25rem;'>
                    {icon} {q["q"]}</div>
                <div style='font-size:.85rem;font-weight:600;'>
                    Your answer: <b>{q["opts"][given] if given != -1 else "Skipped"}</b> ·
                    Correct: <b style='color:#16a34a;'>{q["opts"][q["ans"]]}</b>
                </div>
                <div style='font-size:.82rem;color:#1e3a5f;margin-top:.25rem;font-weight:600;'>{q["exp"]}</div>
            </div>
            """, unsafe_allow_html=True)

        _, rr, _ = st.columns([2, 1, 2])
        with rr:
            if st.button("🔄 Try Again!", use_container_width=True):
                st.session_state.quiz_state = {"current": 0, "answers": {}, "done": False}
                st.rerun()


# ══════════════════════════════════════════════
# PAGE: HEALTH PLAN
# ══════════════════════════════════════════════
def page_plan():
    st.markdown("""
    <div class='header-banner'>
        <h1>📅 My Personal Health Plan</h1>
        <p>Build your own simple 4-week plan to improve your heart health step by step.</p>
    </div>
    """, unsafe_allow_html=True)

    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>👤 Tell Us About You</div>", unsafe_allow_html=True)
        pl_age  = st.slider("Your age", 18, 80, 40, key="pl_age")
        pl_risk = st.selectbox("What was your HeartGuard risk result?",
                               ["I haven't checked yet", "Low Risk (Green)",
                                "Medium Risk (Yellow)", "High Risk (Red)"], key="pl_risk")
        pl_smoke= st.selectbox("Do you smoke?",
                               ["No", "I used to but quit", "Yes I smoke"], key="pl_smoke")
        pl_ex   = st.selectbox("How much do you exercise?",
                               ["I don't exercise at all", "I walk sometimes",
                                "I exercise 1–2 times a week", "I exercise 3+ times a week"], key="pl_ex")
        pl_diet = st.selectbox("How is your diet?",
                               ["Lots of fried and junk food", "Mixed — some healthy some not",
                                "Mostly healthy", "Very healthy"], key="pl_diet")
        st.markdown("</div>", unsafe_allow_html=True)

    with p2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🎯 Choose Your Goals This Month</div>",
                    unsafe_allow_html=True)
        g_ex    = st.checkbox("🚶 Start exercising more",      value=("don't exercise" in pl_ex))
        g_diet  = st.checkbox("🍎 Eat healthier food",         value=("junk" in pl_diet or "Mixed" in pl_diet))
        g_smoke = st.checkbox("🚭 Reduce or quit smoking",     value=("Yes I smoke" in pl_smoke))
        g_sleep = st.checkbox("😴 Sleep better",               value=True)
        g_bp    = st.checkbox("🩸 Monitor blood pressure",     value=("High" in pl_risk or "Medium" in pl_risk))
        g_doc   = st.checkbox("🏥 Visit a doctor for check-up",value=("High" in pl_risk))
        st.markdown("</div>", unsafe_allow_html=True)

    goals = {"Exercise": g_ex, "Diet": g_diet, "Smoking": g_smoke,
             "Sleep": g_sleep, "BP Monitor": g_bp, "Doctor": g_doc}
    active = [g for g, v in goals.items() if v]

    if not active:
        st.info("Please select at least one goal above!")
        return

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(f"## 📋 Your Personalised 4-Week Plan")

    PLANS = {
        "Exercise": [
            ("Week 1 — Just Start", [
                ("Mon / Wed / Fri", "Walk for 10 minutes after a meal — just 10 minutes!"),
                ("Tue / Thu",       "Stretch for 10 minutes at home — any YouTube stretching video"),
                ("Weekend",         "Go for a slightly longer walk — 15–20 minutes"),
            ]),
            ("Week 2 — Build the Habit", [
                ("Mon / Wed / Fri", "Walk 15 minutes at a comfortable pace"),
                ("Tue / Thu",       "10 sit-to-stand from a chair (10 times), march on the spot 2 minutes"),
                ("Weekend",         "Walk 25 minutes — maybe with a family member"),
            ]),
            ("Week 3 — Increase", [
                ("Mon / Wed / Fri", "Walk 20–25 minutes, a bit faster than before"),
                ("Tue / Thu",       "Wall push-ups (10), heel raises (15), march on spot (3 min)"),
                ("Weekend",         "30-minute walk or a fun activity — swimming, dancing, cycling"),
            ]),
            ("Week 4 — Make it Routine", [
                ("Mon / Wed / Fri", "30-minute brisk walk — this is now your habit!"),
                ("Tue / Thu",       "Full home exercise: 10 min stretching + 10 min movement"),
                ("Weekend",         "Enjoy an active outing — park, beach, or neighbourhood"),
            ]),
        ],
        "Diet": [
            ("Week 1 — Small Swaps", [
                ("Every day",  "Replace one unhealthy snack with a fruit or handful of nuts"),
                ("Each meal",  "Add one extra vegetable to your plate"),
                ("Drinks",     "Replace one cola/juice with a glass of water"),
            ]),
            ("Week 2 — Cook Smarter", [
                ("Every day",  "Use less oil when cooking — try baking or steaming instead of frying"),
                ("Meals",      "Have fish twice this week for dinner"),
                ("Salt",       "Stop adding salt at the table — use herbs and lemon instead"),
            ]),
            ("Week 3 — Better Choices", [
                ("Every day",  "Choose brown rice or oats instead of white rice"),
                ("Protein",    "Replace red meat once with lentils, eggs, or chicken"),
                ("Snacks",     "Keep cut fruits or vegetables in the fridge for easy snacking"),
            ]),
            ("Week 4 — New Normal", [
                ("Goal",       "5 servings of vegetables and fruit every single day"),
                ("Limit",      "Fried food maximum once a week"),
                ("Celebrate",  "You can enjoy one treat — but consciously and with joy!"),
            ]),
        ],
        "Smoking": [
            ("Week 1 — Get Ready", [
                ("Day 1–2",  "Write down WHY you want to quit — put it on your fridge"),
                ("Day 3–5",  "Identify your trigger times — after meals? With tea? Reduce by one"),
                ("Day 6–7",  "Talk to a doctor or pharmacist about nicotine patches or gum"),
            ]),
            ("Week 2 — Quit Day", [
                ("Quit Day",  "Remove all cigarettes from your home and car"),
                ("Every day", "When the urge hits — wait 5 minutes. Drink water. Walk a bit. It passes."),
                ("Support",   "Tell a trusted person you are quitting — ask them to check in on you"),
            ]),
            ("Week 3 — Manage Cravings", [
                ("Morning",   "Change your morning routine — replace the cigarette with a short walk"),
                ("Triggers",  "Avoid situations where you used to smoke (for now)"),
                ("Exercise",  "Even a 5-minute walk kills a craving instantly"),
            ]),
            ("Week 4 — Celebrate", [
                ("Celebrate", "You made it 3+ weeks! Treat yourself to something nice (non-food!)"),
                ("Stay alert","Watch out in social situations — plan what to say if offered a cigarette"),
                ("Long term", "Think about joining a stop-smoking group or helpline for ongoing support"),
            ]),
        ],
        "Sleep": [
            ("Week 1 — Set a Schedule", [
                ("Every day",    "Go to bed and wake up at the same time — even on weekends"),
                ("Before bed",   "No phone, TV, or tablet 45 minutes before sleep"),
                ("Your bedroom", "Make it dark, quiet, and a bit cool"),
            ]),
            ("Week 2 — Wind Down Routine", [
                ("Evening",      "Dim your lights 1 hour before bed — bright light wakes your brain"),
                ("Before sleep", "Read a book, pray, or do gentle stretches instead of scrolling"),
                ("Drinks",       "No tea or coffee after 3pm — swap for herbal tea or warm water"),
            ]),
            ("Week 3 — Improve Sleep Quality", [
                ("Daytime",      "Get some sunlight every morning — this sets your body clock"),
                ("Naps",         "If you nap, keep it to 20 minutes before 3pm only"),
                ("Exercise",     "Exercise helps you sleep better — even a 20-min walk"),
            ]),
            ("Week 4 — Review", [
                ("Check",        "Are you waking up feeling rested? If not, see a doctor — you may have sleep apnoea"),
                ("Track",        "Note your sleep time on your phone to see your improvement"),
                ("Maintain",     "Keep your bedtime routine — it's now protecting your heart!"),
            ]),
        ],
        "BP Monitor": [
            ("Week 1 — Know Your Numbers", [
                ("Day 1",        "Buy a home blood pressure monitor (available at chemists) or use the one at your local pharmacy"),
                ("Morning",      "Check BP after sitting quietly for 5 minutes — before eating or drinking"),
                ("Write it down","Keep a simple notebook: date, time, BP reading"),
            ]),
            ("Week 2 — Track Consistently", [
                ("Daily",        "Check once in the morning, once in the evening"),
                ("Log it",       "Note anything unusual — stress, bad sleep, salty meal"),
                ("Share",        "Show your log to your doctor at your next visit"),
            ]),
            ("Week 3 — Make Changes", [
                ("Salt",         "Count how many salty foods you eat — try to reduce by half"),
                ("Water",        "Drink 6–8 glasses of water daily — dehydration raises BP"),
                ("Relax",        "5 minutes of deep breathing twice a day lowers BP naturally"),
            ]),
            ("Week 4 — Evaluate", [
                ("Compare",      "Look at your readings from Week 1 vs Week 4 — is there improvement?"),
                ("Doctor",       "Book an appointment to review readings with your GP"),
                ("Medicines",    "If your BP is still high, talk to your doctor about medicines"),
            ]),
        ],
        "Doctor": [
            ("Week 1 — Book an Appointment", [
                ("Day 1",       "Call your GP or local clinic and book an appointment"),
                ("Prepare",     "Write down all your symptoms and concerns to discuss"),
                ("Gather",      "Bring any home BP readings, medicines you take, and family history"),
            ]),
            ("Week 2 — Before Your Visit", [
                ("Write down",  "All medicines and supplements you take (including dosage)"),
                ("List",        "Any symptoms — chest pain, breathlessness, swollen ankles, dizziness"),
                ("Questions",   "What tests do I need? What lifestyle changes should I make?"),
            ]),
            ("Week 3 — After Your Visit", [
                ("Follow advice","Take medicines as prescribed — don't skip doses"),
                ("Tests",       "Get any blood tests or ECG done that were requested"),
                ("Changes",     "Start the lifestyle changes your doctor recommended"),
            ]),
            ("Week 4 — Follow-Up", [
                ("Review",      "Check if you feel any improvement — note it down"),
                ("Follow-up",   "Book your next appointment as advised"),
                ("Regular",     "Make annual heart check-ups a permanent habit"),
            ]),
        ],
    }

    for goal in active:
        if goal not in PLANS: continue
        icon_m = {"Exercise":"🚶","Diet":"🍎","Smoking":"🚭","Sleep":"😴",
                  "BP Monitor":"🩸","Doctor":"🏥"}
        st.markdown(f"### {icon_m.get(goal,'')} {goal} Plan")
        for week_title, days in PLANS[goal]:
            st.markdown(f"<div class='plan-week'>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-weight:800;font-size:.95rem;color:#1d4ed8;margin-bottom:.55rem;'>"
                f"{week_title}</div>",
                unsafe_allow_html=True)
            for day, task in days:
                st.markdown(
                    f"<div class='plan-day'>"
                    f"<span style='min-width:120px;font-weight:800;color:#1d4ed8;font-size:.84rem;'>{day}</span>"
                    f"<span style='font-size:.88rem;color:#1e3a5f;line-height:1.55;'>{task}</span>"
                    f"</div>",
                    unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='tip-box' style='text-align:center;'>
        🎉 You have selected <b>{len(active)} goal(s)</b>: <b>{", ".join(active)}</b>.<br>
        Remember — <b>small steps done consistently beat big changes that don't last</b>.
        Your heart will thank you!
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class='disclaimer'>
    ⚠️ This plan is for general guidance only. Always speak to your doctor before making
    major changes — especially if you have existing health conditions.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: COMMON QUESTIONS (FAQ)
# ══════════════════════════════════════════════
def page_faq():
    st.markdown("""
    <div class='header-banner'>
        <h1>❓ Common Questions</h1>
        <p>Simple answers to the most common questions about heart health.</p>
    </div>
    """, unsafe_allow_html=True)

    faqs = [
        ("❤️ What is heart disease?",
         "Heart disease means your heart or blood vessels are not working properly. "
         "The most common type is when the pipes (arteries) that carry blood to your heart "
         "get blocked with fatty material. This can cause a heart attack."),
        ("🔴 What is a heart attack?",
         "A heart attack happens when the blood supply to part of your heart is completely blocked. "
         "That part of the heart starts to die because it has no oxygen. "
         "Signs include chest pain, arm pain, breathlessness, and sweating. Call 999/112 immediately!"),
        ("🤔 Can young people get heart disease?",
         "Yes! While it's more common in older people, young people can also get heart disease — "
         "especially if they smoke, have diabetes, are very overweight, or have a family history of heart problems. "
         "It's important to build healthy habits from a young age."),
        ("💊 Do I need medicine for my heart?",
         "That depends on your health. Your doctor will decide based on your blood pressure, "
         "cholesterol, and other factors. Never take heart medicines without a doctor's prescription. "
         "Many people can improve their heart health with lifestyle changes alone — no medicines needed!"),
        ("🏃 Is it safe to exercise if I have heart problems?",
         "Light exercise is often recommended even for heart patients, but always ask your doctor first. "
         "Start with gentle walking and build up slowly. Your doctor may refer you to a cardiac rehabilitation programme."),
        ("🧂 How much salt is too much?",
         "Adults should have no more than 5–6 grams of salt per day (about 1 teaspoon). "
         "Most people eat twice this amount. The biggest sources are bread, processed food, "
         "restaurant food, and canned food. Cook at home to control your salt intake."),
        ("🍺 Does alcohol affect my heart?",
         "Yes. Drinking too much alcohol raises blood pressure, causes weight gain, and can damage the heart muscle. "
         "Small amounts (1 drink per day) may be harmless for some people, but it's safest to limit alcohol. "
         "If you have heart problems, ask your doctor how much is safe for you."),
        ("😟 Can stress cause a heart attack?",
         "Yes, severe stress can trigger a heart attack in people who already have heart disease. "
         "Long-term stress also raises blood pressure and causes poor lifestyle habits. "
         "Learning to manage stress is a real medical treatment — not just a luxury!"),
        ("🧬 Does heart disease run in families?",
         "Yes, there is a genetic component. If your parent, brother, or sister had a heart attack "
         "before age 55 (men) or 65 (women), your risk is higher. This is called family history. "
         "You can't change your genes, but you can control lifestyle factors even more carefully."),
        ("📱 How often should I use HeartGuard?",
         "You can check your heart health once a month to track changes over time. "
         "If you make lifestyle improvements, you may see your risk score improve. "
         "Remember — a real doctor's check-up is still important at least once a year!"),
    ]

    for q, a in faqs:
        with st.expander(q):
            st.markdown(
                f"<div style='font-size:.92rem;line-height:1.82;color:#1e3a5f;font-weight:600;"
                f"padding:.5rem 0;'>{a}</div>",
                unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card' style='text-align:center;'>
        <div style='font-size:1.5rem;margin-bottom:.5rem;'>🏥</div>
        <div style='font-weight:800;font-size:1rem;color:#0c1a2e;margin-bottom:.4rem;'>
            Still have questions?</div>
        <div style='font-size:.9rem;color:#1e3a5f;font-weight:600;line-height:1.7;'>
        The best person to ask is always your family doctor (GP).<br>
        Don't be afraid to visit — that's what they're there for!<br><br>
        In India: Call <b>104</b> for free health advice<br>
        Emergency (India): <b>112</b><br>
        Emergency (UK): <b>999</b> · Emergency (US): <b>911</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN ROUTER
# ──────────────────────────────────────────────
def main():
    with st.spinner("⏳ Loading HeartGuard... just a moment!"):
        M = train_models()

    page = sidebar_nav()

    if   page == "Home":                    page_home()
    elif page == "Check My Heart":          page_check(M)
    elif page == "My Results History":      page_history()
    elif page == "How It Works":            page_how()
    elif page == "BMI & Weight Check":      page_bmi()
    elif page == "Heart Rate Zones":        page_heartrate()
    elif page == "Blood Pressure Guide":    page_bp()
    elif page == "Healthy Eating Tips":     page_eating()
    elif page == "Exercise for Beginners":  page_exercise()
    elif page == "Sleep & Stress Guide":    page_sleep()
    elif page == "Heart Health Quiz":       page_quiz()
    elif page == "My Health Plan":          page_plan()
    elif page == "Common Questions":        page_faq()
    else:                                   page_home()


if __name__ == "__main__":
    main()
