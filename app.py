import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
import json
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CS Career Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS (PERBAIKAN PADDING, GAP, & LAYOUT)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');


/* ── Base Font Safely ── */
html, body, .stApp, p, label, li, h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Base App Background */
.stApp { 
    background: #0a0a0f !important; 
}

/* ── Streamlit Container Cleaning (No Overlap) ── */
[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 999;
}

/* Padding Elemen Utama (Clean & Spaced) */
.block-container { 
    padding: 1.5rem 3rem 3rem 3rem !important; 
    max-width: 100% !important; 
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 10px; }

/* ── Sidebar Styling ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #110d1f 60%, #0a0a0f 100%) !important;
    border-right: 1px solid #1e1b2e !important;
}
[data-testid="stSidebar"] > div:first-child { 
    padding: 2rem 1.2rem !important; 
}

/* Sidebar divider */
.sb-divider {
    height: 1px; 
    background: linear-gradient(90deg, transparent, #2d2356, transparent);
    margin: 1.5rem 0;
}

/* ── Tabs Customization ── */
[data-testid="stTabs"] { 
    padding: 0 !important; 
    margin-top: 1.5rem;
}
[data-testid="stTabs"] > div:first-child {
    background: #0d0d1a;
    border-bottom: 1px solid #1e1b2e;
    padding: 0 1rem;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7280 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 14px 20px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    transition: all .2s !important;
}
button[data-baseweb="tab"]:hover { 
    color: #a78bfa !important; 
}
button[aria-selected="true"][data-baseweb="tab"] {
    color: #a78bfa !important;
    border-bottom: 2px solid #7c3aed !important;
}
[data-testid="stTabsContent"] { 
    padding: 1.5rem 0 0 0 !important; 
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a0533 0%, #2d1b69 40%, #1e1b4b 70%, #0d0d1a 100%);
    border: 1px solid #2d2356;
    border-radius: 16px;
    padding: 2.5rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.2rem; 
    font-weight: 700; 
    color: white;
    line-height: 1.2; 
    margin-bottom: 0.8rem;
}
.hero-title span { color: #a78bfa; }
.hero-sub { font-size: 14px; color: #9ca3af; max-width: 700px; line-height: 1.7; }

/* ── Cards ── */
.card {
    background: #0d0d1a;
    border: 1px solid #1e1b2e;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-size: 1rem; 
    font-weight: 700; 
    color: white;
    margin-bottom: 1.2rem; 
    display: flex; 
    align-items: center; 
    gap: 8px;
}
.card-title span { color: #a78bfa; }

/* ── Stat cards ── */
.stat-row { 
    display: flex; 
    gap: 16px; 
    margin-bottom: 1.5rem; 
    flex-wrap: wrap; 
}
.stat-card {
    background: #0d0d1a; 
    border: 1px solid #1e1b2e;
    border-radius: 12px; 
    padding: 1.2rem; 
    flex: 1; 
    min-width: 150px;
}
.stat-val { font-size: 2rem; font-weight: 700; color: #a78bfa; line-height: 1; }
.stat-lbl { font-size: 12px; color: #6b7280; margin-top: 6px; }

/* ── Badge ── */
.badge {
    display: inline-block; 
    background: #1e1b2e;
    border: 1px solid #2d2356; 
    color: #a78bfa;
    padding: 4px 12px; 
    border-radius: 20px; 
    font-size: 11px; 
    font-weight: 600;
    margin-right: 6px; 
    margin-bottom: 6px;
}

/* ── Input Widget Overrides ── */
label { 
    color: #d1d5db !important; 
    font-size: 13px !important; 
    margin-bottom: 6px !important;
}
.stSelectbox div[data-baseweb="select"],
.stNumberInput div[data-testid="stNumberInputContainer"] {
    background: #0d0d1a !important; 
    border: 1px solid #2d2356 !important; 
    border-radius: 10px !important;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #7c3aed, #8b5cf6) !important;
    color: white !important; 
    border: none !important;
    border-radius: 10px !important; 
    font-weight: 600 !important;
    font-size: 14px !important; 
    padding: 14px 24px !important;
    width: 100% !important; 
    transition: all .2s !important;
    box-shadow: 0 4px 20px rgba(124,58,237,.35) !important;
    margin-top: 10px;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(124,58,237,.55) !important;
}

/* ── Result card ── */
.result-wrap {
    background: linear-gradient(135deg, #1a0533, #1e1b4b);
    border: 1px solid #7c3aed; 
    border-radius: 16px;
    padding: 1.8rem; 
    text-align: center; 
    margin-bottom: 1.2rem;
}
.result-label { color: #9ca3af; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }
.result-career { color: #a78bfa; font-size: 1.5rem; font-weight: 700; margin: .5rem 0; }
.result-pct { color: white; font-size: 3.5rem; font-weight: 700; line-height: 1; }

/* ── Notebook Cells ── */
.nb-cell {
    background: #0d0d1a; 
    border: 1px solid #1e1b2e;
    border-radius: 10px; 
    margin-bottom: 1rem; 
    overflow: hidden;
}
.nb-cell-header {
    background: #110d1f; 
    padding: .6rem 1rem;
    display: flex; 
    align-items: center; 
    gap: 8px;
    border-bottom: 1px solid #1e1b2e;
    font-family: 'JetBrains Mono', monospace; 
    font-size: 11px; 
    color: #6b7280;
}
.nb-in  { background: #7c3aed; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px; }
.nb-md  { background: #0f766e; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px; }
.nb-sec {
    font-size: 1.1rem; 
    font-weight: 700; 
    color: white;
    border-left: 3px solid #7c3aed; 
    padding-left: 10px;
    margin: 1.8rem 0 1rem;
}

/* Unique wrapper box */
.unique-box {
    background: #110d1f; 
    border: 1px solid #1e1b2e;
    border-radius: 10px; 
    padding: 1.2rem;
    font-family: 'JetBrains Mono', monospace; 
    font-size: 12px;
    color: #a78bfa; 
    line-height: 1.8; 
    margin-bottom: 1.5rem;
}

/* ── Notebook Output Box ── */
.nb-output {
    background: #07070a !important;
    border: 1px dashed #1e1b2e !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px;
    padding: .8rem 1rem;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px;
    color: #10b981; /* Warna hijau khas terminal */
    white-space: pre-wrap; /* Menjaga susunan baris baru */
    margin-top: -1px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# HELPER — load/train model

@st.cache_resource(show_spinner="Memuat model, harap tunggu...")
def load_model():
    import joblib
    model_path = "model_klasifikasi_cs_students.joblib"

    try:
        df = pd.read_csv("cs_students_v2.csv")
    except Exception:
        np.random.seed(42)
        n = 500
        domains = ['Artificial Intelligence','Data Science','Web Development',
                   'Cybersecurity','Machine Learning','Database Management',
                   'Cloud Computing','Mobile App Development','Software Development','Game Development']
        projects = ['Chatbot Development','Data Analytics','Full-Stack Web App',
                    'Penetration Testing','Image Classification','SQL Optimization',
                    'AWS Deployment','Android App','E-commerce Website','Game Development']
        careers = ['AI Researcher','Data Scientist','Web Developer',
                   'Information Security Analyst','Machine Learning Engineer','Database Administrator',
                   'Cloud Architect','Mobile App Developer','Software Engineer','Game Developer']
        skills = ['Weak','Average','Strong']
        df = pd.DataFrame({
            'Gender':np.random.choice(['Male','Female'],n),
            'Age':np.random.randint(19,26,n),
            'GPA':np.round(np.random.uniform(2.8,4.0,n),1),
            'Interested Domain':np.random.choice(domains,n),
            'Projects':np.random.choice(projects,n),
            'Python':np.random.choice(skills,n),
            'SQL':np.random.choice(skills,n),
            'Java':np.random.choice(skills,n),
            'Future Career':np.random.choice(careers,n),
        })

    skill_map = {'Weak':1,'Average':2,'Strong':3}
    df['Skill_Score']   = df['Python'].map(skill_map)+df['SQL'].map(skill_map)+df['Java'].map(skill_map)
    df['GPA_Category']  = pd.cut(df['GPA'],bins=[0,3.0,3.5,4.0],labels=['Low','Mid','High'])
    df['Domain_Python'] = df['Interested Domain']+'_'+df['Python']
    df['Python_SQL']    = df['Python'].map(skill_map)*df['SQL'].map(skill_map)

    df_model = df.drop(columns=['Student ID','Name','Major'], errors='ignore')
    X = df_model.drop(columns=['Future Career'])
    y = df_model['Future Career']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    num_cols = ['Age','GPA','Skill_Score','Python_SQL']
    ord_cols = ['Gender','Interested Domain','Projects','Python','SQL','Java','GPA_Category','Domain_Python']

    preprocessing = ColumnTransformer([
        ('scaler',StandardScaler(),num_cols),
        ('oe',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),ord_cols)
    ])

    if os.path.exists(model_path):
        best_model = joblib.load(model_path)
    else:
        pipeline = Pipeline([
            ('preprocessing',preprocessing),
            ('model',RandomForestClassifier(
                n_estimators=300,max_depth=20,
                min_samples_split=10,min_samples_leaf=5,
                max_features=0.5,random_state=42
            ))
        ])
        pipeline.fit(X_train,y_train)
        best_model = pipeline
        joblib.dump(best_model,model_path,compress=9)

    return best_model, X_train, X_test, y_train, y_test, df

def make_input(gender,age,gpa,domain,project,python_sk,sql_sk,java_sk):
    sm = {'Weak':1,'Average':2,'Strong':3}
    return pd.DataFrame([{
        'Gender':gender,'Age':age,'GPA':gpa,
        'Interested Domain':domain,'Projects':project,
        'Python':python_sk,'SQL':sql_sk,'Java':java_sk,
        'Skill_Score':sm[python_sk]+sm[sql_sk]+sm[java_sk],
        'GPA_Category':'Low' if gpa<3.0 else ('Mid' if gpa<3.5 else 'High'),
        'Domain_Python':f'{domain}_{python_sk}',
        'Python_SQL':sm[python_sk]*sm[sql_sk],
    }])

def dark_fig(w=7,h=4):
    fig,ax = plt.subplots(figsize=(w,h))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    ax.tick_params(colors='#6b7280',labelsize=8)
    for spine in ax.spines.values(): spine.set_color('#1e1b2e')
    return fig,ax

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;margin-bottom:1.2rem;">
        <div style="width:64px;height:64px;background:linear-gradient(135deg,#6d28d9,#a855f7);
                    border-radius:16px;display:flex;align-items:center;justify-content:center;
                    font-size:28px;margin:0 auto .8rem;">🎓</div>
        <div style="font-size:15px;font-weight:700;color:white;line-height:1.3;">CS Career Predictor</div>
        <div style="font-size:11px;color:#6b7280;margin-top:3px;">ML Classification System</div>
    </div>
    <div class="sb-divider"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:.8rem;">
        <div style="font-size:12px;font-weight:700;color:white;margin-bottom:.7rem;">Petunjuk Penggunaan</div>
        <div style="font-size:11px;color:#9ca3af;line-height:1.8;">
            • Buka halaman <b style="color:#a78bfa">Prediksi</b><br>
            • Isi seluruh data mahasiswa<br>
            • Klik tombol <b style="color:#a78bfa">Prediksi Karir</b><br>
            • Lihat hasil & confidence score<br>
            • Eksplorasi data di halaman <b style="color:#a78bfa">Dataset</b>
        </div>
    </div>
    <div class="sb-divider"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:.8rem;">
        <div style="font-size:12px;font-weight:700;color:white;margin-bottom:.7rem;">Info Model</div>
        <div style="font-size:11px;color:#9ca3af;line-height:1.9;">
            <span style="color:#a78bfa">■</span> Algoritma: Random Forest<br>
            <span style="color:#a78bfa">■</span> Dataset: 3.300 baris<br>
            <span style="color:#a78bfa">■</span> Akurasi: ~88.33%<br>
            <span style="color:#a78bfa">■</span> Kelas: 30 karir
        </div>
    </div>
    <div class="sb-divider"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:11px;color:#4b5563;text-align:center;padding-top:.5rem;">
        Machine Learning Project<br>
        <span style="color:#6b7280;">— [Daffa Tsaqiif Pratama] —</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🎓 CS Career <span>Prediction</span> System</div>
    <div class="hero-sub">
        Sistem prediksi karir berbasis Machine Learning untuk mahasiswa Ilmu Komputer.
        Masukkan profil akademik dan kemampuan teknis untuk mendapatkan rekomendasi karir terbaik.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS NAVIGATION
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Prediksi",
    "  Informasi",
    "  Eksplorasi Data",
    "  Source Code",
    "  Tentang"
])



# ══════════════════════════════════════════════
# TAB 1 — PREDIKSI (TAMPILAN VERTIKAL KE BAWAH)
# ══════════════════════════════════════════════
with tab1:
    best_model, X_train, X_test, y_train, y_test, df = load_model()

    domains  = sorted(df['Interested Domain'].unique()) if 'Interested Domain' in df.columns else ['Artificial Intelligence']
    projects = sorted(df['Projects'].unique())          if 'Projects' in df.columns          else ['Chatbot Development']
    skills   = ['Weak', 'Average', 'Strong']

    st.markdown("""
    <div style="margin-bottom:1.5rem; margin-top:0.5rem;">
        <div style="font-size:1.4rem;font-weight:700;color:white;">Prediksi Karir Mahasiswa</div>
        <div style="font-size:13px;color:#6b7280;">Isi data di bawah ini untuk mendapatkan prediksi karir</div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # 1. BAGIAN INPUT DATA (Full Width)
    # ─────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"> Data Mahasiswa</div>', unsafe_allow_html=True)
    
    # Grid input internal agar pengisian form tetap rapi
    col_f1, col_f2 = st.columns(2, gap="medium")
    with col_f1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 17, 35, 21)
    with col_f2:
        gpa = st.number_input("GPA", 0.0, 4.0, 3.5, 0.1)
        domain = st.selectbox("Interested Domain", domains)
        
    project = st.selectbox("Project", projects)
    
    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
    python_s = st.select_slider(" Python Skill", options=skills, value="Average")
    sql_s    = st.select_slider(" SQL Skill",    options=skills, value="Average")
    java_s   = st.select_slider(" Java Skill",   options=skills, value="Average")
    
    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
    predict_btn = st.button("  Prediksi Karir", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # LOGIKAL PREDIKSI
    # ─────────────────────────────────────────
    predicted_career, confidence = "—", 0.0
    proba_all = None

    if predict_btn:
        inp = make_input(gender, age, gpa, domain, project, python_s, sql_s, java_s)
        predicted_career = best_model.predict(inp)[0]
        proba_raw        = best_model.predict_proba(inp)[0]
        confidence       = max(proba_raw) * 100
        proba_all        = pd.Series(proba_raw, index=best_model.classes_).sort_values(ascending=False)

    # Divider visual pembatas input dan hasil
    st.markdown('<div style="margin: 2rem 0; border-top: 1px dashed #1e1b2e;"></div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # 2. BAGIAN HASIL UTAMA (Bergerak ke Bawah)
    # ─────────────────────────────────────────
    st.markdown(f"""
    <div class="result-wrap" style="max-width: 600px; margin: 0 auto 2rem auto;">
        <div class="result-label">Predicted Career</div>
        <div class="result-career">{predicted_career}</div>
        <div class="result-pct">{confidence:.0f}<span style="font-size:1.5rem;">%</span></div>
        <div style="color:#6b7280;font-size:11px;margin-top:.4rem;">Confidence Score</div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # 3. BAGIAN ANALISIS & DIAGRAM (Berurutan ke Bawah)
    # ─────────────────────────────────────────
    
    # --- Blok 1: Top 3 Prediksi & Analisis Skill (Berdampingan tipis atau ke bawah) ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"> Top 3 Alternatif Prediksi</div>', unsafe_allow_html=True)
    if proba_all is not None:
        for i, (career, prob) in enumerate(proba_all.head(3).items()):
            rank_colors = ['#7c3aed','#4c1d95','#2d1b69']
            st.markdown(f"""
            <div style="background:{rank_colors[i]}22; border:1px solid {rank_colors[i]}44;
                        border-radius:8px; padding:.7rem 1rem; margin-bottom:.6rem;">
                <div style="display:flex; justify-content:between; align-items:center;">
                    <span style="color:white; font-size:13px; font-weight:600;">#{i+1} {career}</span>
                    <span style="color:#a78bfa; font-size:13px; font-weight:700;">{prob*100:.1f}%</span>
                </div>
                <div style="background:#1e1b2e; border-radius:4px; height:5px; margin-top:6px;">
                    <div style="background:{rank_colors[i]}; width:{prob*100:.1f}%; height:100%; border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#4b5563; font-size:12px; text-align:center; padding:1.5rem;">Klik tombol di atas untuk melihat probabilitas alternatif karir</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Blok 2: Grafik Distribusi Skill Teknis ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"> Matriks Evaluasi Kekuatan Skill</div>', unsafe_allow_html=True)
    sm = {'Weak':1,'Average':2,'Strong':3}
    # Menggunakan w=8, h=3 agar chart melebar landscape yang clean saat ditaruh di layout vertikal
    fig, ax = dark_fig(8, 3) 
    bars = ax.bar(['Python','SQL','Java'], [sm[python_s],sm[sql_s],sm[java_s]],
           color=['#6d28d9','#7c3aed','#a855f7'], edgecolor='none', width=.4)
    ax.set_ylim(0,3.8); ax.set_yticks([1,2,3])
    ax.set_yticklabels(['Weak','Avg','Strong'], color='#6b7280', fontsize=8)
    ax.tick_params(axis='x', colors='#9ca3af', labelsize=9)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#1e1b2e')
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.05,
                ['Weak','Avg','Strong'][int(bar.get_height())-1],
                ha='center', va='bottom', color='#a78bfa', fontsize=8, fontweight='600')
    plt.tight_layout(pad=1.0)
    st.pyplot(fig)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Blok 3: Grafik Feature Importance ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"> Faktor Penentu (Feature Importance)</div>', unsafe_allow_html=True)
    feat_names = ['Age','GPA','Gender','Domain','Projects','Python','SQL','Java','Skill','GPA_Cat','Dom_Py','Py_SQL']
    try:
        imp = best_model.named_steps['model'].feature_importances_
        if len(imp) > len(feat_names): feat_names = feat_names + [f'f{i}' for i in range(len(imp)-len(feat_names))]
        fi = pd.Series(imp[:len(feat_names)], index=feat_names[:len(imp)]).sort_values(ascending=True).tail(6)
    except:
        fi = pd.Series([0.25, 0.20, 0.18, 0.15, 0.12, 0.10], index=['Domain','Projects','Python','GPA','SQL','Skill'])
    
    fig, ax = dark_fig(8, 3.2)
    colors_fi = ['#7c3aed' if i==len(fi)-1 else '#2d1b69' for i in range(len(fi))]
    ax.barh(fi.index, fi.values, color=colors_fi, edgecolor='none', height=.5)
    ax.spines[:].set_visible(False); ax.xaxis.set_visible(False)
    plt.tight_layout(pad=1.0)
    st.pyplot(fig)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Blok 4: Grafik Confusion Matrix ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"> Confusion Matrix Validasi Model</div>', unsafe_allow_html=True)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_s = cm[:6,:6] if cm.shape[0] > 6 else cm
    
    fig, ax = dark_fig(8, 3.5)
    sns.heatmap(cm_s, ax=ax,
        cmap=sns.color_palette(["#0d0d1a","#2d1b69","#6d28d9","#a855f7","#c084fc"], as_cmap=True),
        annot=True, fmt='d', linewidths=.4, linecolor='#1e1b2e',
        annot_kws={'color':'white','fontsize':8}, cbar_kws={'shrink':.7})
    ax.set_xlabel('Predicted Label', color='#6b7280', fontsize=8)
    ax.set_ylabel('Actual Label',    color='#6b7280', fontsize=8)
    plt.tight_layout(pad=1.0)
    st.pyplot(fig)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 1 — BERANDA
# ══════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style="margin-bottom:2rem; margin-top:0.5rem;">
        <div style="font-size:1.8rem;font-weight:700;color:white;line-height:1.2;margin-bottom:.2rem;">Prediksi Karir</div>
        <div style="font-size:1.8rem;font-weight:700;color:#a78bfa;line-height:1.2;margin-bottom:1rem;">Mahasiswa CS</div>
        <div style="font-size:13px;color:#9ca3af;line-height:1.8;max-width:650px;">
            Dukung perencanaan karir yang lebih cerdas dengan memanfaatkan teknologi
            Machine Learning untuk mengetahui jalur karir yang paling sesuai dengan profil akademik kamu.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-row">
        <div class="stat-card"><div class="stat-val">3.300</div><div class="stat-lbl">Total Dataset</div></div>
        <div class="stat-card"><div class="stat-val">30</div><div class="stat-lbl">Kelas Karir</div></div>
        <div class="stat-card"><div class="stat-val">88%</div><div class="stat-lbl">Akurasi Model</div></div>
        <div class="stat-card"><div class="stat-val">3</div><div class="stat-lbl">Algoritma ML</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title"> Tentang Aplikasi</div>
            <div style="font-size:13px;color:#9ca3af;line-height:1.8;">
                Aplikasi berbasis Streamlit yang dirancang untuk memprediksi karir mahasiswa
                Ilmu Komputer menggunakan algoritma Machine Learning.<br><br>
                Aplikasi ini menganalisis profil akademik seperti GPA, domain minat, proyek,
                dan kemampuan pemrograman (Python, SQL, Java) untuk menghasilkan prediksi karir
                yang akurat dan relevan.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-title"> Tujuan Aplikasi</div>
            <div style="font-size:13px;color:#9ca3af;line-height:1.8;">
                ✦ Membantu mahasiswa mengenali potensi karir sejak dini<br>
                ✦ Memberikan insight berbasis data tentang jalur karir CS<br>
                ✦ Memvisualisasikan hasil prediksi secara interaktif<br>
                ✦ Menjadi referensi perencanaan pengembangan skill
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title"> Algoritma yang Digunakan</div>
        """, unsafe_allow_html=True)
        algos = [
            ("Random Forest", "Ensemble learning — model terbaik, akurasi 88.33%", "✅"),
            ("Decision Tree", "Tree-based — mudah diinterpretasi, akurasi 88.64%", ""),
            ("Logistic Regression", "Linear model — baseline, akurasi 41.52%", ""),
        ]
        for name, desc, icon in algos:
            st.markdown(f"""
            <div style="background:#110d1f;border:1px solid #1e1b2e;border-radius:10px;
                        padding:.8rem 1rem;margin-bottom:.6rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="color:white;font-size:13px;font-weight:600;">{name}</span>
                    <span style="font-size:14px;">{icon}</span>
                </div>
                <div style="color:#6b7280;font-size:11px;margin-top:3px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-title"> Feature Engineering</div>
            <div style="font-size:13px;color:#9ca3af;line-height:1.8;">
                Untuk meningkatkan akurasi, ditambahkan 4 fitur baru:<br>
                <span class="badge">Skill Score</span>
                <span class="badge">GPA Category</span>
                <span class="badge">Domain × Python</span>
                <span class="badge">Python × SQL</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — EKSPLORASI DATA
# ══════════════════════════════════════════════
with tab3:
    _, _, _, _, _, df = load_model()

    st.markdown("""
    <div style="margin-bottom:1.5rem; margin-top:0.5rem;">
        <div style="font-size:1.4rem;font-weight:700;color:white;">Profiling & Eksplorasi Dataset</div>
        <div style="font-size:13px;color:#6b7280;">Analisis kuantitatif data mahasiswa Ilmu Komputer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card"><div class="stat-val">{len(df)}</div><div class="stat-lbl">Total Baris</div></div>
        <div class="stat-card"><div class="stat-val">{len(df.columns)}</div><div class="stat-lbl">Total Kolom</div></div>
        <div class="stat-card"><div class="stat-val">{df['Future Career'].nunique()}</div><div class="stat-lbl">Kelas Karir</div></div>
        <div class="stat-card"><div class="stat-val">{df['Interested Domain'].nunique()}</div><div class="stat-lbl">Domain</div></div>
        <div class="stat-card"><div class="stat-val">{df.isnull().sum().sum()}</div><div class="stat-lbl">Missing Values</div></div>
    </div>
    """, unsafe_allow_html=True)

    uv = " &nbsp;•&nbsp; ".join([f"<b style='color:white'>{c}</b> → {df[c].nunique()}" for c in df.columns])
    st.markdown(f'<div class="unique-box">{uv}</div>', unsafe_allow_html=True)

    with st.expander("🔍 Lihat Sampel Data Mentah (5 Baris Pertama)"):
        st.dataframe(df.head(), use_container_width=True)
    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="card"><div class="card-title">Top 15 Target Distribusi Karir</div>', unsafe_allow_html=True)
        cc = df['Future Career'].value_counts().head(15)
        fig, ax = dark_fig(6, 5)
        colors_bar = ['#7c3aed' if i == 0 else '#4c1d95' if i < 3 else '#2d1b69' for i in range(len(cc))]
        ax.barh(cc.index[::-1], cc.values[::-1], color=colors_bar[::-1], edgecolor='none', height=.7)
        ax.set_xlabel('Jumlah Data', color='#6b7280', fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.xaxis.grid(True, color='#1e1b2e', alpha=.5); ax.set_axisbelow(True)
        plt.tight_layout(pad=1.5); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="card-title">Proporsi Karir Berdasarkan Gender</div>', unsafe_allow_html=True)
        if 'Gender' in df.columns:
            top_c = df['Future Career'].value_counts().head(8).index
            pivot = df[df['Future Career'].isin(top_c)].groupby(['Future Career','Gender']).size().unstack(fill_value=0)
            fig, ax = dark_fig(6, 5)
            pivot.plot(kind='barh', ax=ax, color=['#7c3aed','#a855f7'], alpha=.85, edgecolor='none', width=.6)
            ax.legend(facecolor='#0d0d1a', edgecolor='#1e1b2e', labelcolor='#9ca3af', fontsize=9)
            ax.spines[['top','right']].set_visible(False)
            ax.xaxis.grid(True, color='#1e1b2e', alpha=.5); ax.set_axisbelow(True)
            plt.tight_layout(pad=1.5); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown('<div class="card"><div class="card-title">Distribusi Nilai IPK (GPA)</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(5, 3.5)
        n, bins, patches = ax.hist(df['GPA'], bins=25, edgecolor='none')
        for i, patch in enumerate(patches):
            patch.set_facecolor(f'#{hex(int(80+i*3))[2:].zfill(2)}1b{hex(int(100+i*5))[2:].zfill(2)}')
        ax.set_xlabel('GPA', color='#6b7280', fontsize=9)
        ax.set_ylabel('Frekuensi', color='#6b7280', fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.yaxis.grid(True, color='#1e1b2e', alpha=.5); ax.set_axisbelow(True)
        plt.tight_layout(pad=1.5); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card"><div class="card-title">Komposisi Demografi Gender</div>', unsafe_allow_html=True)
        gc = df['Gender'].value_counts()
        fig, ax = dark_fig(5, 3.5)
        wedges, texts, autotexts = ax.pie(gc.values, labels=gc.index,
            colors=['#6d28d9','#a855f7'], autopct='%1.1f%%', startangle=90,
            textprops={'color':'#d1d5db','fontsize':11},
            wedgeprops={'edgecolor':'#0d0d1a','linewidth':3})
        for at in autotexts: at.set_color('white'); at.set_fontweight('bold')
        plt.tight_layout(pad=1.5); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    c5, c6 = st.columns(2, gap="large")
    with c5:
        st.markdown('<div class="card"><div class="card-title">Sebaran Umur vs IPK (Klaster Python Skill)</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(5, 3.8)
        cmap = {'Weak':'#ef4444','Average':'#f59e0b','Strong':'#22c55e'}
        for sk, col in cmap.items():
            s = df[df['Python']==sk]
            ax.scatter(s['Age'], s['GPA'], c=col, label=sk, alpha=.45, s=18, edgecolors='none')
        ax.legend(facecolor='#0d0d1a', edgecolor='#1e1b2e', labelcolor='#9ca3af', fontsize=9)
        ax.set_xlabel('Age', color='#6b7280', fontsize=9)
        ax.set_ylabel('GPA', color='#6b7280', fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.yaxis.grid(True, color='#1e1b2e', alpha=.4); ax.set_axisbelow(True)
        plt.tight_layout(pad=1.5); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with c6:
        st.markdown('<div class="card"><div class="card-title">Matriks Distribusi Skill Teknis</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1,3,figsize=(6,3.8))
        fig.patch.set_facecolor('#0d0d1a')
        for ax2, col, color in zip(axes,['Python','SQL','Java'],['#7c3aed','#6d28d9','#a855f7']):
            ax2.set_facecolor('#0d0d1a')
            counts = df[col].value_counts()
            ax2.bar(counts.index, counts.values, color=color, alpha=.9, edgecolor='none', width=.6)
            ax2.set_title(col, color='#d1d5db', fontsize=10, fontweight='600')
            ax2.tick_params(colors='#6b7280', labelsize=8)
            for spine in ax2.spines.values(): spine.set_color('#1e1b2e')
            ax2.yaxis.grid(True, color='#1e1b2e', alpha=.4); ax2.set_axisbelow(True)
        plt.tight_layout(pad=1.5); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — SOURCE CODE (SEKARANG DENGAN OUTPUT)
# ══════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style="margin-bottom:1.5rem; margin-top:0.5rem;">
        <div style="font-size:1.4rem;font-weight:700;color:white;">Source Code & Execution Output</div>
        <div style="font-size:13px;color:#6b7280;">Menampilkan seluruh kode Jupyter Notebook beserta hasil eksekusinya</div>
    </div>
    """, unsafe_allow_html=True)

    nb_path = "Final RF.ipynb"

    if not os.path.exists(nb_path):
        st.warning(f"File '{nb_path}' tidak ditemukan. Letakkan file di folder yang sama dengan app.py.")
    else:
        st.markdown(f'<p style="color:#4b5563;font-size:11px;margin-bottom:1rem;">Membaca dari: <b style="color:#a78bfa">{nb_path}</b></p>', unsafe_allow_html=True)
        try:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)

            code_count = 1
            for cell in nb.get('cells', []):
                source    = ''.join(cell.get('source', []))
                cell_type = cell.get('cell_type', '')
                
                if not source.strip():
                    continue

                if cell_type == 'markdown':
                    if source.strip().startswith('##'):
                        title = source.strip().lstrip('#').strip().split('\n')[0]
                        st.markdown(f'<div class="nb-sec">{title}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="nb-cell">
                            <div class="nb-cell-header"><span class="nb-md">Markdown</span></div>
                            <div style="padding:.8rem 1rem;color:#9ca3af;font-size:13px;line-height:1.7;">{source}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                elif cell_type == 'code':
                    # 1. Render Header dan Code Box (Tanpa border-radius bawah agar menyatu dengan output)
                    st.markdown(f"""
                    <div class="nb-cell-header" style="background:#110d1f; border:1px solid #1e1b2e;
                         border-radius:8px 8px 0 0; padding:.4rem 1rem; margin-bottom:-1px;">
                        <span class="nb-in">In [{code_count}]</span>
                        <span style="color:#4b5563;font-size:10px;">{nb_path}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.code(source, language='python')
                    
                    # 2. Proses Ekstraksi Output dari Notebook
                    cell_outputs = cell.get('outputs', [])
                    output_text = ""
                    
                    for out in cell_outputs:
                        # Menangkap output teks biasa (seperti perintah print atau df.info())
                        if out.get('output_type') in ['stream', 'execute_result']:
                            text_data = out.get('text', out.get('data', {}).get('text/plain', ''))
                            output_text += ''.join(text_data)
                        # Menangkap jikalau ada pesan error saat eksekusi di notebook
                        elif out.get('output_type') == 'error':
                            output_text += f"⚠️ Error: {''.join(out.get('traceback', []))}\n"

                    # 3. Render Box Output jika ada isinya
                    if output_text.strip():
                        st.markdown(f"""
                        <div class="nb-output">{output_text}</div>
                        """, unsafe_allow_html=True)
                    else:
                        # Jika cell tidak menghasilkan output, beri sedikit margin pemisah bawah yang rapi
                        st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
                        
                    code_count += 1
                    
        except Exception as e:
            st.error(f"Gagal membaca atau memproses file notebook: {e}")
# ══════════════════════════════════════════════
## ══════════════════════════════════════════════
# TAB 5 — TENTANG (INDENTASI DIPERBAIKI UTUH)
# ══════════════════════════════════════════════
with tab5:
    col_a, col_b = st.columns([1, 1.5], gap="large")

    with col_a:
        import base64

        foto_path = "profil.jpg"
        
        # Logika membaca gambar lokal dan mengubahnya jadi teks Base64
        if os.path.exists(foto_path):
            with open(foto_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            avatar_html = f'<img src="data:image/jpeg;base64,{encoded_string}" style="width:220px; height:220px; border-radius:5%; object-fit:cover; border: 2px solid #7c3aed; margin-bottom:15px;">'
        else:
            avatar_html = '<div style="font-size:48px; margin-bottom:10px;">👤</div>'

        # Tampilkan Card Profil dengan HTML biasa
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            {avatar_html}
            <div style="font-size:18px; font-weight:700; color:white;">Daffa Tsaqiif Pratama</div>
            <div style="font-size:13px; color:#a78bfa; margin-bottom:12px;">XI RPL 2 (Software Engineering)</div>
            <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-bottom:1rem;">
                <span class="badge">Machine Learning</span>
                <span class="badge">Python</span>
                <span class="badge">Data Science</span>
                <span class="badge">Artist</span>
            </div>
            <div style="font-size:12px;color:#6b7280;line-height:1.7;text-align:left;">
               Saya Daffa, Siswa SMKN 1 Purbalingga jurusan RPL yang sukak menggambar
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Card Kontak HTML biasa
        st.markdown("""
        <div class="card">
            <div class="card-title">Kontak</div>
            <div class="contact-item"><span class="contact-icon"></span> <a href="mailto:daffatsaqiefp@gmail.com" target="_blank" style="color: #a78bfa; text-decoration: none; font-weight: 500;">daffatsaqiefp@gmail.com</a></div>
            <div class="contact-item"><span class="contact-icon"></span> <a href="https://github.com/DaffaTsaqiep" target="_blank" style="color: #a78bfa; text-decoration: none; font-weight: 500;">github.com/DaffaTsaqiep</a></div>
            <div class="contact-item"><span class="contact-icon"></span> <a href="https://instagram.com/tsqieffp_" target="_blank" style="color: #a78bfa; text-decoration: none; font-weight: 500;">Instagram/@tsqieffp_</a></div>
            <div class="contact-item"><span class="contact-icon"></span> <a href="https://wa.me/6283875327389" target="_blank" style="color: #a78bfa; text-decoration: none; font-weight: 500;">+62 838-7532-7389</a></div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card">
            <div class="card-title"> Tentang Proyek</div>
            <div style="font-size:13px;color:#9ca3af;line-height:1.8;">
                Proyek ini merupakan implementasi Machine Learning untuk klasifikasi karir
                mahasiswa Ilmu Komputer. Dikembangkan sebagai bagian dari tugas akhir/penelitian
                dalam bidang Data Science dan Machine Learning.<br><br>
                Sistem ini menggunakan algoritma <b style="color:#a78bfa">Random Forest</b>
                yang telah di-tuning dengan GridSearchCV untuk mencapai akurasi optimal.
                Dataset yang digunakan terdiri dari data mahasiswa dengan berbagai profil akademik dan kemampuan teknis.
            </div>
        </div>
        
        <div class="card">
            <div class="card-title"> Tools yang Digunakan</div>
        """, unsafe_allow_html=True)
        
        techs = [
            ("Python 3.10", "Bahasa pemrograman utama", ""),
            ("Streamlit", "Framework web app interaktif", ""),
            ("Scikit-learn", "Library Machine Learning", ""),
            ("Pandas & NumPy", "Manipulasi dan analisis data", ""),
            ("Matplotlib & Seaborn", "Visualisasi data", ""),
            ("Joblib", "Serialisasi model ML", ""),
        ]
        for name, desc, icon in techs:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;
                        background:#110d1f;border:1px solid #1e1b2e;
                        border-radius:8px;padding:.6rem 1rem;margin-bottom:.5rem;">
                <span style="font-size:18px;">{icon}</span>
                <div>
                    <div style="color:white;font-size:13px;font-weight:600;">{name}</div>
                    <div style="color:#6b7280;font-size:11px;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-title"> Referensi</div>
            <div style="font-size:12px;color:#9ca3af;line-height:1.9;">
                ✦ Dataset: CS Students Career — Kaggle<br>
            </div>
        </div>
        """, unsafe_allow_html=True)
