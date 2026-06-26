import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from lifelines import KaplanMeierFitter, CoxPHFitter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Đường dẫn tương đối so với file script (luôn đúng dù chạy từ đâu)
BASE_DIR = Path(__file__).parent

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Causal Survival Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS (theo style Telco) ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117;
    color: #e0e0e0;
}
[data-testid="stAppViewContainer"] { background-color: #0f1117; }
[data-testid="stMain"]             { background-color: #0f1117; }

.stSelectbox > label, .stSlider > label, .stCheckbox > label {
    color: #9aa0b4 !important;
    font-size: 0.82rem !important;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #9aa0b4 !important; font-size: 0.78rem !important; }

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }
.kpi-card {
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-radius: 10px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.kpi-card.blue::before  { background: #4f8ef7; }
.kpi-card.amber::before { background: #f5a623; }
.kpi-card.red::before   { background: #e05c5c; }
.kpi-card.green::before { background: #4caf7d; }

.kpi-label { font-size: 0.72rem; color: #7a7f94; font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 8px; }
.kpi-value { font-size: 1.85rem; font-weight: 700; color: #f0f0f0; font-family: 'JetBrains Mono', monospace; line-height: 1; white-space: nowrap; }
.kpi-sub   { font-size: 0.75rem; color: #5a6070; margin-top: 6px; }

/* Section headers */
.section-header {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #4f8ef7;
    margin: 32px 0 14px; padding-bottom: 8px;
    border-bottom: 1px solid #1e2130;
}

/* Insight box */
.insight-box {
    background: #0f1a2e; border: 1px solid #1e3a5f;
    border-left: 3px solid #4f8ef7;
    border-radius: 6px; padding: 14px 18px; margin: 10px 0;
    font-size: 0.85rem; color: #c0cce0; line-height: 1.6;
}
.insight-box strong { color: #4f8ef7; }
.insight-box.warn   { border-left-color: #f5a623; background: #1e1a0f; border-color: #3a2e10; }
.insight-box.warn strong { color: #f5a623; }

/* Metric row */
.metric-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; }
.metric-item { flex: 1; min-width: 120px; background: #1e2235; border: 1px solid #2e3350; border-radius: 8px; padding: 12px 16px; }
.metric-item .m-label { font-size: 0.68rem; color: #9aa0b4; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500; }
.metric-item .m-value { font-size: 1.1rem; font-weight: 600; color: #ffffff; font-family: 'JetBrains Mono', monospace; margin-top: 4px; }

/* Table styling */
[data-testid="stDataFrame"] { background: #1a1d2e !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load data ─────────────────────────────────────────────────────────────────
def find_file(filename):
    """Tìm file ở cùng cấp với app.py, hoặc ở thư mục cha, hoặc trong subfolder 'app'."""
    candidates = [
        BASE_DIR / filename,
        BASE_DIR.parent / filename,
        BASE_DIR / 'app' / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Không tìm thấy '{filename}'. Đã thử các vị trí:\n{tried}"
    )

@st.cache_data
def load_data():
    df_ps      = pd.read_csv(find_file('df_ps.csv'))
    df_matched = pd.read_csv(find_file('df_matched.csv'))
    ate        = pd.read_csv(find_file('ate_results.csv'))
    return df_ps, df_matched, ate

df_ps, df_matched, ate_results = load_data()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 HR Causal Analysis")
    st.markdown("<div style='font-size:0.75rem;color:#5a6070;margin:-10px 0 20px'>Causal Inference +<br>Survival Analysis Study</div>", unsafe_allow_html=True)
    page = st.radio("", ["Act 1 — Naive View", "Act 2 — Uncovering Bias", "Act 3 — Causal Truth"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
<div style='font-size:0.72rem;color:#5a6070'>
<b style='color:#9aa0b4'>The Question</b><br>
Does training <i>cause</i> employees to stay longer — or do companies simply send training to people who were already going to stay?<br><br>
<b style='color:#9aa0b4'>The Method</b><br>
1. Naive KM (biased)<br>
2. Propensity Score Matching<br>
3. Causal Survival Estimate<br>
4. Bootstrap ATE with 95% CI<br><br>
<b style='color:#9aa0b4'>Key Finding</b><br>
Training has <b style='color:#e05c5c'>no causal effect</b> on retention.<br>
The apparent benefit is entirely selection bias.
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# ACT 1 — NAIVE VIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "Act 1 — Naive View":

    st.markdown("# Does Employee Training Actually Reduce Attrition?")
    st.markdown("<div style='color:#7a7f94;margin-top:-10px;margin-bottom:24px'>A Causal Inference + Survival Analysis Study &nbsp;·&nbsp; IBM HR Analytics Dataset &nbsp;·&nbsp; Propensity Score Matching &nbsp;·&nbsp; Cox Proportional Hazards</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Act 1 — The Naive View (Misleading)</div>', unsafe_allow_html=True)
    st.markdown("<div style='color:#9aa0b4;margin-bottom:20px'>Before adjusting for confounding, trained employees appear to have significantly better retention. <strong style='color:#e0e0e0'>But this is a statistical illusion.</strong></div>", unsafe_allow_html=True)

    # KPI cards — 3 metrics như bản gốc
    st.markdown("""
    <div class="kpi-grid" style="grid-template-columns: repeat(3, 1fr);">
      <div class="kpi-card blue">
        <div class="kpi-label">Total Employees</div>
        <div class="kpi-value">1,470</div>
        <div class="kpi-sub">IBM HR Analytics Dataset</div>
      </div>
      <div class="kpi-card amber">
        <div class="kpi-label">Attrition Rate</div>
        <div class="kpi-value">16.1%</div>
        <div class="kpi-sub">237 employees left</div>
      </div>
      <div class="kpi-card red">
        <div class="kpi-label">Naive Log-rank p</div>
        <div class="kpi-value">0.044 ⚠️</div>
        <div class="kpi-sub">Apparently significant</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KM Naive plot
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0f1117')
    ax.set_facecolor('#1a1d2e')
    for spine in ax.spines.values(): spine.set_color('#2a2d3e')
    ax.tick_params(colors='#7a7f94', labelsize=8)

    kmf = KaplanMeierFitter()
    for group, label, color in [(1, 'Trained (≥3×)', '#4caf7d'),
                                 (0, 'Control (<3×)',  '#e05c5c')]:
        mask = df_ps['Treated'] == group
        kmf.fit(df_ps.loc[mask, 'YearsAtCompany'],
                event_observed=df_ps.loc[mask, 'Attrition_flag'],
                label=label)
        kmf.plot_survival_function(ax=ax, color=color, ci_show=True)

    ax.set_title('Kaplan-Meier — NAIVE (Unadjusted)\nTraining appears protective — but is it?',
                 fontsize=11, color='#e0e0e0', pad=12)
    ax.set_xlabel('Years at Company', color='#7a7f94', fontsize=9)
    ax.set_ylabel('Survival Probability', color='#7a7f94', fontsize=9)
    ax.grid(True, alpha=0.1, color='white')
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=8, facecolor='#1a1d2e', labelcolor='white', edgecolor='#2a2d3e')
    plt.tight_layout()
    st.pyplot(fig, clear_figure=False, use_container_width=True)
    plt.close(fig)

    st.markdown("""
    <div class="insight-box warn">
        ⚠️ &nbsp;This result is <strong>not causal</strong>. Companies tend to train employees who are already loyal —
        creating an illusion of training effectiveness.
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# ACT 2 — UNCOVERING BIAS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Act 2 — Uncovering Bias":

    st.markdown("# Does Employee Training Actually Reduce Attrition?")
    st.markdown("<div style='color:#7a7f94;margin-top:-10px;margin-bottom:24px'>A Causal Inference + Survival Analysis Study &nbsp;·&nbsp; IBM HR Analytics Dataset &nbsp;·&nbsp; Propensity Score Matching &nbsp;·&nbsp; Cox Proportional Hazards</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Act 2 — Uncovering the Bias</div>', unsafe_allow_html=True)
    st.markdown("<div style='color:#9aa0b4;margin-bottom:20px'><strong style='color:#4f8ef7'>Propensity Score Matching</strong> creates a fair comparison: for each trained employee, we find a comparable untrained employee with the same observable characteristics.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Propensity Score Overlap</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='#0f1117')
        ax.set_facecolor('#1a1d2e')
        for spine in ax.spines.values(): spine.set_color('#2a2d3e')
        ax.tick_params(colors='#7a7f94', labelsize=8)

        ax.hist(df_ps[df_ps['Treated']==0]['propensity_score'],
                bins=25, alpha=0.6, color='#e05c5c', label='Control', density=True)
        ax.hist(df_ps[df_ps['Treated']==1]['propensity_score'],
                bins=25, alpha=0.6, color='#4caf7d', label='Trained', density=True)
        ax.set_xlabel('P(Trained | Covariates)', color='#7a7f94', fontsize=8)
        ax.set_title('Common Support — Matching is Valid', color='#e0e0e0', fontsize=10, pad=10)
        ax.legend(fontsize=8, facecolor='#1a1d2e', labelcolor='white', edgecolor='#2a2d3e')
        ax.grid(True, alpha=0.1, color='white')
        plt.tight_layout()
        st.pyplot(fig, clear_figure=False, use_container_width=True)
        plt.close(fig)
        st.markdown("<div style='font-size:0.75rem;color:#5a6070;margin-top:4px'>Perfect overlap → matching will find good pairs for nearly every treated employee.</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Covariate Balance (Love Plot)</div>', unsafe_allow_html=True)
        numeric_features = ['Age', 'MonthlyIncome', 'JobLevel', 'TotalWorkingYears',
                            'YearsAtCompany', 'JobSatisfaction', 'EnvironmentSatisfaction',
                            'WorkLifeBalance', 'JobInvolvement', 'DistanceFromHome']
        smd_before, smd_after = [], []
        for col in numeric_features:
            std = df_ps[col].std()
            sb  = abs(df_ps[df_ps['Treated']==1][col].mean() -
                      df_ps[df_ps['Treated']==0][col].mean()) / std if std > 0 else 0
            sa  = abs(df_matched[df_matched['Treated']==1][col].mean() -
                      df_matched[df_matched['Treated']==0][col].mean()) / std if std > 0 else 0
            smd_before.append(sb); smd_after.append(sa)

        fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0f1117')
        ax.set_facecolor('#1a1d2e')
        for spine in ax.spines.values(): spine.set_color('#2a2d3e')
        ax.tick_params(colors='#7a7f94', labelsize=8)

        y = range(len(numeric_features))
        ax.scatter(smd_before, y, color='#e05c5c', s=70, label='Before Matching', zorder=5)
        ax.scatter(smd_after,  y, color='#4caf7d', s=70, label='After Matching',  zorder=5)
        for i in y:
            ax.plot([smd_before[i], smd_after[i]], [i, i], color='#4a4f6a', alpha=0.6)
        ax.axvline(0.1, color='#f5a623', linestyle='--', alpha=0.7, label='Threshold 0.1')
        ax.set_yticks(list(y)); ax.set_yticklabels(numeric_features, fontsize=8, color='#9aa0b4')
        ax.set_xlabel('Standardized Mean Difference', color='#7a7f94', fontsize=8)
        ax.set_title('Love Plot — All Features Balanced After Matching', color='#e0e0e0', fontsize=10, pad=10)
        ax.legend(fontsize=8, facecolor='#1a1d2e', labelcolor='white', edgecolor='#2a2d3e')
        ax.grid(True, alpha=0.1, color='white', axis='x')
        plt.tight_layout()
        st.pyplot(fig, clear_figure=False, use_container_width=True)
        plt.close(fig)
        st.markdown(f"<div style='font-size:0.75rem;color:#5a6070;margin-top:4px'>Mean SMD: Before = {np.mean(smd_before):.3f} → After = {np.mean(smd_after):.3f} (−{(1-np.mean(smd_after)/np.mean(smd_before))*100:.0f}%)</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-box">
        <strong>Matching result:</strong> {len(df_matched)//2} matched pairs from {len(df_ps)} employees.
        Match rate: <strong>79.9%</strong> — sufficient for causal inference.
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# ACT 3 — CAUSAL TRUTH
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Act 3 — Causal Truth":

    st.markdown("# Does Employee Training Actually Reduce Attrition?")
    st.markdown("<div style='color:#7a7f94;margin-top:-10px;margin-bottom:24px'>A Causal Inference + Survival Analysis Study &nbsp;·&nbsp; IBM HR Analytics Dataset &nbsp;·&nbsp; Propensity Score Matching &nbsp;·&nbsp; Cox Proportional Hazards</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Act 3 — The Causal Truth</div>', unsafe_allow_html=True)

    # KPI cards — 3 metrics như bản gốc
    st.markdown("""
    <div class="kpi-grid" style="grid-template-columns: repeat(3, 1fr);">
      <div class="kpi-card red">
        <div class="kpi-label">Naive p-value</div>
        <div class="kpi-value">0.044</div>
        <div class="kpi-sub">Misleadingly significant</div>
      </div>
      <div class="kpi-card green">
        <div class="kpi-label">Matched p-value</div>
        <div class="kpi-value">0.570</div>
        <div class="kpi-sub">Not significant — bias removed</div>
      </div>
      <div class="kpi-card blue">
        <div class="kpi-label">Training HR (Cox)</div>
        <div class="kpi-value">0.937</div>
        <div class="kpi-sub">p=0.545 — no causal effect</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    kmf = KaplanMeierFitter()

    with col1:
        st.markdown('<div class="section-header">Naive vs Matched KM</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, facecolor='#0f1117')
        for ax, (dataset, title) in zip(axes, [
            (df_ps,      'NAIVE\n(biased)'),
            (df_matched, 'MATCHED\n(causal)')
        ]):
            ax.set_facecolor('#1a1d2e')
            for spine in ax.spines.values(): spine.set_color('#2a2d3e')
            ax.tick_params(colors='#7a7f94', labelsize=8)
            for group, label, color in [(1, 'Trained', '#4caf7d'), (0, 'Control', '#e05c5c')]:
                mask = dataset['Treated'] == group
                kmf.fit(dataset.loc[mask, 'YearsAtCompany'],
                        event_observed=dataset.loc[mask, 'Attrition_flag'], label=label)
                kmf.plot_survival_function(ax=ax, color=color, ci_show=True)
            ax.set_title(title, color='#e0e0e0', fontsize=10)
            ax.set_xlabel('Years', color='#7a7f94', fontsize=8)
            ax.grid(True, alpha=0.1, color='white')
            ax.set_ylim(0.4, 1.05)
            ax.legend(fontsize=7, facecolor='#1a1d2e', labelcolor='white', edgecolor='#2a2d3e')
        axes[0].set_ylabel('Survival Probability', color='#7a7f94', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=False, use_container_width=True)
        plt.close(fig)

    with col2:
        st.markdown('<div class="section-header">Average Treatment Effect — Bootstrap 95% CI</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='#0f1117')
        ax.set_facecolor('#1a1d2e')
        for spine in ax.spines.values(): spine.set_color('#2a2d3e')
        ax.tick_params(colors='#7a7f94', labelsize=8)

        ax.plot(ate_results['year'], ate_results['ate'],
                color='#4f8ef7', linewidth=2.5, label='ATE = S_trained(t) − S_control(t)')
        ax.fill_between(ate_results['year'], ate_results['ci_low'], ate_results['ci_high'],
                        color='#4f8ef7', alpha=0.2, label='95% Bootstrap CI')
        ax.axhline(0, color='#e05c5c', linestyle='--', linewidth=1.5, label='ATE = 0')
        ax.set_title('Training Effect on Survival\n(CI crosses 0 at every time point)',
                     color='#e0e0e0', fontsize=10, pad=10)
        ax.set_xlabel('Years at Company', color='#7a7f94', fontsize=8)
        ax.set_ylabel('Difference in Survival Probability', color='#7a7f94', fontsize=8)
        ax.legend(fontsize=8, facecolor='#1a1d2e', labelcolor='white', edgecolor='#2a2d3e')
        ax.grid(True, alpha=0.1, color='white')
        plt.tight_layout()
        st.pyplot(fig, clear_figure=False, use_container_width=True)
        plt.close(fig)

    st.markdown("""
    <div class="insight-box">
        <strong>Business Recommendation</strong><br>
        Training budgets are being spent on already-loyal employees — not at-risk ones.
        To reduce attrition, focus on:<br><br>
        1. <strong>Control overtime</strong> — the single largest driver<br>
        2. <strong>Clear promotion paths</strong> — each job level cuts risk by 21%<br>
        3. <strong>Increase job involvement</strong> — engagement, not training frequency
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 Technical Details"):
        st.markdown("""
| Step | Method | Result |
|------|--------|--------|
| Treatment | TrainingTimesLastYear ≥ 3 | 798 treated / 672 control |
| PS Model | Logistic Regression (18 covariates) | Accuracy 56.2% (ideal for PSM) |
| Matching | Nearest Neighbor 1:1, caliper=0.2×SD | 638 pairs, 79.9% match rate |
| Balance | Standardized Mean Difference | Mean SMD: 0.041 → 0.016 |
| Survival | Cox PH on matched sample | Concordance 0.860 |
| Causal Estimate | Bootstrap ATE (500 iterations) | CI crosses 0 at all time points |
""")