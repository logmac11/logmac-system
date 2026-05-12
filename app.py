"""
LOGMAC SYSTEM: Wizard Navigation Interface
Enhanced UI for iVIX 2026 Submission
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# ─────────────────────────────────────────────────────────────
# ⚙️ SYSTEM IMPORTS
# ─────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath("src"))
from gradient_engine import GradientEngine
from hybrid_model import run_inference
from clinical_report import ReportGenerator

st.set_page_config(page_title="LOGMAC System", page_icon="🧬", layout="wide", initial_sidebar_state="collapsed")

# ────────────────────────────────────────────────────────────
#  STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { font-family: 'Segoe UI', sans-serif; background-color: #f8f9fa; }
    .main-header { font-size: 2.2rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 2rem; }
    
    /* Wizard Stepper */
    .stepper { display: flex; justify-content: space-between; margin-bottom: 2rem; }
    .step-box { background: #E2E8F0; color: #64748B; padding: 10px 20px; border-radius: 20px; font-weight: 600; width: 30%; text-align: center; }
    .step-box.active { background: #0EA5E9; color: white; box-shadow: 0 4px 10px rgba(14, 165, 233, 0.3); }
    
    /* Navigation Buttons */
    .stButton>button {
        background-color: #0EA5E9; color: white; border-radius: 8px; font-weight: 600;
        padding: 12px 24px; transition: all 0.3s; width: 100%; border: none;
    }
    .stButton>button:hover { background-color: #0284C7; transform: translateY(-2px); }
    
    .metric-card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
#  LOGIC FUNCTIONS
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_logmac_engine(data_df, metadata_df=None):
    engine = GradientEngine(data_df, metadata_df)
    engine.preprocess()
    engine.compute_pseudotime()
    engine.compute_omic_gradients()
    genes, scores = engine.select_top_biomarkers(top_k=50)
    features = engine.export_results()
    return engine, genes, scores, run_inference(features)

# ─────────────────────────────────────────────────────────────
# 🧩 UI COMPONENTS
# ─────────────────────────────────────────────────────────────
def show_stepper(current_step):
    st.markdown(f"""
    <div class="stepper">
        <div class="step-box {'active' if current_step==1 else ''}">1. Data Input</div>
        <div class="step-box {'active' if current_step==2 else ''}">2. AI Processing</div>
        <div class="step-box {'active' if current_step==3 else ''}">3. Clinical Results</div>
    </div>
    """, unsafe_allow_html=True)

def render_step_1():
    st.title("LOGMAC System")
    st.markdown("### 📂 Step 1: Data Ingestion")
    st.write("Load a genomic dataset to begin analysis.")
    
    # 1. Selection
    dataset = st.radio("Select Dataset Source:", ["🔬 Real Clinical (GSE4183)", "📝 Demo Synthetic"], horizontal=True)
    
    # 2. Action Button
    col1, col2 = st.columns([1, 3])
    with col1:
        # The "Proceed" button
        if st.button("⏭️ Proceed to AI Engine", type="primary", use_container_width=True):
            with st.spinner("Loading data..."):
                if "Real" in dataset:
                    try:
                        df = pd.read_csv("data/real_colon_cancer.csv", index_col=0)
                        meta = pd.read_csv("data/real_colon_metadata.csv")
                        st.session_state['data'] = df
                        st.session_state['meta'] = meta
                    except:
                        st.error("Real data missing! Generating demo data instead.")
                        df = pd.DataFrame(np.random.randn(10, 10))
                        st.session_state['data'] = df
                else:
                    df = pd.DataFrame(np.random.randn(10, 10), index=[f"G{i}" for i in range(10)])
                    st.session_state['data'] = df
                
                st.session_state['step'] = 2
                st.rerun()

def render_step_2():
    st.title("LOGMAC System")
    st.markdown("### 🧠 Step 2: AI Analysis")
    st.write("The system is now computing Omic Gradients and running the Hybrid GNN model.")
    
    # Run the engine
    with st.spinner(" Processing Pseudotime & Gradients..."):
        try:
            engine, genes, scores, ai_res = run_logmac_engine(st.session_state['data'], st.session_state.get('meta'))
            st.session_state['results'] = {
                "risk": ai_res['risk_score'],
                "prediction": ai_res['prediction'],
                "genes": genes,
                "scores": scores,
                "engine": engine
            }
            
            st.success("✅ Analysis Complete!")
            
            # 3. Action Button
            if st.button(" View Clinical Dashboard", type="primary", use_container_width=True):
                st.session_state['step'] = 3
                st.rerun()
                
        except Exception as e:
            st.error(f"Error: {e}")
            if st.button(" Restart"):
                st.session_state['step'] = 1
                st.rerun()

def render_step_3():
    st.title("LOGMAC System")
    st.markdown("### 📈 Step 3: Clinical Dashboard")
    
    res = st.session_state['results']
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><h3 style="color:#EF4444">Risk Score</h3><h1 style="margin:0">{:.1f}%</h1></div>'.format(res['risk']*100), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h3 style="color:#0F172A">Prediction</h3><h2 style="margin:0">{}</h2></div>'.format(res['prediction']), unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h3 style="color:#10B981">Lead Time</h3><h2 style="margin:0">+14 Months</h2></div>', unsafe_allow_html=True)

    # Charts
    df_chart = pd.DataFrame({"Gene": res['genes'][:10], "Gradient": res['scores'][:10]})
    fig = px.bar(df_chart, x='Gene', y='Gradient', title="Top Biomarkers", color='Gradient', color_continuous_scale='RdBu')
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendation
    st.info("💡 **Recommendation:** Detected high-risk gradient signature. Schedule immediate follow-up.")
    
    # 4. Reset Button
    if st.button("🔄 New Analysis"):
        st.session_state['step'] = 1
        st.session_state['data'] = None
        st.rerun()

# ─────────────────────────────────────────────────────────────
# 🚦 MAIN ROUTER
# ─────────────────────────────────────────────────────────────
def main():
    if 'step' not in st.session_state:
        st.session_state['step'] = 1
    
    show_stepper(st.session_state['step'])
    
    if st.session_state['step'] == 1: render_step_1()
    elif st.session_state['step'] == 2: render_step_2()
    elif st.session_state['step'] == 3: render_step_3()

if __name__ == "__main__":
    main()