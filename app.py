import os

import joblib
import pandas as pd
import streamlit as st
from tensorflow import keras

AUTHOR = "Abhishek Pandey"

st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {background: radial-gradient(1100px 600px at 8% -10%, #1d2a5c 0%, #0b1020 55%) fixed;}
.block-container {padding-top: 1.6rem; max-width: 1180px;}
h1, h2, h3, h4, p, label, span, div {color: #e8ecf8;}
.hero {padding: 26px 30px; border-radius: 20px; margin-bottom: 18px;
  background: linear-gradient(135deg, rgba(79,124,255,0.22), rgba(155,92,255,0.16));
  border: 1px solid rgba(255,255,255,0.10);}
.hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px;}
.hero p {margin: 6px 0 12px 0; color: #aab3d1; font-size: 1.02rem;}
.chip {display: inline-block; padding: 4px 12px; margin: 0 8px 6px 0; border-radius: 999px;
  font-size: 0.78rem; color: #cdd6f4; background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.10);}
.card {background: rgba(255,255,255,0.045); border: 1px solid rgba(255,255,255,0.09);
  border-radius: 18px; padding: 20px 22px; margin-bottom: 14px;}
.tile {background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09);
  border-radius: 16px; padding: 18px 18px 14px 18px;}
.tile .v {font-size: 2rem; font-weight: 800; line-height: 1.1;}
.tile .l {font-size: 0.85rem; color: #9aa4c7; margin-top: 4px;}
.badge {display: inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 700;
  font-size: 0.85rem; letter-spacing: 0.5px; margin-top: 6px;}
.decision {margin-top: 10px; font-size: 1.15rem; font-weight: 700;}
.muted {color: #8f99bd; font-size: 0.85rem;}
.layer {display: flex; justify-content: space-between; align-items: center; padding: 10px 14px;
  margin-bottom: 8px; border-radius: 12px; background: rgba(255,255,255,0.05);
  border-left: 4px solid #4f7cff;}
.layer b {font-size: 0.95rem;}
.layer span {font-size: 0.82rem; color: #9aa4c7;}
div[data-testid="stForm"] {border: 1px solid rgba(255,255,255,0.09); border-radius: 18px;
  background: rgba(255,255,255,0.03); padding: 18px 20px;}
.stButton > button, div[data-testid="stFormSubmitButton"] > button {
  width: 100%; border: 0; border-radius: 12px; padding: 0.65rem 1rem; font-weight: 700;
  color: #fff; background: linear-gradient(135deg, #4f7cff, #9b5cff);}
button[data-baseweb="tab"] {font-weight: 600;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Model ----------
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model("loan_ann_model.keras")
    pre = joblib.load("preprocessor.joblib")
    return model, pre


try:
    model, pre = load_artifacts()
except Exception as e:  # noqa: BLE001
    st.error("Model files not found. Run `python train_ann.py` first in this folder.")
    st.caption(str(e))
    st.stop()


def predict(row: dict) -> float:
    x = pre.transform(pd.DataFrame([row]))
    if hasattr(x, "toarray"):
        x = x.toarray()
    return float(model.predict(x, verbose=0)[0][0])


def gauge_html(prob: float) -> str:
    if prob < 0.30:
        color, label, icon, decision = "#22c55e", "LOW RISK", "&#10003;", "APPROVE"
    elif prob < 0.50:
        color, label, icon, decision = "#f59e0b", "MEDIUM RISK", "!", "APPROVE (needs review)"
    else:
        color, label, icon, decision = "#ef4444", "HIGH RISK", "&#10005;", "REJECT"
    arc = 251.33
    return (
        '<div class="card" style="text-align:center">'
        '<svg viewBox="0 0 200 122" width="100%" style="max-width:380px">'
        '<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#2a3150" '
        'stroke-width="16" stroke-linecap="round"/>'
        f'<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" '
        f'stroke-width="16" stroke-linecap="round" stroke-dasharray="{arc * prob:.2f} {arc}"/>'
        f'<text x="100" y="90" text-anchor="middle" font-size="30" font-weight="800" fill="#ffffff">{prob:.1%}</text>'
        '<text x="100" y="112" text-anchor="middle" font-size="9.5" fill="#9aa4c7">probability of default</text>'
        "</svg>"
        f'<div class="badge" style="background:{color}22;color:{color};border:1px solid {color}66">{icon} {label}</div>'
        f'<div class="decision">Decision: {decision}</div>'
        "</div>"
    )


def tile_html(value: str, label: str) -> str:
    return f'<div class="tile"><div class="v">{value}</div><div class="l">{label}</div></div>'


# ---------- Header ----------
st.markdown(
    '<div class="hero"><h1>Loan Default Predictor</h1>'
    "<p>An Artificial Neural Network that estimates the chance an applicant will default on a loan.</p>"
    '<span class="chip">TensorFlow / Keras</span><span class="chip">100,000 applicants</span>'
    '<span class="chip">15 input features</span><span class="chip">ROC-AUC 0.899</span>'
    '<span class="chip">Synthetic dataset</span></div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Predict", "Model performance", "About"])

# ---------- Tab 1: Predict ----------
with tab1:
    left, right = st.columns([1.25, 1], gap="large")

    with left:
        with st.form("applicant"):
            st.markdown("**Applicant**")
            a1, a2, a3 = st.columns(3)
            age = a1.slider("Age", 21, 65, 35)
            education = a2.selectbox("Education", ["High School", "Graduate", "Postgraduate"], index=1)
            marital = a3.selectbox("Marital status", ["Single", "Married", "Divorced"], index=1)
            b1, b2, b3 = st.columns(3)
            emp_type = b1.selectbox("Employment type", ["Salaried", "Self-Employed", "Business", "Unemployed"])
            emp_years = b2.slider("Employment years", 0, 40, 10)
            home = b3.selectbox("Home ownership", ["Rent", "Own", "Mortgage"])
            dependents = st.slider("Dependents", 0, 5, 2)

            st.markdown("**Loan**")
            c1, c2 = st.columns(2)
            income = c1.number_input("Yearly income (₹)", 120000, 5000000, 600000, step=10000)
            loan_amount = c2.number_input("Loan amount (₹)", 50000, 8000000, 900000, step=10000)
            d1, d2 = st.columns(2)
            term = d1.selectbox("Loan term (months)", [12, 24, 36, 48, 60, 84, 120], index=4)
            purpose = d2.selectbox("Loan purpose", ["Home", "Car", "Education", "Personal", "Business"], index=1)

            st.markdown("**Credit history**")
            e1, e2 = st.columns(2)
            credit_score = e1.slider("Credit score", 300, 900, 710)
            utilization = e2.slider("Credit utilization", 0.0, 1.0, 0.30)
            f1, f2 = st.columns(2)
            existing = f1.slider("Existing loans", 0, 6, 1)
            late = f2.slider("Late payments", 0, 10, 0)

            submitted = st.form_submit_button("Predict default risk")

    if submitted:
        row = {
            "age": age, "income": income, "loan_amount": loan_amount,
            "loan_term_months": term, "credit_score": credit_score,
            "employment_years": emp_years, "num_existing_loans": existing,
            "num_late_payments": late, "credit_utilization": utilization,
            "dependents": dependents, "education": education,
            "employment_type": emp_type, "home_ownership": home,
            "loan_purpose": purpose, "marital_status": marital,
        }
        st.session_state["result"] = {
            "prob": predict(row),
            "ratio": loan_amount / income,
            "monthly": loan_amount / term,
        }

    with right:
        res = st.session_state.get("result")
        if res is None:
            st.markdown(
                '<div class="card" style="text-align:center;padding:60px 20px">'
                '<div style="font-size:2.4rem">🏦</div>'
                '<div class="decision">Fill the form and press Predict</div>'
                '<div class="muted">The risk meter will appear here.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(gauge_html(res["prob"]), unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.markdown(tile_html(f"{res['ratio']:.2f}x", "Loan-to-income ratio"), unsafe_allow_html=True)
            m2.markdown(tile_html(f"₹{res['monthly']:,.0f}", "Principal per month (no interest)"), unsafe_allow_html=True)
            st.markdown(
                '<p class="muted" style="margin-top:12px">Decision rule: probability below 50% is approved. '
                "This is a learning project, not financial advice.</p>",
                unsafe_allow_html=True,
            )

# ---------- Tab 2: Model performance ----------
with tab2:
    t1, t2, t3, t4 = st.columns(4)
    t1.markdown(tile_html("82%", "Accuracy"), unsafe_allow_html=True)
    t2.markdown(tile_html("0.899", "ROC-AUC"), unsafe_allow_html=True)
    t3.markdown(tile_html("0.78", "Precision (default)"), unsafe_allow_html=True)
    t4.markdown(tile_html("0.77", "Recall (default)"), unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    p1, p2 = st.columns([1.2, 1], gap="large")
    with p1:
        st.markdown("**Per-class performance (20,000 unseen test records)**")
        perf = pd.DataFrame(
            {"Repaid (class 0)": [0.85, 0.86, 0.85], "Default (class 1)": [0.78, 0.77, 0.78]},
            index=["Precision", "Recall", "F1-score"],
        )
        st.bar_chart(perf, color=["#2a78d6", "#eb6834"])
    with p2:
        st.markdown("**What these numbers mean**")
        st.markdown(
            '<div class="card"><p><b>Accuracy 82%</b>: about 82 of every 100 applicants are classified correctly.</p>'
            "<p><b>ROC-AUC 0.899</b>: the model ranks a risky applicant above a safe one about 90% of the time.</p>"
            "<p><b>Precision 0.78</b>: when it says <i>default</i>, it is right 78% of the time.</p>"
            '<p style="margin:0"><b>Recall 0.77</b>: it catches 77% of the applicants who really default.</p></div>',
            unsafe_allow_html=True,
        )
    if os.path.exists("ann_results_card.png"):
        st.image("ann_results_card.png", use_container_width=True)

# ---------- Tab 3: About ----------
with tab3:
    x1, x2 = st.columns(2, gap="large")
    with x1:
        st.markdown("**Network architecture**")
        for name, desc in [
            ("Input", "28 features after encoding"),
            ("Dense 64", "ReLU + BatchNorm + Dropout 0.3"),
            ("Dense 32", "ReLU + Dropout 0.2"),
            ("Dense 16", "ReLU"),
            ("Output", "Sigmoid, default probability"),
        ]:
            st.markdown(f'<div class="layer"><b>{name}</b><span>{desc}</span></div>', unsafe_allow_html=True)
        st.markdown('<p class="muted">4,737 parameters | Adam optimizer | early stopping</p>', unsafe_allow_html=True)
    with x2:
        st.markdown("**Pipeline**")
        st.markdown(
            '<div class="card"><p>1. Synthetic dataset of 100,000 loan applicants (missing values included)</p>'
            "<p>2. Median imputation, standard scaling, one-hot encoding</p>"
            "<p>3. ANN trained with early stopping on a validation split</p>"
            "<p>4. Model and preprocessor saved, then served in this Streamlit app</p>"
            '<p style="margin:0"><b>Stack:</b> Python, Pandas, scikit-learn, TensorFlow/Keras, Streamlit</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card"><b>Note</b><p class="muted" style="margin:6px 0 0 0">The data is synthetic and '
            "generated for learning and portfolio purposes. It does not contain real customer information.</p></div>",
            unsafe_allow_html=True,
        )

st.markdown(f'<p class="muted" style="text-align:center;margin-top:18px">Built by {AUTHOR}</p>', unsafe_allow_html=True)
