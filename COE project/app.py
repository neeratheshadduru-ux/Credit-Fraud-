import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#6C5CE7"
DANGER = "#FF4B6E"
SAFE = "#00C29A"
MUTED = "#8A8FA3"
BG_CARD = "#161A2B"

st.markdown(
    f"""
    <style>
        .stApp {{
            background: radial-gradient(circle at top left, #10131f 0%, #0b0d16 55%, #08090f 100%);
        }}
        section[data-testid="stSidebar"] {{
            background-color: #0d0f1a;
            border-right: 1px solid #22263c;
        }}
        h1, h2, h3, h4 {{
            color: #F4F4F8 !important;
            font-family: 'Segoe UI', sans-serif;
        }}
        p, span, label, .stMarkdown {{
            color: #C7CADB;
        }}
        div[data-testid="stMetric"] {{
            background: linear-gradient(145deg, {BG_CARD}, #1c2038);
            border: 1px solid #262b45;
            border-radius: 16px;
            padding: 18px 20px 10px 20px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.35);
        }}
        div[data-testid="stMetricLabel"] {{
            color: {MUTED} !important;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        div[data-testid="stMetricValue"] {{
            color: #F4F4F8 !important;
            font-weight: 700;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            border-bottom: 1px solid #22263c;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            color: {MUTED};
            border-radius: 10px 10px 0 0;
            padding: 10px 18px;
            font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {BG_CARD} !important;
            color: #FFFFFF !important;
            border-bottom: 3px solid {PRIMARY} !important;
        }}
        .verdict-card {{
            border-radius: 18px;
            padding: 28px;
            text-align: center;
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: 0.03em;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }}
        .subtle-card {{
            background: linear-gradient(145deg, {BG_CARD}, #1c2038);
            border: 1px solid #262b45;
            border-radius: 16px;
            padding: 16px 20px;
        }}
        div[data-testid="stForm"] {{
            background: {BG_CARD};
            border: 1px solid #262b45;
            border-radius: 18px;
            padding: 20px 24px;
        }}
        button[kind="primary"], button[kind="secondaryFormSubmit"] {{
            background-color: {PRIMARY} !important;
            border: none !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

CAT_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOL_COLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
NUM_COLS = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]

PLOTLY_TEMPLATE = "plotly_dark"


# ---------------------------------------------------------------------------
# Data + model
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv("credit_card_fraud_2026.csv")


REQUIRED_COLUMNS = {"transaction_id", "is_fraud", *CAT_COLS, *BOOL_COLS, *NUM_COLS}


def validate_csv(uploaded_file) -> tuple[bool, str]:
    try:
        preview = pd.read_csv(uploaded_file, nrows=5)
    except Exception as e:
        return False, f"Couldn't read that file as a CSV ({e})."
    uploaded_file.seek(0)
    missing = REQUIRED_COLUMNS - set(preview.columns)
    if missing:
        return False, f"Missing expected column(s): {', '.join(sorted(missing))}"
    return True, ""


@st.cache_resource
def train_model(df: pd.DataFrame):
    inputs = df.drop(columns=["transaction_id", "is_fraud"])
    target = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        inputs, target, train_size=0.75, random_state=1, stratify=target
    )

    X_train_cat = X_train.select_dtypes(include=["object", "str", "bool"]).copy()
    bc = X_train_cat.select_dtypes(include=["bool"]).columns
    X_train_cat[bc] = X_train_cat[bc].astype(int)
    X_train_cat = pd.get_dummies(X_train_cat, columns=CAT_COLS, drop_first=True)

    X_train_num = X_train.select_dtypes(include=["int64", "float64"])
    scaler = StandardScaler()
    X_train_num = pd.DataFrame(
        scaler.fit_transform(X_train_num), columns=X_train_num.columns, index=X_train_num.index
    )
    X_train_proc = pd.merge(X_train_num, X_train_cat, left_index=True, right_index=True)

    X_test_cat = X_test.select_dtypes(include=["object", "str", "bool"]).copy()
    bc = X_test_cat.select_dtypes(include=["bool"]).columns
    X_test_cat[bc] = X_test_cat[bc].astype(int)
    X_test_cat = pd.get_dummies(X_test_cat, columns=CAT_COLS, drop_first=True)
    X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)

    X_test_num = X_test.select_dtypes(include=["int64", "float64"])
    X_test_num = pd.DataFrame(
        scaler.transform(X_test_num), columns=X_test_num.columns, index=X_test_num.index
    )
    X_test_proc = pd.merge(X_test_num, X_test_cat, left_index=True, right_index=True)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_proc, y_train)
    y_proba = model.predict_proba(X_test_proc)[:, 1]

    return {
        "model": model,
        "scaler": scaler,
        "columns": X_train_proc.columns,
        "X_test": X_test_proc,
        "y_test": y_test,
        "y_proba": y_proba,
    }


def preprocess_single_row(row: pd.DataFrame, scaler: StandardScaler, columns: pd.Index):
    row = row.copy()
    row_cat = row.select_dtypes(include=["object", "str", "bool"]).copy()
    bc = row_cat.select_dtypes(include=["bool"]).columns
    row_cat[bc] = row_cat[bc].astype(int)
    row_cat = pd.get_dummies(row_cat, columns=CAT_COLS, drop_first=True)

    row_num = row.select_dtypes(include=["int64", "float64"])
    row_num = pd.DataFrame(scaler.transform(row_num), columns=row_num.columns, index=row_num.index)

    row_proc = pd.merge(row_num, row_cat, left_index=True, right_index=True)
    return row_proc.reindex(columns=columns, fill_value=0)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
if "data_source" not in st.session_state:
    st.session_state.data_source = None  # None | "sample" | "uploaded"
if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None

with st.sidebar:
    st.markdown("## 💳 FraudLens")
    st.caption("Credit card transaction risk dashboard")
    st.divider()
    if st.session_state.data_source is not None:
        source_label = "📄 your uploaded file" if st.session_state.data_source == "uploaded" else "📦 bundled sample"
        st.success(f"Data loaded from {source_label}")
        if st.button("↺ Change data source", width='stretch'):
            st.session_state.data_source = None
            st.session_state.uploaded_bytes = None
            st.rerun()
        st.divider()
        st.markdown("**Navigate**")
        st.caption("Use the tabs at the top to move between Overview, Explore, Model, and Predict.")

# ---------------------------------------------------------------------------
# Landing / upload screen — shown until a data source is chosen
# ---------------------------------------------------------------------------
if st.session_state.data_source is None:
    st.markdown("# 💳 FraudLens")
    st.markdown(
        "<p style='color:#8A8FA3;'>Upload a credit card transactions CSV to get started, "
        "or explore the dashboard with a bundled sample dataset.</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    up_col, sample_col = st.columns([1.4, 1], gap="large")

    with up_col:
        st.markdown("#### 📤 Upload your data")
        candidate = st.file_uploader(
            "Drag and drop a CSV here, or click to browse",
            type=["csv"],
            key="landing_uploader",
        )
        if candidate is not None:
            is_valid, message = validate_csv(candidate)
            if is_valid:
                st.session_state.uploaded_bytes = candidate.getvalue()
                st.session_state.data_source = "uploaded"
                st.rerun()
            else:
                st.error(f"⚠️ {message}")
        st.caption(
            "Expected columns: `transaction_id`, `is_fraud`, plus the usual transaction "
            "fields (amount, merchant category, device type, velocity score, etc.)."
        )

    with sample_col:
        st.markdown("#### 📦 Or use the sample")
        st.markdown(
            "<div class='subtle-card'>20,000 transactions<br>≈1.7% fraud rate<br>"
            "Ready to explore immediately.</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Use sample dataset →", width='stretch', type="primary"):
            st.session_state.data_source = "sample"
            st.rerun()

    st.stop()

# ---------------------------------------------------------------------------
# Load data + train model (only reached once a data source is chosen)
# ---------------------------------------------------------------------------
if st.session_state.data_source == "uploaded":
    import io
    df = load_data(io.BytesIO(st.session_state.uploaded_bytes))
else:
    df = load_data(None)

results = train_model(df)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# Credit Card Fraud Detection")
st.markdown(
    "<p style='color:#8A8FA3; margin-top:-10px;'>Logistic Regression model · trained on your transaction data</p>",
    unsafe_allow_html=True,
)

tab_overview, tab_explore, tab_model, tab_predict = st.tabs(
    ["🏠  Overview", "🔎  Explore", "📈  Model", "🔮  Predict"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Overview
# ---------------------------------------------------------------------------
with tab_overview:
    fraud_count = int(df["is_fraud"].sum())
    fraud_rate = df["is_fraud"].mean() * 100
    total_amount = df["amount_usd"].sum()
    avg_risk = df["merchant_risk_score"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions", f"{df.shape[0]:,}")
    c2.metric("Flagged Fraud", f"{fraud_count:,}", delta=f"{fraud_rate:.2f}% of total", delta_color="inverse")
    c3.metric("Total Volume", f"${total_amount:,.0f}")
    c4.metric("Avg Merchant Risk", f"{avg_risk:.1f}")
    c5.metric("Model ROC-AUC", f"{roc_auc_score(results['y_test'], results['y_proba']):.3f}")

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("#### Fraud vs. genuine transactions")
        pie_df = df["is_fraud"].map({0: "Genuine", 1: "Fraud"}).value_counts().reset_index()
        pie_df.columns = ["label", "count"]
        fig = px.pie(
            pie_df, names="label", values="count", hole=0.6,
            color="label", color_discrete_map={"Genuine": SAFE, "Fraud": DANGER},
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=True, margin=dict(t=10, b=10, l=10, r=10), height=320)
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, width='stretch')

    with right:
        st.markdown("#### Fraud rate by merchant category")
        cat_rate = df.groupby("merchant_category")["is_fraud"].mean().sort_values(ascending=True) * 100
        fig = px.bar(
            cat_rate, orientation="h", labels={"value": "Fraud rate (%)", "merchant_category": ""},
            color=cat_rate.values, color_continuous_scale=["#2E3350", DANGER],
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, coloraxis_showscale=False,
                           margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Transaction amount: fraud vs. genuine")
    fig = px.box(
        df, x="is_fraud", y="amount_usd", color="is_fraud",
        color_discrete_map={0: SAFE, 1: DANGER},
        labels={"is_fraud": "", "amount_usd": "Amount (USD)"},
    )
    fig.update_xaxes(tickvals=[0, 1], ticktext=["Genuine", "Fraud"])
    fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=340)
    st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------------
# TAB 2 — Explore
# ---------------------------------------------------------------------------
with tab_explore:
    st.markdown("#### Explore any feature")
    c1, c2 = st.columns(2)
    with c1:
        cat_choice = st.selectbox("Categorical / boolean feature", CAT_COLS + BOOL_COLS)
        fig = px.histogram(
            df, x=cat_choice, color="is_fraud", barmode="group",
            color_discrete_map={0: SAFE, 1: DANGER},
            category_orders={cat_choice: df[cat_choice].value_counts().index.tolist()},
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, legend_title_text="is_fraud",
                           margin=dict(t=10, b=10, l=10, r=10), height=380)
        st.plotly_chart(fig, width='stretch')

    with c2:
        num_choice = st.selectbox("Numerical feature", NUM_COLS)
        fig = px.histogram(
            df, x=num_choice, color="is_fraud", barmode="overlay", nbins=40, opacity=0.65,
            color_discrete_map={0: SAFE, 1: DANGER},
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, legend_title_text="is_fraud",
                           margin=dict(t=10, b=10, l=10, r=10), height=380)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Correlation heatmap")
    numish = df.select_dtypes(include=["int64", "float64", "bool"])
    corr = numish.corr()
    fig = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(t=10, b=10, l=10, r=10), height=560)
    st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------------
# TAB 3 — Model
# ---------------------------------------------------------------------------
with tab_model:
    st.markdown("#### Logistic Regression (`class_weight='balanced'`)")
    st.caption("Fraud is rare (~1.7%), so recall and ROC-AUC matter more here than raw accuracy.")

    y_test = results["y_test"]
    y_proba = results["y_proba"]
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
    y_pred = (y_proba >= threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
    c2.metric("Fraud precision", f"{report['1']['precision']:.2f}")
    c3.metric("Fraud recall", f"{report['1']['recall']:.2f}")
    c4.metric("Fraud F1", f"{report['1']['f1-score']:.2f}")

    left, right = st.columns(2)
    with left:
        st.markdown("###### Confusion matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale=["#1c2038", PRIMARY],
            labels=dict(x="Predicted", y="Actual", color="count"),
            x=["Genuine", "Fraud"], y=["Genuine", "Fraud"],
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(t=10, b=10, l=10, r=10), height=360)
        st.plotly_chart(fig, width='stretch')

    with right:
        st.markdown("###### ROC curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color=PRIMARY, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=MUTED, dash="dash")))
        fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                           margin=dict(t=10, b=10, l=10, r=10), height=360)
        st.plotly_chart(fig, width='stretch')

    st.markdown("###### Top feature coefficients")
    coef_df = pd.DataFrame(
        {"feature": results["columns"], "coefficient": results["model"].coef_[0]}
    ).sort_values("coefficient", key=abs, ascending=False).head(15)
    fig = px.bar(
        coef_df.sort_values("coefficient"), x="coefficient", y="feature", orientation="h",
        color="coefficient", color_continuous_scale=["#2E7DFF", "#2E3350", DANGER],
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10), height=440)
    st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------------
# TAB 4 — Predict
# ---------------------------------------------------------------------------
with tab_predict:
    st.markdown("#### Score a new transaction")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            amount_usd = st.number_input("Amount (USD)", min_value=0.0, value=80.0)
            merchant_category = st.selectbox("Merchant category", sorted(df.merchant_category.unique()))
            card_type = st.selectbox("Card type", sorted(df.card_type.unique()))
            auth_method = st.selectbox("Auth method", sorted(df.auth_method.unique()))
            channel = st.selectbox("Channel", sorted(df.channel.unique()))
            device_type = st.selectbox("Device type", sorted(df.device_type.unique()))

        with c2:
            hours_since_last_txn = st.number_input("Hours since last txn", min_value=0.0, value=5.0)
            txn_count_last_24h = st.number_input("Txn count last 24h", min_value=0, value=2)
            distance_from_home_km = st.number_input("Distance from home (km)", min_value=0.0, value=10.0)
            card_age_months = st.number_input("Card age (months)", min_value=0.0, value=40.0)
            customer_age = st.number_input("Customer age", min_value=18.0, value=35.0)
            account_balance_usd = st.number_input("Account balance (USD)", min_value=0.0, value=2000.0)
            velocity_score = st.number_input("Velocity score", min_value=0.0, value=10.0)

        with c3:
            cvv_retry_count = st.number_input("CVV retry count", min_value=0, value=0)
            time_of_day_hour = st.slider("Time of day (hour)", 0, 23, 12)
            day_of_week = st.slider("Day of week (0=Mon)", 0, 6, 2)
            merchant_risk_score = st.number_input("Merchant risk score", min_value=0.0, value=20.0)
            prior_disputes = st.number_input("Prior disputes", min_value=0, value=0)
            is_foreign_transaction = st.checkbox("Foreign transaction")
            is_new_merchant = st.checkbox("New merchant")
            used_vpn = st.checkbox("VPN used")
            ip_country_mismatch = st.checkbox("IP/country mismatch")
            billing_shipping_mismatch = st.checkbox("Billing/shipping mismatch")
            is_ai_generated_scam_attempt = st.checkbox("Flagged as AI-generated scam pattern")

        pred_threshold = st.slider("Prediction threshold", 0.05, 0.95, 0.5, 0.05, key="pred_thresh")
        submitted = st.form_submit_button("Score transaction", width='stretch')

    if submitted:
        new_row = pd.DataFrame([{
            "amount_usd": amount_usd, "merchant_category": merchant_category, "card_type": card_type,
            "auth_method": auth_method, "channel": channel, "device_type": device_type,
            "is_foreign_transaction": is_foreign_transaction, "hours_since_last_txn": hours_since_last_txn,
            "txn_count_last_24h": txn_count_last_24h, "distance_from_home_km": distance_from_home_km,
            "card_age_months": card_age_months, "customer_age": customer_age,
            "account_balance_usd": account_balance_usd, "is_new_merchant": is_new_merchant,
            "used_vpn": used_vpn, "ip_country_mismatch": ip_country_mismatch,
            "billing_shipping_mismatch": billing_shipping_mismatch, "cvv_retry_count": cvv_retry_count,
            "velocity_score": velocity_score, "time_of_day_hour": time_of_day_hour,
            "day_of_week": day_of_week, "is_ai_generated_scam_attempt": is_ai_generated_scam_attempt,
            "merchant_risk_score": merchant_risk_score, "prior_disputes": prior_disputes,
        }])

        row_proc = preprocess_single_row(new_row, results["scaler"], results["columns"])
        proba = results["model"].predict_proba(row_proc)[0, 1]
        is_fraud = proba >= pred_threshold

        color = DANGER if is_fraud else SAFE
        label = "⚠️ LIKELY FRAUD" if is_fraud else "✅ LOOKS GENUINE"

        st.write("")
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.markdown(
                f"<div class='verdict-card' style='background:{color}22; border:1px solid {color}; color:{color};'>{label}</div>",
                unsafe_allow_html=True,
            )
        with c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "bgcolor": "#1c2038",
                    "steps": [
                        {"range": [0, 50], "color": "#1c2038"},
                        {"range": [50, 100], "color": "#2a1f2c"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "value": pred_threshold * 100},
                },
            ))
            fig.update_layout(template=PLOTLY_TEMPLATE, height=220, margin=dict(t=10, b=10, l=20, r=20))
            st.plotly_chart(fig, width='stretch')
