"""
HONY (Humans of New York) — Phase 3: Streamlit Analytics Dashboard
====================================================================
Interactive dashboard integrating Phase 2 models for:
  - Live post prediction (tier + engagement score)
  - Portfolio analytics on the full predictions dataset
  - Model performance exploration

Run:
  pip install streamlit scikit-learn xgboost pandas matplotlib seaborn plotly
  streamlit run hony_dashboard.py

Expected files (same directory):
  outputs/hony_classifier.pkl
  outputs/hony_regressor.pkl
  outputs/hony_predictions.csv
  outputs/model_performance_summary.txt   (optional, for display)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HONY Engagement Analytics",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette (mirrors Phase 1/2) ───────────────────────────────────────────────
TIER_COLORS  = {"Low": "#E63946", "Mid": "#457B9D", "High": "#1D3557", "Viral": "#A8DADC"}
TIER_ORDER   = ["Low", "Mid", "High", "Viral"]
TOPIC_LABELS = [
    "Family & Childhood", "Work & Career", "Love & Relationships",
    "Health & Struggle",  "Immigration & Identity", "Loss & Grief",
    "Dreams & Ambition",  "War & Conflict",
]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .metric-card {
        background: white; border-radius: 10px;
        padding: 1rem 1.2rem; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #457B9D;
    }
    .tier-badge {
        display: inline-block; padding: 0.25rem 0.75rem;
        border-radius: 20px; font-weight: 700;
        font-size: 1.1rem; letter-spacing: 0.03em;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
    .disclaimer {
        background: #fff8e1; border-left: 4px solid #F4A261;
        padding: 0.6rem 1rem; border-radius: 4px;
        font-size: 0.82rem; color: #7a5c00; margin-top: 0.5rem;
    }
    h1 { color: #1D3557; }
    h2, h3 { color: #1D3557; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path("outputs")

@st.cache_resource
def load_models():
    with open(BASE / "hony_classifier.pkl", "rb") as f:
        cls_bundle = pickle.load(f)
    with open(BASE / "hony_regressor.pkl", "rb") as f:
        reg_bundle = pickle.load(f)
    return cls_bundle, reg_bundle

@st.cache_data
def load_predictions():
    df = pd.read_csv(BASE / "hony_predictions.csv")
    df["dt"] = pd.to_datetime(df["dt"])
    df["year"] = df["dt"].dt.year
    return df

try:
    cls_bundle, reg_bundle = load_models()
    df_preds = load_predictions()
    MODELS_LOADED = True
except FileNotFoundError as e:
    MODELS_LOADED = False
    st.error(f"⚠️ Could not load model files: {e}\n\nMake sure the `outputs/` folder "
             f"with `hony_classifier.pkl`, `hony_regressor.pkl`, and "
             f"`hony_predictions.csv` is in the same directory as this script.")
    st.stop()

cls_model    = cls_bundle["model"]
le_sentiment = cls_bundle["le_sentiment"]
le_topic     = cls_bundle["le_topic"]
FEAT_NAMES   = cls_bundle["feature_names"]
reg_model    = reg_bundle["model"]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — feature vector builder
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vector(
    text: str,
    hour: int,
    dayofweek: int,
    month: int,
    sentiment_label: str,
    topic_label: str,
) -> pd.DataFrame:
    """
    Derive all 22 engineered features from raw inputs, matching exactly what
    Phase 1 produced (same logic, same column names).
    Uses VADER for sentiment scores; everything else is rule-based.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    vs    = analyzer.polarity_scores(text)
    words = text.split()
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s for s in sents if s.strip()]

    word_count          = len(words)
    sentence_count      = max(len(sents), 1)
    avg_sentence_length = word_count / sentence_count
    paragraph_count     = max(len([p for p in text.split('\n') if p.strip()]), 1)
    question_count      = text.count('?')
    exclaim_count       = text.count('!')
    quote_count         = text.count('"') // 2
    dialogue_lines      = sum(1 for s in sents if s.strip().startswith('"'))
    dialogue_ratio      = dialogue_lines / sentence_count
    first_person_words  = {"i", "me", "my", "mine", "myself", "we", "our", "us"}
    fp_count            = sum(1 for w in words if w.lower() in first_person_words)
    first_person_ratio  = fp_count / max(word_count, 1)
    upper_words         = sum(1 for w in words if w.isupper() and len(w) > 1)
    uppercase_ratio     = upper_words / max(word_count, 1)
    entity_count        = 0   # NER not run live; use 0 (mean imputation approx)
    topic_idx           = TOPIC_LABELS.index(topic_label) if topic_label in TOPIC_LABELS else 0
    topic_confidence    = 0.85  # reasonable default for manual topic selection
    dominant_topic      = topic_idx

    try:
        sentiment_enc = int(le_sentiment.transform([sentiment_label])[0])
    except Exception:
        sentiment_enc = 1   # fallback: Neutral

    try:
        topic_enc = int(le_topic.transform([topic_label])[0])
    except Exception:
        topic_enc = 0

    row = {
        "word_count":          word_count,
        "sentence_count":      sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "paragraph_count":     paragraph_count,
        "question_count":      question_count,
        "exclaim_count":       exclaim_count,
        "quote_count":         quote_count,
        "dialogue_ratio":      dialogue_ratio,
        "first_person_ratio":  first_person_ratio,
        "uppercase_ratio":     uppercase_ratio,
        "vs_compound":         vs["compound"],
        "vs_pos":              vs["pos"],
        "vs_neg":              vs["neg"],
        "vs_neu":              vs["neu"],
        "sentiment_enc":       sentiment_enc,
        "entity_count":        entity_count,
        "dominant_topic":      dominant_topic,
        "topic_enc":           topic_enc,
        "topic_confidence":    topic_confidence,
        "hour":                hour,
        "dayofweek":           dayofweek,
        "month":               month,
    }
    return pd.DataFrame([row])[FEAT_NAMES]


def tier_badge_html(tier: str) -> str:
    bg = {"Low": "#E63946", "Mid": "#457B9D", "High": "#1D3557", "Viral": "#A8DADC"}
    fg = {"Low": "white",   "Mid": "white",   "High": "white",   "Viral": "#1D3557"}
    return (f'<span class="tier-badge" '
            f'style="background:{bg.get(tier,"#ccc")};color:{fg.get(tier,"#000")}">'
            f'{tier}</span>')


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/"
             "Instagram_icon.png/240px-Instagram_icon.png", width=40)
    st.title("HONY Analytics")
    st.caption("Humans of New York · Engagement ML")
    st.divider()

    st.markdown("**Models loaded**")
    st.success(f"🏷 Classifier: {cls_bundle['model_name']}")
    st.success(f"📈 Regressor: {reg_bundle['model_name']}")
    st.divider()

    st.markdown("**Dataset**")
    st.info(f"📂 {len(df_preds):,} posts  |  2015–2024")
    overall_acc = df_preds["correct_cls"].mean()
    st.metric("Overall accuracy", f"{overall_acc:.1%}")
    st.divider()

    st.caption("Phase 3 of 3 · Built with scikit-learn + XGBoost")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🔮 Live Predictor", "📊 Analytics", "🔍 Post Explorer", "📋 Model Report"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("🔮 Live Post Predictor")
    st.markdown("Paste or type a HONY-style post caption, configure metadata, "
                "and get an instant engagement prediction.")

    col_input, col_result = st.columns([3, 2], gap="large")

    with col_input:
        post_text = st.text_area(
            "Post caption",
            height=220,
            placeholder='e.g. "She told me she\'d been working at the same diner for '
                        '40 years. \'What keeps you here?\' I asked. She smiled. '
                        '\'The people,\' she said. \'Always the people.\'"',
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            hour = st.selectbox(
                "Hour posted (24h)",
                options=list(range(24)),
                index=21,
                format_func=lambda h: f"{h:02d}:00",
            )
        with c2:
            dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dayofweek = st.selectbox(
                "Day of week",
                options=list(range(7)),
                index=5,
                format_func=lambda d: dow_names[d],
            )
        with c3:
            month = st.selectbox(
                "Month",
                options=list(range(1, 13)),
                index=8,
                format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                       "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
            )

        c4, c5 = st.columns(2)
        with c4:
            sentiment_label = st.selectbox(
                "Tone (override VADER?)",
                options=["Auto (VADER)", "Positive", "Neutral", "Negative"],
                index=0,
            )
        with c5:
            topic_label = st.selectbox(
                "Topic", options=TOPIC_LABELS, index=0,
            )

        predict_btn = st.button("✨ Predict Engagement", type="primary",
                                use_container_width=True)

    with col_result:
        if predict_btn:
            if not post_text.strip():
                st.warning("Please enter a post caption first.")
            else:
                # Resolve sentiment
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                _vs = SentimentIntensityAnalyzer().polarity_scores(post_text)
                if sentiment_label == "Auto (VADER)":
                    if _vs["compound"] >= 0.05:
                        sent_lbl = "Positive"
                    elif _vs["compound"] <= -0.05:
                        sent_lbl = "Negative"
                    else:
                        sent_lbl = "Neutral"
                else:
                    sent_lbl = sentiment_label

                X_live = build_feature_vector(
                    post_text, hour, dayofweek, month, sent_lbl, topic_label
                )

                pred_tier_idx  = cls_model.predict(X_live)[0]
                pred_tier      = TIER_ORDER[pred_tier_idx]
                pred_probs     = cls_model.predict_proba(X_live)[0]
                pred_log       = reg_model.predict(X_live)[0]
                pred_notes_est = int(np.expm1(pred_log))

                st.subheader("Prediction Result")
                st.markdown(f"**Predicted tier:** {tier_badge_html(pred_tier)}",
                            unsafe_allow_html=True)

                st.metric("Estimated engagement (notes)", f"~{pred_notes_est:,}",
                          help="Back-transformed from log(notes). Treat as a signal, "
                               "not an exact forecast.")

                # Probability gauge
                prob_df = pd.DataFrame({
                    "Tier": TIER_ORDER,
                    "Probability": pred_probs,
                    "Color": [TIER_COLORS[t] for t in TIER_ORDER],
                })
                fig = px.bar(
                    prob_df, x="Probability", y="Tier",
                    orientation="h", color="Tier",
                    color_discrete_map=TIER_COLORS,
                    range_x=[0, 1],
                    labels={"Probability": "P(tier)"},
                    height=200,
                )
                fig.update_layout(
                    showlegend=False, margin=dict(l=0, r=10, t=10, b=10),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(categoryorder="array", categoryarray=TIER_ORDER),
                )
                fig.update_traces(texttemplate="%{x:.2f}", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

                # Text-derived feature summary
                wc = len(post_text.split())
                st.markdown("**Auto-detected features:**")
                feat_cols = st.columns(3)
                feat_cols[0].metric("Word count", wc)
                feat_cols[1].metric("VADER compound", f"{_vs['compound']:.2f}")
                feat_cols[2].metric("Tone", sent_lbl)

                st.markdown(
                    '<div class="disclaimer">⚠️ Regression predicts log(note_count). '
                    'Raw note estimates are approximate — viral outliers make exact '
                    'counts unpredictable. Use tier prediction as the primary signal.'
                    '</div>', unsafe_allow_html=True
                )
        else:
            st.info("👈 Enter a post and click **Predict Engagement** to see results.")
            st.markdown("""
**Tips for high engagement (from EDA):**
- ✂️ Keep it short — word count negatively correlates with notes
- 🌙 Post at night / on weekends
- 💬 Dialogue-heavy posts outperform narration
- 📖 Topics: War & Conflict and Health & Struggle drive the highest median notes
- 😐 Neutral/bittersweet tone slightly outperforms overtly positive
            """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("📊 Portfolio Analytics")

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total posts", f"{len(df_preds):,}")
    k2.metric("Median notes", f"{df_preds['note_count'].median():,.0f}")
    k3.metric("Classifier accuracy", f"{df_preds['correct_cls'].mean():.1%}")
    k4.metric("Viral posts", f"{(df_preds['tier']=='Viral').sum()}")

    st.divider()

    # ── Row 1: Tier distribution + accuracy by tier ───────────────────────────
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.subheader("Tier Distribution")
        tier_cnt = df_preds["tier"].value_counts().reindex(TIER_ORDER).reset_index()
        tier_cnt.columns = ["tier", "count"]
        fig = px.bar(tier_cnt, x="tier", y="count", color="tier",
                     color_discrete_map=TIER_COLORS,
                     labels={"tier": "Tier", "count": "Posts"},
                     text="count")
        fig.update_layout(showlegend=False, plot_bgcolor="white",
                          paper_bgcolor="white", height=320)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.subheader("Prediction Accuracy by Tier")
        acc_tier = (
            df_preds.groupby("tier", observed=True)["correct_cls"]
            .mean()
            .reindex(TIER_ORDER)
            .reset_index()
        )
        acc_tier.columns = ["tier", "accuracy"]
        fig = px.bar(acc_tier, x="tier", y="accuracy", color="tier",
                     color_discrete_map=TIER_COLORS,
                     labels={"tier": "Tier", "accuracy": "Accuracy"},
                     range_y=[0, 1], text_auto=".1%")
        fig.update_layout(showlegend=False, plot_bgcolor="white",
                          paper_bgcolor="white", height=320)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Engagement over time + predicted vs actual scatter ─────────────
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.subheader("Median Notes Over Time")
        yearly = (
            df_preds.groupby("year")["note_count"]
            .median()
            .reset_index()
        )
        fig = px.line(yearly, x="year", y="note_count", markers=True,
                      labels={"year": "Year", "note_count": "Median Notes"},
                      color_discrete_sequence=["#457B9D"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=320)
        fig.add_annotation(
            x=yearly["year"].max() * 0.7, y=yearly["note_count"].max(),
            text="⚠️ Temporal decay: older posts had structurally higher engagement",
            showarrow=False, font=dict(size=9, color="#888"),
            bgcolor="rgba(255,248,225,0.9)", bordercolor="#F4A261",
        )
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        st.subheader("Predicted vs Actual log(notes)")
        fig = px.scatter(
            df_preds, x="log_notes", y="pred_log_notes",
            color="tier", color_discrete_map=TIER_COLORS,
            opacity=0.45, labels={"log_notes": "Actual", "pred_log_notes": "Predicted"},
            height=320,
        )
        lo = df_preds["log_notes"].min()
        hi = df_preds["log_notes"].max()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(color="#E63946", dash="dash", width=1.5),
                                 name="Perfect fit", showlegend=True))
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Probability distributions ─────────────────────────────────────
    st.subheader("Prediction Confidence Distribution")
    prob_cols = [f"prob_{t}" for t in TIER_ORDER]
    fig = make_subplots(rows=1, cols=4,
                        subplot_titles=[f"P({t})" for t in TIER_ORDER])
    for i, (t, col) in enumerate(zip(TIER_ORDER, prob_cols), 1):
        fig.add_trace(
            go.Histogram(x=df_preds[col], nbinsx=30,
                         marker_color=TIER_COLORS[t], opacity=0.75,
                         showlegend=False),
            row=1, col=i,
        )
    fig.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(t=40))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — POST EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("🔍 Post Explorer")
    st.markdown("Filter and explore individual post predictions.")

    # Filters
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        tier_filter = st.multiselect(
            "True tier", options=TIER_ORDER, default=TIER_ORDER
        )
    with fc2:
        correct_filter = st.selectbox(
            "Prediction", options=["All", "Correct only", "Incorrect only"]
        )
    with fc3:
        year_min, year_max = int(df_preds["year"].min()), int(df_preds["year"].max())
        year_range = st.slider("Year range", year_min, year_max,
                               (year_min, year_max))
    with fc4:
        sort_by = st.selectbox(
            "Sort by", ["note_count (desc)", "note_count (asc)",
                        "pred_note_count_approx (desc)"]
        )

    filtered = df_preds[df_preds["tier"].isin(tier_filter)]
    filtered = filtered[
        (filtered["year"] >= year_range[0]) &
        (filtered["year"] <= year_range[1])
    ]
    if correct_filter == "Correct only":
        filtered = filtered[filtered["correct_cls"]]
    elif correct_filter == "Incorrect only":
        filtered = filtered[~filtered["correct_cls"]]

    sort_col = sort_by.split(" ")[0]
    ascending = "asc" in sort_by
    filtered = filtered.sort_values(sort_col, ascending=ascending)

    st.markdown(f"**{len(filtered):,} posts** match your filters.")

    # Display table
    display_df = filtered[[
        "post_id", "dt", "note_count", "tier", "pred_tier",
        "pred_note_count_approx", "prob_Low", "prob_Mid", "prob_High", "prob_Viral",
        "correct_cls",
    ]].copy()
    display_df["dt"] = display_df["dt"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["pred_note_count_approx"] = display_df["pred_note_count_approx"].round(0).astype(int)
    display_df.columns = [
        "Post ID", "Date", "Actual Notes", "True Tier", "Pred Tier",
        "Est. Notes", "P(Low)", "P(Mid)", "P(High)", "P(Viral)", "Correct?",
    ]

    st.dataframe(
        display_df.head(200),
        use_container_width=True,
        height=420,
        column_config={
            "P(Low)":  st.column_config.ProgressColumn("P(Low)",  min_value=0, max_value=1, format="%.2f"),
            "P(Mid)":  st.column_config.ProgressColumn("P(Mid)",  min_value=0, max_value=1, format="%.2f"),
            "P(High)": st.column_config.ProgressColumn("P(High)", min_value=0, max_value=1, format="%.2f"),
            "P(Viral)":st.column_config.ProgressColumn("P(Viral)",min_value=0, max_value=1, format="%.2f"),
            "Correct?":st.column_config.CheckboxColumn("Correct?"),
        },
    )

    # Misclassification breakdown
    st.divider()
    st.subheader("Misclassification Breakdown")
    mc_cols = st.columns(2)

    with mc_cols[0]:
        st.markdown("**Confusion heatmap (full dataset)**")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(
            df_preds["tier"].map({t: i for i, t in enumerate(TIER_ORDER)}),
            df_preds["pred_tier"].map({t: i for i, t in enumerate(TIER_ORDER)}),
        )
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig_cm = px.imshow(
            cm_norm, x=TIER_ORDER, y=TIER_ORDER,
            color_continuous_scale="Blues", zmin=0, zmax=1,
            labels=dict(x="Predicted", y="True", color="Proportion"),
            text_auto=".2f", height=320,
        )
        fig_cm.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_cm, use_container_width=True)

    with mc_cols[1]:
        st.markdown("**Most common misclassification paths**")
        wrong = df_preds[~df_preds["correct_cls"]].copy()
        wrong["path"] = wrong["tier"] + " → " + wrong["pred_tier"]
        path_counts = wrong["path"].value_counts().head(8).reset_index()
        path_counts.columns = ["Misclassification", "Count"]
        fig_p = px.bar(path_counts, x="Count", y="Misclassification",
                       orientation="h", color="Count",
                       color_continuous_scale="Reds", height=320)
        fig_p.update_layout(showlegend=False, plot_bgcolor="white",
                            paper_bgcolor="white",
                            yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_p, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("📋 Model Performance Report")

    # ── Classification summary ────────────────────────────────────────────────
    cls_summary = {
        "Logistic Regression": {"cv_f1": 0.395, "test_f1": 0.414},
        "Random Forest":        {"cv_f1": 0.392, "test_f1": 0.433},
        "XGBoost":              {"cv_f1": 0.390, "test_f1": 0.471},
    }
    reg_summary = {
        "Ridge":         {"rmse": 0.778, "r2": 0.384},
        "Random Forest": {"rmse": 0.748, "r2": 0.430},
        "XGBoost":       {"rmse": 0.759, "r2": 0.413},
    }

    st.subheader("Classification — Tier Prediction")
    rep_c1, rep_c2 = st.columns(2)

    with rep_c1:
        cls_df = pd.DataFrame(cls_summary).T.reset_index()
        cls_df.columns = ["Model", "CV F1-Macro", "Test F1-Macro"]
        fig = px.bar(
            cls_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Model", y="Score", color="Metric", barmode="group",
            color_discrete_sequence=["#457B9D", "#1D3557"],
            range_y=[0, 0.7], text_auto=".3f", height=320,
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with rep_c2:
        st.markdown("**Per-class F1 (XGBoost — best)**")
        per_cls = {"Low": 0.735, "Mid": 0.502, "High": 0.387, "Viral": 0.259}
        fig = px.bar(
            x=list(per_cls.keys()), y=list(per_cls.values()),
            color=list(per_cls.keys()), color_discrete_map=TIER_COLORS,
            labels={"x": "Tier", "y": "F1"}, range_y=[0, 1],
            text_auto=".2f", height=320,
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
> **Interpretation:** The classifier is most confident on **Low** tier (F1=0.73), 
> reflecting the clear linguistic signature of low-engagement posts (longer, fewer quotes, less dialogue). 
> **Viral** prediction is hardest (F1=0.26) — virality is partially driven by timing, 
> platform context, and social amplification that text features alone can't capture.
    """)

    st.divider()
    st.subheader("Regression — log(note_count) Prediction")
    reg_c1, reg_c2 = st.columns(2)

    with reg_c1:
        reg_df = pd.DataFrame(reg_summary).T.reset_index()
        reg_df.columns = ["Model", "RMSE", "R²"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["RMSE (lower=better)", "R² (higher=better)"])
        colors_r = ["#E63946", "#457B9D", "#1D3557"]
        for i, (m, c) in enumerate(zip(reg_df["Model"], colors_r)):
            fig.add_trace(go.Bar(name=m, x=[m], y=[reg_df.loc[i, "RMSE"]],
                                  marker_color=c, showlegend=False), row=1, col=1)
            fig.add_trace(go.Bar(name=m, x=[m], y=[reg_df.loc[i, "R²"]],
                                  marker_color=c, showlegend=i == 0), row=1, col=2)
        fig.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
                          barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    with reg_c2:
        st.markdown("**R² = 0.43 means what?**")
        st.markdown("""
The Random Forest regressor explains **43% of variance** in log(note_count).  
The remaining ~57% is attributable to:
- Platform-level timing & trending effects
- Photo quality (not available as a feature)
- Share/repost cascades (network effects)
- Residual temporal confound even after excluding `year`

This is **reasonable for social media prediction** — text and metadata alone 
rarely exceed R²~0.5 on engagement targets.
        """)
        st.markdown(
            '<div class="disclaimer">⚠️ Regression targets <code>log(note_count)</code> '
            'for stability. Raw note_count regression is unreliable due to extreme viral '
            'outliers. Use <code>exp(pred) − 1</code> as an approximate back-transform only.'
            '</div>', unsafe_allow_html=True
        )

    # ── Raw summary file ──────────────────────────────────────────────────────
    st.divider()
    summary_path = BASE / "model_performance_summary.txt"
    if summary_path.exists():
        with st.expander("📄 Raw performance summary (model_performance_summary.txt)"):
            st.code(summary_path.read_text(), language="text")

    # ── Feature importance text notes ─────────────────────────────────────────
    st.divider()
    st.subheader("Key Feature Insights")
    fi_c1, fi_c2 = st.columns(2)

    with fi_c1:
        st.markdown("""
**Top drivers (both models):**
1. 🔴 `word_count` — #1 in regression; longer posts consistently underperform
2. 🔴 `dialogue_ratio` — quoted speech / dialogue lifts engagement
3. 🟢 `dominant_topic` — topic matters; War & Conflict / Health & Struggle lead
4. 🔴 `quote_count` — direct quotes signal richer storytelling
5. 🔴 `question_count` — rhetorical questions engage readers
        """)
    with fi_c2:
        st.markdown("""
**Weaker signals:**
- `paragraph_count` — near zero importance in both models
- `exclaim_count`, `topic_enc` — minimal predictive power
- Sentiment features (vs_pos/neg/neu) — moderate, not dominant
- Temporal (hour/month/dayofweek) — mild signal after removing `year`

**Implication for content strategy:** 
Focus on *structure* (shorter, dialogue-driven) over *tone* (positive vs negative).
        """)