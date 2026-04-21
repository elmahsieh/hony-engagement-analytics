"""
HONY (Humans of New York) — Phase 2: ML Classification + Regression
=====================================================================
Builds on hony_ml_ready.csv output from Phase 1 (hony_eda_nlp.py).

Goals:
  1. Preprocessing pipeline — encode categoricals, scale numerics,
     exclude 'year' as temporal confound
  2. Classification — predict tier (Low/Mid/High/Viral)
     Models: Logistic Regression (baseline), Random Forest, XGBoost
     Eval: F1-macro, confusion matrix, per-class breakdown
  3. Regression — predict log_notes
     Models: Ridge (baseline), Random Forest, XGBoost
     Eval: RMSE, R²  ⚠️  see disclaimer at bottom
  4. Feature importance — unified chart across both best models
  5. Export: .pkl models, predictions CSV, performance summary TXT

Run:
  pip install scikit-learn xgboost pandas matplotlib seaborn
  python hony_ml_phase2.py

Input:  outputs/hony_ml_ready.csv   (from Phase 1)
Output: outputs/  (11_–15_ charts, models, predictions, summary)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

# ── palette (mirrors Phase 1 style) ──────────────────────────────────────────
PALETTE     = ["#E63946", "#457B9D", "#1D3557", "#A8DADC", "#F4A261", "#2A9D8F"]
TIER_COLORS = {"Low": "#E63946", "Mid": "#457B9D", "High": "#1D3557", "Viral": "#A8DADC"}
TIER_ORDER  = ["Low", "Mid", "High", "Viral"]
plt.rcParams.update({"figure.dpi": 140, "font.family": "sans-serif"})

# ─────────────────────────────────────────────────────────────────────────────
# 0. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("outputs/hony_ml_ready.csv")
print(f"Loaded {len(df):,} posts | columns: {df.shape[1]}")
print("\nTier distribution:")
print(df["tier"].value_counts().reindex(TIER_ORDER))

# ─────────────────────────────────────────────────────────────────────────────
# 1. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

# Encode categoricals
le_sentiment = LabelEncoder()
df["sentiment_enc"] = le_sentiment.fit_transform(df["sentiment_label"])

le_topic = LabelEncoder()
df["topic_enc"] = le_topic.fit_transform(df["topic_label"])

# Encode tier as ordered integer
tier_map = {t: i for i, t in enumerate(TIER_ORDER)}
df["tier_enc"] = df["tier"].map(tier_map)

# Feature groups
TEXT_FEATS      = [
    "word_count", "sentence_count", "avg_sentence_length", "paragraph_count",
    "question_count", "exclaim_count", "quote_count",
    "dialogue_ratio", "first_person_ratio", "uppercase_ratio",
]
SENTIMENT_FEATS = ["vs_compound", "vs_pos", "vs_neg", "vs_neu", "sentiment_enc"]
NLP_FEATS       = ["entity_count", "dominant_topic", "topic_enc", "topic_confidence"]
TEMPORAL_FEATS  = ["hour", "dayofweek", "month"]
# NOTE: 'year' deliberately excluded — platform engagement decayed 2016→2022,
#       making year a structural confound rather than a signal.

ALL_FEATS = TEXT_FEATS + SENTIMENT_FEATS + NLP_FEATS + TEMPORAL_FEATS

X      = df[ALL_FEATS].copy()
y_cls  = df["tier_enc"].values
y_reg  = df["log_notes"].values

# Year-detrended regression target (secondary — removes platform decay signal)
yr_detrend = LinearRegression().fit(df[["year"]], y_reg)
y_reg_detrended = y_reg - yr_detrend.predict(df[["year"]])

# ── Train / test split (stratified by tier) ───────────────────────────────────
X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls,
)
_, _, y_det_train, y_det_test = train_test_split(
    X, y_reg_detrended, test_size=0.2, random_state=42, stratify=y_cls,
)

# Preprocessing transformer (scales all numeric features)
preprocessor = ColumnTransformer(
    [("scale", StandardScaler(), ALL_FEATS)],
    remainder="drop",
)

print(f"\nFeature set ({len(ALL_FEATS)} features): {', '.join(ALL_FEATS)}")
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CLASSIFICATION — predict tier
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== CLASSIFICATION ===")

cls_models = {
    "Logistic Regression": Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42,
                                   class_weight="balanced", C=1.0)),
    ]),
    "Random Forest": Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42,
                                        class_weight="balanced", n_jobs=-1)),
    ]),
    "XGBoost": Pipeline([
        ("pre", preprocessor),
        ("clf", XGBClassifier(n_estimators=300, learning_rate=0.05,
                               max_depth=6, subsample=0.8, colsample_bytree=0.8,
                               eval_metric="mlogloss", random_state=42, n_jobs=-1)),
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cls_results = {}

for name, pipe in cls_models.items():
    cv_scores = cross_val_score(pipe, X_train, y_cls_train, cv=cv,
                                 scoring="f1_macro", n_jobs=-1)
    pipe.fit(X_train, y_cls_train)
    y_pred   = pipe.predict(X_test)
    test_f1  = f1_score(y_cls_test, y_pred, average="macro")
    cls_results[name] = {
        "pipe":        pipe,
        "cv_mean":     cv_scores.mean(),
        "cv_std":      cv_scores.std(),
        "test_f1":     test_f1,
        "y_pred":      y_pred,
        "report":      classification_report(y_cls_test, y_pred,
                                              target_names=TIER_ORDER,
                                              output_dict=True),
    }
    print(f"  {name:22s}: CV F1={cv_scores.mean():.3f}±{cv_scores.std():.3f} "
          f"| Test F1={test_f1:.3f}")

best_cls = max(cls_results, key=lambda k: cls_results[k]["test_f1"])
print(f"  → Best classifier: {best_cls} (F1={cls_results[best_cls]['test_f1']:.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# 3. REGRESSION — predict log_notes
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== REGRESSION (log_notes) ===")

reg_models = {
    "Ridge": Pipeline([
        ("pre", preprocessor),
        ("reg", Ridge(alpha=1.0)),
    ]),
    "Random Forest": Pipeline([
        ("pre", preprocessor),
        ("reg", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ]),
    "XGBoost": Pipeline([
        ("pre", preprocessor),
        ("reg", XGBRegressor(n_estimators=300, learning_rate=0.05,
                              max_depth=6, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, n_jobs=-1)),
    ]),
}

reg_results = {}

for name, pipe in reg_models.items():
    pipe.fit(X_train, y_reg_train)
    y_pred = pipe.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    r2     = r2_score(y_reg_test, y_pred)
    reg_results[name] = {
        "pipe":   pipe,
        "rmse":   rmse,
        "r2":     r2,
        "y_pred": y_pred,
    }
    print(f"  {name:22s}: RMSE={rmse:.3f} | R²={r2:.3f}")

best_reg = max(reg_results, key=lambda k: reg_results[k]["r2"])
print(f"  → Best regressor: {best_reg} (R²={reg_results[best_reg]['r2']:.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

def despine(ax):
    sns.despine(ax=ax)

# ── 11. Classification model comparison ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

model_names = list(cls_results.keys())
colors = PALETTE[:3]

# CV F1
ax = axes[0]
cv_means = [cls_results[m]["cv_mean"] for m in model_names]
cv_stds  = [cls_results[m]["cv_std"]  for m in model_names]
bars = ax.bar(model_names, cv_means, yerr=cv_stds, color=colors,
              edgecolor="white", capsize=5, error_kw={"color": "#333"})
for bar, val in zip(bars, cv_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0, 1)
ax.set_title("5-Fold CV F1-Macro", fontweight="bold")
ax.set_ylabel("F1-Macro")
ax.tick_params(axis="x", rotation=12)
despine(ax)

# Test F1
ax = axes[1]
test_f1s = [cls_results[m]["test_f1"] for m in model_names]
bars = ax.bar(model_names, test_f1s, color=colors, edgecolor="white")
for bar, val in zip(bars, test_f1s):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0, 1)
ax.set_title("Test Set F1-Macro", fontweight="bold")
ax.set_ylabel("F1-Macro")
ax.tick_params(axis="x", rotation=12)
despine(ax)

# Per-class F1 for best model
ax = axes[2]
report = cls_results[best_cls]["report"]
per_cls_f1 = [report[t]["f1-score"] for t in TIER_ORDER]
tier_bar_colors = [TIER_COLORS[t] for t in TIER_ORDER]
bars = ax.bar(TIER_ORDER, per_cls_f1, color=tier_bar_colors, edgecolor="white")
for bar, val in zip(bars, per_cls_f1):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
            f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0, 1)
ax.set_title(f"{best_cls}\nPer-Class F1", fontweight="bold")
ax.set_ylabel("F1 Score")
despine(ax)

plt.suptitle("Classification Model Comparison", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/11_classification_comparison.png", bbox_inches="tight")
plt.close()
print("  11_classification_comparison.png")

# ── 12. Confusion matrix (best classifier, normalised) ───────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
cm      = confusion_matrix(y_cls_test, cls_results[best_cls]["y_pred"])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

sns.heatmap(
    cm_norm, annot=False, cmap="Blues",
    xticklabels=TIER_ORDER, yticklabels=TIER_ORDER,
    linewidths=0.5, linecolor="#ddd", ax=ax,
    vmin=0, vmax=1,
)
# Annotate with proportion + raw count
for i in range(len(TIER_ORDER)):
    for j in range(len(TIER_ORDER)):
        ax.text(j + 0.5, i + 0.42, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white" if cm_norm[i, j] > 0.55 else "#222")
        ax.text(j + 0.5, i + 0.65, f"(n={cm[i, j]})",
                ha="center", va="center", fontsize=8,
                color="white" if cm_norm[i, j] > 0.55 else "#555")

ax.set_title(f"Confusion Matrix — {best_cls} (Normalised)", fontweight="bold")
ax.set_xlabel("Predicted Tier")
ax.set_ylabel("True Tier")
plt.tight_layout()
plt.savefig("outputs/12_confusion_matrix.png", bbox_inches="tight")
plt.close()
print("  12_confusion_matrix.png")

# ── 13. Regression model comparison ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

reg_names = list(reg_results.keys())

# RMSE
ax = axes[0]
rmses = [reg_results[m]["rmse"] for m in reg_names]
bars  = ax.bar(reg_names, rmses, color=PALETTE[:3], edgecolor="white")
for bar, val in zip(bars, rmses):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("RMSE (lower = better)", fontweight="bold")
ax.set_ylabel("RMSE")
ax.tick_params(axis="x", rotation=12)
despine(ax)

# R²
ax = axes[1]
r2s  = [reg_results[m]["r2"] for m in reg_names]
bars = ax.bar(reg_names, r2s, color=PALETTE[:3], edgecolor="white")
for bar, val in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width() / 2, max(val, 0) + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("R² Score (higher = better)", fontweight="bold")
ax.set_ylabel("R²")
ax.tick_params(axis="x", rotation=12)
despine(ax)

# Predicted vs actual (best regressor)
ax = axes[2]
y_pred_best = reg_results[best_reg]["y_pred"]
ax.scatter(y_reg_test, y_pred_best, alpha=0.3, color=PALETTE[1], s=12, edgecolors="none")
lo = min(y_reg_test.min(), y_pred_best.min())
hi = max(y_reg_test.max(), y_pred_best.max())
ax.plot([lo, hi], [lo, hi], color=PALETTE[0], lw=1.5, ls="--", label="Perfect fit")
ax.set_title(f"{best_reg}\nPredicted vs Actual log(notes)", fontweight="bold")
ax.set_xlabel("Actual log(notes)")
ax.set_ylabel("Predicted log(notes)")
ax.legend()
despine(ax)

fig.text(
    0.5, -0.04,
    "⚠️  Models predict log(note_count) for stability. "
    "Raw note_count regression is unreliable due to extreme viral outliers. "
    "Back-transform via exp(pred) − 1 for approximate counts only.",
    ha="center", fontsize=8, style="italic", color="#555",
)

plt.suptitle("Regression Model Comparison — Predicting log(note_count)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/13_regression_comparison.png", bbox_inches="tight")
plt.close()
print("  13_regression_comparison.png")

# ── 14. Unified feature importance ───────────────────────────────────────────
def get_importance(pipe, model_key):
    model = pipe.named_steps[model_key]
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "coef_"):
        c = model.coef_
        return np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c.ravel())
    return None

cls_imp = get_importance(cls_results[best_cls]["pipe"], "clf")
reg_imp = get_importance(reg_results[best_reg]["pipe"], "reg")

# Color by feature group
GROUP_COLORS = {
    "Text":      PALETTE[0],
    "Sentiment": PALETTE[1],
    "NLP/Topic": PALETTE[5],
    "Temporal":  PALETTE[4],
}
def feat_group_color(f):
    if f in TEXT_FEATS:       return GROUP_COLORS["Text"]
    if f in SENTIMENT_FEATS:  return GROUP_COLORS["Sentiment"]
    if f in NLP_FEATS:        return GROUP_COLORS["NLP/Topic"]
    return GROUP_COLORS["Temporal"]

legend_patches = [mpatches.Patch(color=v, label=k) for k, v in GROUP_COLORS.items()]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, imp, title in zip(
    axes,
    [cls_imp, reg_imp],
    [f"Classification Importance\n({best_cls})",
     f"Regression Importance\n({best_reg})"],
):
    if imp is None:
        ax.text(0.5, 0.5, "Importance N/A", ha="center", va="center",
                transform=ax.transAxes)
        continue
    df_imp = pd.DataFrame({"feature": ALL_FEATS, "importance": imp})
    df_imp = df_imp.sort_values("importance", ascending=True)
    colors = [feat_group_color(f) for f in df_imp["feature"]]
    ax.barh(df_imp["feature"], df_imp["importance"],
            color=colors, edgecolor="white", height=0.7)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    despine(ax)

plt.suptitle("Unified Feature Importance — What Drives HONY Engagement?",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/14_feature_importance.png", bbox_inches="tight")
plt.close()
print("  14_feature_importance.png")

# ── 15. Prediction residual distribution (regression) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

residuals = y_reg_test - reg_results[best_reg]["y_pred"]

ax = axes[0]
ax.hist(residuals, bins=40, color=PALETTE[1], edgecolor="white", alpha=0.85)
ax.axvline(0, color=PALETTE[0], linestyle="--", linewidth=1.5, label="Zero error")
ax.set_title(f"Residual Distribution\n({best_reg})", fontweight="bold")
ax.set_xlabel("Residual (actual − predicted)")
ax.set_ylabel("Count")
ax.legend()
despine(ax)

ax = axes[1]
# Residual by true tier
test_tiers = [TIER_ORDER[y] for y in y_cls_test]
resid_df   = pd.DataFrame({"tier": test_tiers, "residual": residuals})
resid_df["tier"] = pd.Categorical(resid_df["tier"], categories=TIER_ORDER, ordered=True)
resid_df.groupby("tier", observed=True)["residual"].plot(
    kind="kde", ax=ax, legend=True,
    color=[TIER_COLORS[t] for t in TIER_ORDER]
)
ax.axvline(0, color="#888", linestyle="--", linewidth=1)
ax.set_title("Residual Distribution by Tier", fontweight="bold")
ax.set_xlabel("Residual")
ax.set_ylabel("Density")
despine(ax)

plt.suptitle("Regression Residual Analysis", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/15_regression_residuals.png", bbox_inches="tight")
plt.close()
print("  15_regression_residuals.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. EXPORT TRAINED MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\nExporting models...")

with open("outputs/hony_classifier.pkl", "wb") as f:
    pickle.dump({
        "model":                  cls_results[best_cls]["pipe"],
        "model_name":             best_cls,
        "feature_names":          ALL_FEATS,
        "tier_order":             TIER_ORDER,
        "tier_map":               tier_map,
        "le_sentiment":           le_sentiment,
        "le_topic":               le_topic,
    }, f)

with open("outputs/hony_regressor.pkl", "wb") as f:
    pickle.dump({
        "model":                  reg_results[best_reg]["pipe"],
        "model_name":             best_reg,
        "feature_names":          ALL_FEATS,
        "year_detrend_model":     yr_detrend,   # for optional detrended inference
        "le_sentiment":           le_sentiment,
        "le_topic":               le_topic,
    }, f)

print("  hony_classifier.pkl")
print("  hony_regressor.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PREDICTIONS CSV (full dataset)
# ─────────────────────────────────────────────────────────────────────────────
print("\nExporting predictions...")

X_all       = df[ALL_FEATS].copy()
pred_cls    = cls_results[best_cls]["pipe"].predict(X_all)
pred_prob   = cls_results[best_cls]["pipe"].predict_proba(X_all)
pred_reg    = reg_results[best_reg]["pipe"].predict(X_all)

out_df = df[["post_id", "dt", "note_count", "tier", "log_notes"]].copy()
out_df["pred_tier"]              = [TIER_ORDER[p] for p in pred_cls]
out_df["pred_log_notes"]         = pred_reg
out_df["pred_note_count_approx"] = np.expm1(pred_reg)   # back-transform; approximate
for i, t in enumerate(TIER_ORDER):
    out_df[f"prob_{t}"] = pred_prob[:, i]
out_df["correct_cls"] = out_df["tier"] == out_df["pred_tier"]

out_df.to_csv("outputs/hony_predictions.csv", index=False)
print("  hony_predictions.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PERFORMANCE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
lines = []
sep   = "=" * 62

lines += [sep, "HONY Phase 2 — Model Performance Summary", sep, ""]

lines += ["CLASSIFICATION  (target: tier — Low / Mid / High / Viral)",
          "-" * 50]
for name, res in cls_results.items():
    tag = "  ← BEST" if name == best_cls else ""
    lines.append(f"  {name}{tag}")
    lines.append(f"    CV  F1-Macro : {res['cv_mean']:.4f} ± {res['cv_std']:.4f}")
    lines.append(f"    Test F1-Macro: {res['test_f1']:.4f}")
    r = res["report"]
    for t in TIER_ORDER:
        lines.append(
            f"      {t:5s}: P={r[t]['precision']:.3f}  "
            f"R={r[t]['recall']:.3f}  F1={r[t]['f1-score']:.3f}  "
            f"n={int(r[t]['support'])}"
        )
    lines.append("")

lines += ["REGRESSION  (target: log_notes)", "-" * 50]
for name, res in reg_results.items():
    tag = "  ← BEST" if name == best_reg else ""
    lines.append(f"  {name}{tag}")
    lines.append(f"    RMSE : {res['rmse']:.4f}")
    lines.append(f"    R²   : {res['r2']:.4f}")
    lines.append("")

lines += [
    "FEATURE SET", "-" * 50,
    f"  {len(ALL_FEATS)} features used: {', '.join(ALL_FEATS)}",
    "  'year' excluded — temporal confound (platform decay 2016→2022).",
    "",
    "DISCLAIMER", "-" * 50,
    "  Regression targets log(note_count) for numerical stability.",
    "  Raw note_count is right-skewed with extreme viral outliers;",
    "  direct regression on it is unreliable. Back-transform predictions",
    "  via exp(pred_log_notes) − 1 for approximate raw counts only.",
    "  Treat regression outputs as engagement signals, not exact counts.",
    "",
    "OUTPUTS", "-" * 50,
    "  11_classification_comparison.png",
    "  12_confusion_matrix.png",
    "  13_regression_comparison.png",
    "  14_feature_importance.png",
    "  15_regression_residuals.png",
    "  hony_classifier.pkl",
    "  hony_regressor.pkl",
    "  hony_predictions.csv",
    "  model_performance_summary.txt",
    sep,
]

summary = "\n".join(lines)
print("\n" + summary)

with open("outputs/model_performance_summary.txt", "w") as f:
    f.write(summary)
print("\n✓ model_performance_summary.txt")
print("\nPhase 2 complete. All outputs in ./outputs/")