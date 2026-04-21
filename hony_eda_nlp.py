"""
HONY (Humans of New York) — Full NLP EDA Pipeline
====================================================
Covers:
  1. Sentiment Analysis (VADER + trend over time)
  2. Topic Modeling (LDA via sklearn)
  3. Named Entity Recognition (NLTK NE chunker)
  4. Narrative / Linguistic Feature Analysis
  5. Engagement Tier Labeling
  6. Feature Correlation Matrix (ML-ready)

Run:
  pip install nltk gensim textblob wordcloud plotly scikit-learn matplotlib seaborn
  python -c "import nltk; nltk.download('all')"
  python hony_eda_nlp.py

Outputs: /outputs/ folder with all charts + a summary CSV
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────
PALETTE = ["#E63946", "#457B9D", "#1D3557", "#A8DADC", "#F4A261", "#2A9D8F"]
plt.rcParams.update({"figure.dpi": 140, "font.family": "sans-serif"})

# ─────────────────────────────────────────────────────────────────────────────
# 0. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("hony_posts_enriched.csv")
df["dt"] = pd.to_datetime(df["dt"])
print(f"Loaded {len(df):,} posts | columns: {df.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. ENGAGEMENT TIER LABELING
# ─────────────────────────────────────────────────────────────────────────────
# Quartile-based tiers on note_count (year-adjusted to reduce temporal bias)
df["tier"] = pd.qcut(
    df["note_count"],
    q=[0, 0.25, 0.60, 0.85, 1.0],
    labels=["Low", "Mid", "High", "Viral"],
)
tier_order = ["Low", "Mid", "High", "Viral"]
print("\nEngagement tier distribution:")
print(df["tier"].value_counts().reindex(tier_order))

fig, ax = plt.subplots(figsize=(7, 4))
counts = df["tier"].value_counts().reindex(tier_order)
bars = ax.bar(tier_order, counts.values, color=PALETTE[:4], edgecolor="white", linewidth=0.8)
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15, str(v),
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Engagement Tier Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Post Count")
ax.set_xlabel("Tier (note_count quartiles)")
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("outputs/01_engagement_tiers.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 2. SENTIMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Re-score (VADER already in enriched, but re-run for validation)
scores = df["text"].apply(lambda t: analyzer.polarity_scores(str(t)))
df["vs_compound"] = scores.apply(lambda x: x["compound"])
df["vs_pos"]      = scores.apply(lambda x: x["pos"])
df["vs_neg"]      = scores.apply(lambda x: x["neg"])
df["vs_neu"]      = scores.apply(lambda x: x["neu"])

df["sentiment_label"] = pd.cut(
    df["vs_compound"],
    bins=[-1, -0.05, 0.05, 1],
    labels=["Negative", "Neutral", "Positive"],
)

# 2a. Sentiment label distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sent_counts = df["sentiment_label"].value_counts().reindex(["Negative", "Neutral", "Positive"])
axes[0].bar(sent_counts.index, sent_counts.values,
            color=["#E63946", "#A8DADC", "#2A9D8F"], edgecolor="white")
axes[0].set_title("Sentiment Label Distribution", fontweight="bold")
axes[0].set_ylabel("Post Count")
sns.despine(ax=axes[0])

# Compound score distribution
axes[1].hist(df["vs_compound"], bins=40, color=PALETTE[1], edgecolor="white", alpha=0.85)
axes[1].axvline(df["vs_compound"].median(), color="red", linestyle="--",
                label=f"Median: {df['vs_compound'].median():.2f}")
axes[1].set_title("VADER Compound Score Distribution", fontweight="bold")
axes[1].set_xlabel("Compound Score (-1 to +1)")
axes[1].legend()
sns.despine(ax=axes[1])

plt.suptitle("Sentiment Overview", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/02_sentiment_distribution.png", bbox_inches="tight")
plt.close()

# 2b. Sentiment x Engagement Tier
fig, ax = plt.subplots(figsize=(8, 5))
sent_tier = df.groupby(["tier", "sentiment_label"], observed=True)["note_count"].median().unstack()
sent_tier = sent_tier.reindex(tier_order)
sent_tier.plot(kind="bar", ax=ax, color=["#E63946", "#A8DADC", "#2A9D8F"],
               edgecolor="white", width=0.7)
ax.set_title("Median Note Count: Tier × Sentiment", fontweight="bold")
ax.set_xlabel("Engagement Tier")
ax.set_ylabel("Median Note Count")
ax.legend(title="Sentiment")
ax.tick_params(axis="x", rotation=0)
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("outputs/03_sentiment_x_tier.png")
plt.close()

# 2c. Sentiment trend over time (yearly)
fig, ax = plt.subplots(figsize=(10, 4))
yearly_sent = df.groupby("year")[["vs_pos", "vs_neg", "vs_neu"]].mean()
yearly_sent.plot(ax=ax, color=["#2A9D8F", "#E63946", "#A8DADC"], marker="o", linewidth=2)
ax.set_title("Average Sentiment Components by Year", fontweight="bold")
ax.set_ylabel("Average Score")
ax.set_xlabel("Year")
ax.legend(["Positive", "Negative", "Neutral"])
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("outputs/04_sentiment_trend_yearly.png")
plt.close()

print("\n✓ Sentiment analysis complete")

# ─────────────────────────────────────────────────────────────────────────────
# 3. NAMED ENTITY RECOGNITION (NLTK)
# ─────────────────────────────────────────────────────────────────────────────
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# Download required NLTK data
for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
                 "averaged_perceptron_tagger_eng",
                 "maxent_ne_chunker", "maxent_ne_chunker_tab", "words", "stopwords"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

def extract_entities(text, max_chars=1000):
    """Extract named entities from text using NLTK NE chunker."""
    entities = []
    try:
        tokens = word_tokenize(str(text)[:max_chars])
        tagged = pos_tag(tokens)
        chunked = ne_chunk(tagged, binary=False)
        for subtree in chunked:
            if isinstance(subtree, Tree):
                entity = " ".join(word for word, tag in subtree.leaves())
                label = subtree.label()
                entities.append((entity, label))
    except Exception:
        pass
    return entities

print("Running NER on posts (may take 1-2 min)...")
df["entities"] = df["text"].apply(extract_entities)

# Flatten and count
all_entities = [e for sublist in df["entities"] for e in sublist]
entity_types = ["PERSON", "GPE", "ORGANIZATION", "LOCATION", "FACILITY"]

fig, axes = plt.subplots(1, len(entity_types), figsize=(18, 5))
for ax, etype in zip(axes, entity_types):
    filtered = [e for e, t in all_entities if t == etype]
    top = Counter(filtered).most_common(10)
    if top:
        names, counts = zip(*top)
        ax.barh(list(names)[::-1], list(counts)[::-1], color=PALETTE[entity_types.index(etype) % len(PALETTE)])
        ax.set_title(etype, fontweight="bold", fontsize=11)
        ax.set_xlabel("Frequency")
        sns.despine(ax=ax)
    else:
        ax.set_visible(False)

plt.suptitle("Top Named Entities by Type", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/05_named_entities.png", bbox_inches="tight")
plt.close()

# NER density vs engagement
df["entity_count"] = df["entities"].apply(len)
print(f"\nAvg entities per post: {df['entity_count'].mean():.1f}")
print("Entity count vs note_count correlation:", df[["entity_count","note_count"]].corr().iloc[0,1].round(3))

print("✓ NER complete")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TOPIC MODELING (LDA via sklearn)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Custom stop words — HONY-specific noise
from nltk.corpus import stopwords as nltk_stopwords
try:
    base_stops = set(nltk_stopwords.words("english"))
except Exception:
    base_stops = set()

extra_stops = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "said", "like", "know", "just", "got", "get", "one", "really", "think",
    "would", "could", "going", "wanted", "want", "always", "never", "back",
    "time", "year", "years", "day", "people", "s", "t", "re", "ve", "ll",
    "didn", "wasn", "couldn", "don", "doesn", "isn", "thing", "things"
}
all_stops = base_stops | extra_stops

vectorizer = CountVectorizer(
    max_df=0.85,
    min_df=5,
    stop_words=list(all_stops),
    max_features=1500,
    ngram_range=(1, 2),
)
dtm = vectorizer.fit_transform(df["text"].fillna(""))
feature_names = vectorizer.get_feature_names_out()

N_TOPICS = 8
lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    max_iter=20,
    learning_method="online",
    random_state=42,
    n_jobs=-1,
)
lda.fit(dtm)

# Topic labels (manually interpret top words)
TOPIC_LABELS = [
    "Family & Childhood",
    "Work & Career",
    "Love & Relationships",
    "Health & Struggle",
    "Immigration & Identity",
    "Loss & Grief",
    "Dreams & Ambition",
    "War & Conflict",
]

def get_top_words(model, feature_names, n=12):
    topics = []
    for topic in model.components_:
        top_idx = topic.argsort()[-n:][::-1]
        topics.append([feature_names[i] for i in top_idx])
    return topics

top_words = get_top_words(lda, feature_names)

# Plot topic word distributions
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for idx, (ax, words, label) in enumerate(zip(axes.flatten(), top_words, TOPIC_LABELS)):
    topic_weights = lda.components_[idx]
    top_idx = topic_weights.argsort()[-10:][::-1]
    top_w = [feature_names[i] for i in top_idx]
    top_v = [topic_weights[i] for i in top_idx]
    ax.barh(top_w[::-1], top_v[::-1], color=PALETTE[idx % len(PALETTE)])
    ax.set_title(f"Topic {idx+1}: {label}", fontweight="bold", fontsize=9)
    ax.set_xlabel("Weight")
    sns.despine(ax=ax)

plt.suptitle("LDA Topic Modeling — Top Keywords per Topic", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/06_lda_topics.png", bbox_inches="tight")
plt.close()

# Assign dominant topic to each post
doc_topics = lda.transform(dtm)
df["dominant_topic"] = doc_topics.argmax(axis=1)
df["topic_label"] = df["dominant_topic"].map(dict(enumerate(TOPIC_LABELS)))
df["topic_confidence"] = doc_topics.max(axis=1)

# Topic x Engagement
fig, ax = plt.subplots(figsize=(12, 5))
topic_engagement = (
    df.groupby("topic_label")["note_count"]
    .median()
    .sort_values(ascending=True)
)
colors = [PALETTE[i % len(PALETTE)] for i in range(len(topic_engagement))]
topic_engagement.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
ax.set_title("Median Note Count by Topic", fontweight="bold")
ax.set_xlabel("Median Note Count")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("outputs/07_topic_engagement.png")
plt.close()

print(f"✓ LDA topic modeling complete ({N_TOPICS} topics)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. NARRATIVE / LINGUISTIC FEATURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

# 5a. Narrative arcs — emotional progression within a post
def sentiment_arc(text, n_chunks=5):
    """Split text into n chunks and score each — detects emotional arc."""
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    if len(sentences) < n_chunks:
        return None
    chunk_size = len(sentences) // n_chunks
    arcs = []
    for i in range(n_chunks):
        chunk = " ".join(sentences[i*chunk_size:(i+1)*chunk_size])
        score = analyzer.polarity_scores(chunk)["compound"]
        arcs.append(score)
    return arcs

print("Computing sentiment arcs...")
df["sentiment_arc"] = df["text"].apply(sentiment_arc)
valid_arcs = df["sentiment_arc"].dropna()

# Average arc by engagement tier
arc_by_tier = {}
for tier in tier_order:
    arcs = df[df["tier"] == tier]["sentiment_arc"].dropna().tolist()
    if arcs:
        arc_by_tier[tier] = np.mean(arcs, axis=0)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(1, 6)
for tier, arc in arc_by_tier.items():
    ax.plot(x, arc, marker="o", linewidth=2.5,
            color=PALETTE[tier_order.index(tier)], label=tier)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_title("Average Sentiment Arc by Engagement Tier\n(Beginning → End of post)", fontweight="bold")
ax.set_xlabel("Post Segment (1=Start, 5=End)")
ax.set_ylabel("Average VADER Compound Score")
ax.set_xticks(x)
ax.set_xticklabels(["Start", "Early", "Middle", "Late", "End"])
ax.legend(title="Tier")
sns.despine(ax=ax)
plt.tight_layout()
plt.savefig("outputs/08_sentiment_arc_by_tier.png")
plt.close()

# 5b. Linguistic feature summary by tier
ling_features = [
    "word_count", "sentence_count", "avg_sentence_length",
    "question_count", "exclaim_count", "first_person_ratio",
    "dialogue_ratio", "vs_compound", "entity_count", "topic_confidence"
]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, feat in zip(axes.flatten(), ling_features):
    data = [df[df["tier"] == t][feat].dropna() for t in tier_order]
    ax.boxplot(data, labels=tier_order, patch_artist=True,
               boxprops=dict(facecolor="#A8DADC", color="#1D3557"),
               medianprops=dict(color="#E63946", linewidth=2),
               whiskerprops=dict(color="#1D3557"),
               capprops=dict(color="#1D3557"),
               flierprops=dict(marker=".", color="#457B9D", alpha=0.4))
    ax.set_title(feat.replace("_", " ").title(), fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)
    sns.despine(ax=ax)

plt.suptitle("Linguistic Features by Engagement Tier", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/09_linguistic_features_by_tier.png", bbox_inches="tight")
plt.close()

print("✓ Narrative analysis complete")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE CORRELATION MATRIX (ML-ready)
# ─────────────────────────────────────────────────────────────────────────────
ml_features = [
    "word_count", "sentence_count", "avg_sentence_length", "paragraph_count",
    "vs_compound", "vs_pos", "vs_neg", "vs_neu",
    "question_count", "exclaim_count", "quote_count",
    "dialogue_ratio", "first_person_ratio", "uppercase_ratio",
    "hour", "dayofweek", "month", "year",
    "entity_count", "dominant_topic", "topic_confidence",
    "note_count", "log_notes",
]
corr_df = df[ml_features].corr()

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(
    corr_df, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    linewidths=0.4, annot_kws={"size": 7},
    ax=ax, square=True,
)
ax.set_title("Feature Correlation Matrix (ML-Ready Features)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/10_feature_correlation_matrix.png", bbox_inches="tight")
plt.close()

print("✓ Correlation matrix complete")

# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPORT ENRICHED DATASET
# ─────────────────────────────────────────────────────────────────────────────
export_cols = [
    "post_id", "dt", "year", "month", "dayofweek", "hour",
    "word_count", "sentence_count", "avg_sentence_length", "paragraph_count",
    "vs_compound", "vs_pos", "vs_neg", "vs_neu", "sentiment_label",
    "question_count", "exclaim_count", "quote_count",
    "dialogue_ratio", "first_person_ratio", "uppercase_ratio",
    "entity_count", "dominant_topic", "topic_label", "topic_confidence",
    "note_count", "log_notes", "tier",
]
df[export_cols].to_csv("outputs/hony_ml_ready.csv", index=False)
print(f"\n✓ ML-ready dataset exported → outputs/hony_ml_ready.csv")
print(f"  Shape: {df[export_cols].shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EDA SUMMARY")
print("="*60)
print(f"Total posts analyzed   : {len(df):,}")
print(f"Date range             : {df['dt'].min().date()} → {df['dt'].max().date()}")
print(f"\nEngagement tiers:")
for t in tier_order:
    sub = df[df['tier']==t]
    print(f"  {t:6s}: {len(sub):4d} posts | median notes: {sub['note_count'].median():,.0f}")
print(f"\nSentiment breakdown:")
for s in ["Negative","Neutral","Positive"]:
    n = (df['sentiment_label']==s).sum()
    print(f"  {s:9s}: {n:4d} posts ({n/len(df)*100:.1f}%)")
print(f"\nTop topics by median engagement:")
print(
    df.groupby("topic_label")["note_count"]
    .median().sort_values(ascending=False)
    .apply(lambda x: f"{x:,.0f}").to_string()
)
print("\nOutputs saved to ./outputs/")
print("  01_engagement_tiers.png")
print("  02_sentiment_distribution.png")
print("  03_sentiment_x_tier.png")
print("  04_sentiment_trend_yearly.png")
print("  05_named_entities.png")
print("  06_lda_topics.png")
print("  07_topic_engagement.png")
print("  08_sentiment_arc_by_tier.png")
print("  09_linguistic_features_by_tier.png")
print("  10_feature_correlation_matrix.png")
print("  hony_ml_ready.csv")