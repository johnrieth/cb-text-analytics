# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "scikit-learn",
#   "nltk",
#   "pandas",
#   "matplotlib",
# ]
# ///

import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk

from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# --- Load ---

def load_statements(directory):
    records = []
    for path in sorted(Path(directory).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        records.append({"date": path.stem, "text": text, "source": directory})
    return records

records = (
    load_statements("usa-central-bank/fomc-statements") +
    load_statements("nz-central-bank/ocr")
)

df = pd.DataFrame(records)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

print(f"Loaded {len(df)} statements")
print(df.groupby("source")["date"].agg(["min", "max"]))

# --- Preprocess ---

stop_words = set(stopwords.words("english"))
stop_words.update({"committee", "federal", "reserve", "percent", "bank", "will", "also", "rate"})

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

df["clean"] = df["text"].apply(clean)

# --- Fit LDA ---

N_TOPICS = 5

vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
dtm = vectorizer.fit_transform(df["clean"])

lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42, max_iter=20)
lda.fit(dtm)

# --- Print topics ---

vocab = vectorizer.get_feature_names_out()
print("\nTop words per topic:")
for i, component in enumerate(lda.components_):
    top_words = [vocab[j] for j in component.argsort()[-10:][::-1]]
    print(f"  Topic {i+1}: {', '.join(top_words)}")

# --- Plot topic prevalence over time ---

topic_dist = lda.transform(dtm)
topic_cols = [f"topic_{i+1}" for i in range(N_TOPICS)]
topic_df = pd.DataFrame(topic_dist, columns=topic_cols, index=df["date"])

fig, ax = plt.subplots(figsize=(14, 5))
topic_df.plot(ax=ax, title="Topic Prevalence Over Time (FOMC + RBNZ)")
ax.set_xlabel("Date")
ax.set_ylabel("Topic Weight")
plt.tight_layout()
plt.savefig("quick_analysis_results.png", dpi=150)
print("\nSaved quick_analysis_results.png")