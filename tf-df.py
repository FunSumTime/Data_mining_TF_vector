# -------------------------
# Imports
# -------------------------
import pickle
import re

import nltk
nltk.download("punkt")
nltk.download('punkt_tab')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# -------------------------
# One-time setup (run once)
# -------------------------
# nltk.download('punkt')


# -------------------------
# Load data
# -------------------------
with open("data.pkl", "rb") as f:
    data = pickle.load(f)

print("Loaded documents:", data.keys())


# -------------------------
# Custom stop words
# -------------------------
custom_stop_words = set(ENGLISH_STOP_WORDS).union({
    "www", "http", "https", "org", "com",
    "ref", "references", "external", "links",
    "isbn", "citation"
})


# -------------------------
# Text cleaning
# -------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"\d+", "", text)              # remove numbers
    text = re.sub(r"[^a-z\s]", "", text)         # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()     # normalize spaces
    return text


# -------------------------
# Stemming
# -------------------------
stemmer = PorterStemmer()

def stem_text(text: str) -> str:
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(t) for t in tokens]
    return " ".join(stemmed)


# -------------------------
# Build corpus
# -------------------------
documents = [data[key] for key in data]

print("Number of documents:", len(documents))

documents = [clean_text(doc) for doc in documents]
documents = [stem_text(doc) for doc in documents]

# sanity check
print("\nSample cleaned text:\n", documents[0][:300])


# -------------------------
# TF-IDF Vectorization
# -------------------------
vectorizer = TfidfVectorizer(
    stop_words=sorted(custom_stop_words),
    ngram_range=(1, 2),   # safer than (1,5)
    min_df=2,
    max_df=0.85
)

tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()


# -------------------------
# Convert to DataFrame
# -------------------------
df = pd.DataFrame(
    tfidf_matrix.T.toarray(),
    index=feature_names,
    columns=[f"Doc {i+1}" for i in range(len(documents))]
)

print("\nTop TF-IDF terms for Doc 1:")
print(df.sort_values("Doc 1", ascending=False).head(15))


# -------------------------
# Visualization
# -------------------------
top_words = df["Doc 1"].sort_values(ascending=False).head(20)

plt.figure(figsize=(6, 4))
top_words.plot(kind="barh", title="Top TF-IDF Words in Document 1")
plt.xlabel("TF-IDF Score")
plt.tight_layout()
plt.savefig("first_doc_with_nltk_20_words.png", dpi=300)
plt.close()

print("\nSaved plot as first_doc.png")


import matplotlib.pyplot as plt
import numpy as np

# Use the 'df' from the previous TF-IDF example
# Filter for top 5 terms by max score across documents
df['max_score'] = df.max(axis=1)
df_top = df.sort_values('max_score', ascending=False).head(20)
df_top = df_top.drop('max_score', axis=1)

data = df_top.values
terms = df_top.index
docs = df_top.columns

fig, ax = plt.subplots()
im = ax.imshow(data, cmap='YlGnBu')

# Labels
ax.set_xticks(np.arange(len(docs)))
ax.set_yticks(np.arange(len(terms)))
ax.set_xticklabels(docs)
ax.set_yticklabels(terms)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("TF-IDF Score", rotation=-90, va="bottom")

plt.title("Top Terms Heatmap")
plt.tight_layout()
plt.show()
plt.savefig("heatmap.png", dpi=300)