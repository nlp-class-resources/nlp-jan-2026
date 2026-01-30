import json
import os

# Notebook structure
nb = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

def add_markdown(source):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source]
    })

def add_code(source):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source]
    })

# --- Content ---

add_markdown([
    "# Review Summarization Tutorial",
    "",
    "In this tutorial, we will build a review summarization feature similar to Amazon's review highlights. We will use NLTK for text preprocessing, identify common bigrams (two-word phrases), analyze their sentiment, and cluster them using Word2Vec.",
    "",
    "**Goal:** Show meaningful analysis/feature outcome from simple bigram probabilities.",
    "",
    "**Steps:**",
    "1. EDA and Text Preprocessing using NLTK",
    "2. Identify most common bigrams",
    "3. Identify bigram probabilities for positive/negative reviews",
    "4. Use Word2Vec for clustering",
    "5. Draw conclusions"
])

add_code([
    "import pandas as pd",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "import nltk",
    "from nltk.corpus import stopwords",
    "from nltk.tokenize import word_tokenize",
    "from nltk.stem import WordNetLemmatizer",
    "from nltk import ngrams",
    "from collections import Counter",
    "import gensim",
    "from gensim.models import Word2Vec",
    "from gensim.models.phrases import Phrases, Phraser",
    "from sklearn.cluster import KMeans",
    "",
    "# Download necessary NLTK data",
    "nltk.download('punkt')",
    "nltk.download('stopwords')",
    "nltk.download('wordnet')",
    "nltk.download('omw-1.4')"
])

add_markdown([
    "## Load Data",
    "We will load the reviews and items datasets."
])

add_code([
    "df_reviews = pd.read_csv(\"20191226-reviews.csv\")",
    "df_items = pd.read_csv(\"20191226-items.csv\")",
    "",
    "print(\"Reviews shape:\", df_reviews.shape)",
    "print(\"Items shape:\", df_items.shape)",
    "df_reviews.head()"
])

add_markdown([
    "## Step 1: EDA and Text Preprocessing",
    "We'll start by exploring the rating distribution and then preprocess the text data."
])

add_code([
    "# Rating Distribution",
    "plt.figure(figsize=(8, 5))",
    "sns.countplot(x='rating', data=df_reviews)",
    "plt.title('Rating Distribution')",
    "plt.show()"
])

add_markdown([
    "### Text Preprocessing",
    "We will perform the following steps:",
    "1. **Tokenization:** Splitting text into words.",
    "2. **Lowercasing:** Converting all text to lower case.",
    "3. **Stopword Removal:** Removing common words like 'the', 'is', 'and' that don't carry much meaning.",
    "4. **Lemmatization:** Converting words to their base form (e.g., 'batteries' -> 'battery').",
    "5. **Filtering:** Keeping only alphabetic tokens."
])

add_code([
    "stop_words = set(stopwords.words('english'))",
    "lemmatizer = WordNetLemmatizer()",
    "",
    "def preprocess_text(text):",
    "    if not isinstance(text, str):",
    "        return []",
    "    # Tokenize",
    "    tokens = word_tokenize(text.lower())",
    "    # Remove stopwords and non-alphabetic tokens, lemmatize",
    "    clean_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]",
    "    return clean_tokens",
    "",
    "# Apply preprocessing (using a subset for speed if needed, but here we do all)",
    "# Dropping rows with missing body",
    "df_reviews = df_reviews.dropna(subset=['body'])",
    "df_reviews['clean_tokens'] = df_reviews['body'].apply(preprocess_text)",
    "df_reviews[['body', 'clean_tokens']].head()"
])

add_markdown([
    "## Step 2: Identify Most Common Bigrams",
    "Bigrams are pairs of consecutive words. They often represent features (e.g., 'battery life', 'camera quality')."
])

add_code([
    "def get_bigrams(tokens_list):",
    "    all_bigrams = []",
    "    for tokens in tokens_list:",
    "        all_bigrams.extend(list(ngrams(tokens, 2)))",
    "    return all_bigrams",
    "",
    "bigrams = get_bigrams(df_reviews['clean_tokens'])",
    "bigram_counts = Counter(bigrams)",
    "",
    "# Top 20 bigrams",
    "common_bigrams = bigram_counts.most_common(20)",
    "print(\"Top 20 Bigrams:\")",
    "for b, c in common_bigrams:",
    "    print(f\"{b}: {c}\")",
    "",
    "# Visualize",
    "bigram_labels = [f\"{b[0]} {b[1]}\" for b, c in common_bigrams]",
    "bigram_values = [c for b, c in common_bigrams]",
    "",
    "plt.figure(figsize=(12, 6))",
    "sns.barplot(x=bigram_values, y=bigram_labels)",
    "plt.title('Top 20 Most Common Bigrams')",
    "plt.xlabel('Frequency')",
    "plt.show()"
])

add_markdown([
    "## Step 3: Identify Bigram Probabilities for Positive/Negative Reviews",
    "We want to see how these bigrams are distributed across positive and negative reviews.",
    "We define **Positive** as rating > 3 and **Negative** as rating <= 3."
])

add_code([
    "# Define Sentiment",
    "df_reviews['sentiment'] = df_reviews['rating'].apply(lambda x: 'positive' if x > 3 else 'negative')",
    "",
    "# Separate tokens",
    "pos_tokens = df_reviews[df_reviews['sentiment'] == 'positive']['clean_tokens']",
    "neg_tokens = df_reviews[df_reviews['sentiment'] == 'negative']['clean_tokens']",
    "",
    "# Get bigrams for each sentiment",
    "pos_bigrams = get_bigrams(pos_tokens)",
    "neg_bigrams = get_bigrams(neg_tokens)",
    "",
    "pos_counts = Counter(pos_bigrams)",
    "neg_counts = Counter(neg_bigrams)",
    "",
    "# Analyze Top 50 Global Bigrams",
    "top_50_bigrams = [b for b, c in bigram_counts.most_common(50)]",
    "",
    "data = []",
    "for bg in top_50_bigrams:",
    "    bg_str = f\"{bg[0]} {bg[1]}\"",
    "    p_count = pos_counts[bg]",
    "    n_count = neg_counts[bg]",
    "    total = p_count + n_count",
    "    if total > 0:",
    "        pos_prob = p_count / total",
    "        neg_prob = n_count / total",
    "        data.append({",
    "            'bigram': bg_str,",
    "            'positive': p_count,",
    "            'negative': n_count,",
    "            'pos_prob': pos_prob,",
    "            'neg_prob': neg_prob,",
    "            'total': total",
    "        })",
    "",
    "df_bigram_sentiment = pd.DataFrame(data)",
    "df_bigram_sentiment = df_bigram_sentiment.sort_values('total', ascending=False)",
    "",
    "print(df_bigram_sentiment.head(10))"
])

add_markdown([
    "### Visualize Sentiment Breakdown",
    "This stacked bar chart shows the number of positive and negative mentions for the top features."
])

add_code([
    "top_features = df_bigram_sentiment.head(15)",
    "top_features.set_index('bigram')[['positive', 'negative']].plot(kind='bar', stacked=True, figsize=(14, 7), color=['green', 'red'])",
    "plt.title('Sentiment Breakdown for Top Features (Bigrams)')",
    "plt.ylabel('Count')",
    "plt.xticks(rotation=45, ha='right')",
    "plt.show()"
])

add_markdown([
    "## Step 4: Use Word2Vec for Clustering",
    "We will use Word2Vec to learn vector representations of our words and bigrams. Then we'll cluster them to group similar features (e.g., 'battery life' and 'battery backup').",
    "",
    "First, we use Gensim's `Phrases` to automatically detect common bigrams in the text and treat them as single tokens (e.g., 'battery_life')."
])

add_code([
    "# Detect phrases (bigrams)",
    "phrases = Phrases(df_reviews['clean_tokens'], min_count=5, threshold=10)",
    "bigram_phraser = Phraser(phrases)",
    "",
    "# Transform sentences",
    "bigram_sentences = [bigram_phraser[sent] for sent in df_reviews['clean_tokens']]",
    "",
    "# Train Word2Vec",
    "model_bg = Word2Vec(sentences=bigram_sentences, vector_size=100, window=5, min_count=5, workers=4)",
    "",
    "# Get vectors for our top bigrams",
    "# We need to match the bigrams from Step 2 to the Phraser's output format (usually joined by _)",
    "feature_vectors = []",
    "feature_names = []",
    "",
    "for bg in top_50_bigrams:",
    "    bg_str = f\"{bg[0]}_{bg[1]}\"",
    "    if bg_str in model_bg.wv:",
    "        feature_vectors.append(model_bg.wv[bg_str])",
    "        feature_names.append(bg_str)",
    "    else:",
    "        # Fallback: try average of individual word vectors if bigram not found as a unit",
    "        if bg[0] in model_bg.wv and bg[1] in model_bg.wv:",
    "             vec = (model_bg.wv[bg[0]] + model_bg.wv[bg[1]]) / 2",
    "             feature_vectors.append(vec)",
    "             feature_names.append(bg_str)",
    "",
    "print(f\"Vectorized {len(feature_vectors)} features.\")"
])

add_code([
    "# Clustering with K-Means",
    "num_clusters = 5  # Adjust based on desired granularity",
    "kmeans = KMeans(n_clusters=num_clusters, random_state=42)",
    "clusters = kmeans.fit_predict(feature_vectors)",
    "",
    "# Group features by cluster",
    "cluster_map = {}",
    "for i, cluster_id in enumerate(clusters):",
    "    if cluster_id not in cluster_map:",
    "        cluster_map[cluster_id] = []",
    "    cluster_map[cluster_id].append(feature_names[i])",
    "",
    "print(\"Feature Clusters:\")",
    "for c_id, features in cluster_map.items():",
    "    print(f\"Cluster {c_id}: {features}\")"
])

add_markdown([
    "## Conclusions & Feature Summaries",
    "Now we can generate an Amazon-style summary for the top features, grouped by their clusters."
])

add_code([
    "def get_feature_summary(feature_name_str):",
    "    # feature_name_str format: \"word1_word2\"",
    "    parts = feature_name_str.split('_')",
    "    if len(parts) == 2:",
    "        bg = (parts[0], parts[1])",
    "    else:",
    "        return # Skip if not a bigram",
    "    ",
    "    if bg in pos_counts or bg in neg_counts:",
    "        p_count = pos_counts[bg]",
    "        n_count = neg_counts[bg]",
    "        total = p_count + n_count",
    "        ",
    "        print(f\"Feature: {parts[0]} {parts[1]}\")",
    "        print(f\"Total mentions: {total}\")",
    "        print(f\"Positive: {p_count} ({p_count/total:.1%})\")",
    "        print(f\"Negative: {n_count} ({n_count/total:.1%})\")",
    "        print(\"-\" * 30)",
    "",
    "# Display summaries for features in each cluster",
    "for c_id, features in cluster_map.items():",
    "    print(f\"\\n--- Cluster {c_id} ---\")",
    "    for feat in features[:3]: # Show top 3 per cluster to save space",
    "        get_feature_summary(feat)"
])

# Write to file
with open("lecture3/review_summarization.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
