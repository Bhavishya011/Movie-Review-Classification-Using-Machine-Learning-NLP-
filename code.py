"""
============================================================
  Movie Review Classification — Naive Bayes & Logistic Regression
  From First Principles | Real IMDB Dataset
============================================================


DATASET USED
------------
IMDB Large Movie Review Dataset (Maas et al., 2011)
50,000 reviews | 25,000 positive, 25,000 negative
Source: https://huggingface.co/datasets/imdb
"""

import re
import math
import random
import os
import sys
import csv
import time
from collections import defaultdict

# ═══════════════════════════════════════════════════════════
#  (a)  TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "this", "that", "these",
    "those", "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "it", "they", "them", "his", "her", "its", "their", "what", "which",
    "who", "whom", "as", "if", "then", "than", "so", "yet", "both",
    "not", "no", "nor", "up", "out", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "each", "more", "most",
    "other", "some", "such", "own", "same", "just", "because", "while",
    "although", "however", "therefore", "thus", "also", "very", "too",
    "only", "well", "even", "still", "here", "there", "when", "where",
    "why", "how", "all", "any", "few", "again", "further", "once",
    "br", "www", "http", "com"   # HTML/URL artifacts common in IMDB
}

CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "couldn't": "could not",
    "wouldn't": "would not", "shouldn't": "should not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "doesn't": "does not", "don't": "do not", "didn't": "did not",
    "n't": " not", "'re": " are", "'ve": " have",
    "'ll": " will", "'d": " would", "'m": " am"
}

def clean_text(text):
    """
    Pre-process raw review text:
      1. Normalize  — lowercase + expand contractions
      2. Strip HTML — remove <br />, <p>, etc.
      3. Remove special characters — keep only alphabetic + spaces
      4. Stop word + short token removal
    Returns a list of clean tokens.
    """
    # Lowercase
    text = text.lower()

    # Strip HTML tags (IMDB reviews contain <br /> etc.)
    text = re.sub(r"<[^>]+>", " ", text)

    # Expand contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)

    # Remove non-alpha characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize and filter
    tokens = [
        t for t in text.split()
        if t not in STOP_WORDS and len(t) > 2
    ]
    return tokens


# ═══════════════════════════════════════════════════════════
#  DATASET LOADING
# ═══════════════════════════════════════════════════════════

DATASET_FILE = "imdb_reviews.tsv"   # cached TSV after first download

def download_and_save_dataset():
    """
    Download the IMDB dataset via HuggingFace `datasets` and save
    as a local TSV file (label \\t text) for fast future loading.
    Requires:  pip install datasets
    """
    print("\n[DOWNLOAD] Fetching IMDB dataset from HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' library not found.")
        print("        Install it with:  pip install datasets")
        sys.exit(1)

    ds = load_dataset("imdb")

    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for split in ["train", "test"]:
            for item in ds[split]:
                text = item["text"].replace("\n", " ").replace("\t", " ")
                f.write(f"{item['label']}\t{text}\n")

    total = sum(1 for _ in open(DATASET_FILE, encoding="utf-8"))
    print(f"[DOWNLOAD] Saved {total:,} reviews → {DATASET_FILE}")


def load_tsv_dataset(filepath, max_samples=None):
    """
    Load pre-saved TSV dataset.
    Returns list of (tokens, label) tuples.
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            label, text = int(parts[0]), parts[1]
            data.append((clean_text(text), label))
            if max_samples and len(data) >= max_samples:
                break
    return data


def load_any_csv(filepath):
    """
    Flexible loader for CSV files with columns 'review'/'sentiment'
    or 'text'/'label'. Useful for Kaggle IMDB CSV.
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = [h.lower() for h in reader.fieldnames]
        text_col  = next((h for h in reader.fieldnames if h.lower() in ("review","text","comment")), None)
        label_col = next((h for h in reader.fieldnames if h.lower() in ("sentiment","label","class")), None)
        if not text_col or not label_col:
            raise ValueError(f"Cannot find text/label columns. Found: {reader.fieldnames}")
        for row in reader:
            label_raw = row[label_col].strip().lower()
            label = 1 if label_raw in ("positive","pos","1") else 0
            data.append((clean_text(row[text_col]), label))
    return data


# ═══════════════════════════════════════════════════════════
#  (b)  STRATIFIED TRAIN / TEST SPLIT  (ratio = 0.2)
# ═══════════════════════════════════════════════════════════

def train_test_split(data, test_ratio=0.2, seed=42):
    """
    Stratified split: maintains original class proportions
    in both train and test sets.
    """
    random.seed(seed)
    classes = defaultdict(list)
    for item in data:
        classes[item[1]].append(item)

    train, test = [], []
    for label, items in classes.items():
        shuffled = items[:]
        random.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * test_ratio))
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])

    random.shuffle(train)
    random.shuffle(test)
    return train, test


# ═══════════════════════════════════════════════════════════
#  (c)  BAG OF WORDS MODEL
# ═══════════════════════════════════════════════════════════

class BagOfWords:
    """
    Bag-of-Words vectorizer with minimum frequency threshold.
    Low-frequency words are typically noise; cutting them reduces
    dimensionality and improves generalization.
    """
    def __init__(self, min_freq=3, max_vocab=20000):
        self.min_freq  = min_freq
        self.max_vocab = max_vocab
        self.vocab     = {}    # word → index
        self.idf       = {}    # word → idf weight (optional, for TF-IDF)

    def fit(self, token_lists):
        freq = defaultdict(int)
        for tokens in token_lists:
            for t in tokens:
                freq[t] += 1

        # Sort by frequency descending, keep top max_vocab
        sorted_words = sorted(
            [(w, c) for w, c in freq.items() if c >= self.min_freq],
            key=lambda x: -x[1]
        )[:self.max_vocab]

        self.vocab = {word: idx for idx, (word, _) in enumerate(sorted_words)}
        return self

    def transform(self, token_lists):
        """Convert token lists to BoW count vectors."""
        V = len(self.vocab)
        vectors = []
        for tokens in token_lists:
            vec = [0] * V
            for t in tokens:
                if t in self.vocab:
                    vec[self.vocab[t]] += 1
            vectors.append(vec)
        return vectors

    def fit_transform(self, token_lists):
        self.fit(token_lists)
        return self.transform(token_lists)


# ═══════════════════════════════════════════════════════════
#  (d)  NAIVE BAYES CLASSIFIER
# ═══════════════════════════════════════════════════════════

class MyNB:
    """
    Multinomial Naive Bayes Classifier with Laplace Smoothing.

    Bayes' Theorem:
        P(c | x) ∝ P(c) × ∏ P(w_i | c)^{count(w_i)}

    Taking logs to avoid underflow:
        log P(c | x) = log P(c) + Σ count(w_i) × log P(w_i | c)

    Laplace Smoothing:
        P(w | c) = (count(w, c) + α) / (total_words_in_c + α × |V|)
    """
    def __init__(self, alpha=1.0):
        self.alpha              = alpha
        self.classes            = []
        self.class_log_prior    = {}
        self.feature_log_prob   = {}   # class → list of log P(word | class)

    def fit(self, X, y):
        self.classes  = list(set(y))
        n_samples     = len(y)
        V             = len(X[0])

        for c in self.classes:
            idx = [i for i, label in enumerate(y) if label == c]

            # Log prior
            self.class_log_prior[c] = math.log(len(idx) / n_samples)

            # Aggregate word counts for class c
            word_counts = [0] * V
            for i in idx:
                for j in range(V):
                    word_counts[j] += X[i][j]

            total = sum(word_counts)

            # Smoothed log-likelihoods
            self.feature_log_prob[c] = [
                math.log((word_counts[j] + self.alpha) /
                         (total + self.alpha * V))
                for j in range(V)
            ]
        return self

    def predict(self, X):
        preds = []
        for x in X:
            scores = {}
            for c in self.classes:
                score = self.class_log_prior[c]
                for j, count in enumerate(x):
                    if count > 0:
                        score += count * self.feature_log_prob[c][j]
                scores[c] = score
            preds.append(max(scores, key=scores.get))
        return preds


# ═══════════════════════════════════════════════════════════
#  (e)  LOGISTIC REGRESSION CLASSIFIER
# ═══════════════════════════════════════════════════════════

class MyLogR:
    """
    Binary Logistic Regression trained with Mini-Batch Gradient Descent.

    Model:    P(y=1 | x) = σ(w·x + b)   where σ(z) = 1/(1+e^{-z})
    Loss:     Binary Cross-Entropy = -[y log ŷ + (1-y) log(1-ŷ)]
    Update:   w ← w + lr × (1/B) × Σ (y_i - ŷ_i) × x_i  −  lr × λ × w
              b ← b + lr × (1/B) × Σ (y_i - ŷ_i)
    L2 reg:   Penalizes large weights to reduce overfitting.
    """
    def __init__(self, lr=0.1, n_iter=30, batch_size=64, reg_lambda=0.001):
        self.lr          = lr
        self.n_iter      = n_iter        # epochs
        self.batch_size  = batch_size
        self.reg_lambda  = reg_lambda
        self.weights     = None
        self.bias        = 0.0

    @staticmethod
    def _sigmoid(z):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-min(z, 500)))
        e = math.exp(max(z, -500))
        return e / (1.0 + e)

    def _dot(self, w, x):
        return sum(wi * xi for wi, xi in zip(w, x))

    def fit(self, X, y):
        V = len(X[0])
        self.weights = [0.0] * V
        self.bias    = 0.0
        n            = len(X)
        indices      = list(range(n))

        for epoch in range(self.n_iter):
            random.shuffle(indices)

            # Mini-batch gradient descent
            for start in range(0, n, self.batch_size):
                batch = indices[start: start + self.batch_size]
                B     = len(batch)

                grad_w = [0.0] * V
                grad_b = 0.0

                for i in batch:
                    z     = self._dot(self.weights, X[i]) + self.bias
                    y_hat = self._sigmoid(z)
                    error = y[i] - y_hat

                    grad_b += error
                    for j in range(V):
                        if X[i][j] != 0:
                            grad_w[j] += error * X[i][j]

                # Update weights with L2 regularization
                scale = self.lr / B
                for j in range(V):
                    self.weights[j] += scale * grad_w[j] - \
                                       self.lr * self.reg_lambda * self.weights[j]
                self.bias += scale * grad_b

            # Print epoch loss every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                loss = self._cross_entropy_loss(X, y)
                print(f"    Epoch {epoch+1:>3}/{self.n_iter}  |  Loss: {loss:.4f}")

        return self

    def _cross_entropy_loss(self, X, y):
        total = 0.0
        eps   = 1e-12
        for i in range(len(X)):
            z     = self._dot(self.weights, X[i]) + self.bias
            y_hat = self._sigmoid(z)
            total += -(y[i] * math.log(y_hat + eps) +
                       (1 - y[i]) * math.log(1 - y_hat + eps))
        return total / len(X)

    def predict_proba(self, X):
        return [self._sigmoid(self._dot(self.weights, x) + self.bias) for x in X]

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]


# ═══════════════════════════════════════════════════════════
#  EVALUATION UTILITIES
# ═══════════════════════════════════════════════════════════

def accuracy(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

def confusion_matrix(y_true, y_pred, classes=(0, 1)):
    cm = {(a, b): 0 for a in classes for b in classes}
    for t, p in zip(y_true, y_pred):
        cm[(t, p)] += 1
    return cm

def classification_report(y_true, y_pred, model_name="Model"):
    cm   = confusion_matrix(y_true, y_pred)
    acc  = accuracy(y_true, y_pred)

    # Per-class precision, recall, F1
    lines = []
    lines.append(f"\n{'─'*52}")
    lines.append(f"  {model_name}")
    lines.append(f"{'─'*52}")
    lines.append(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    lines.append(f"  {'─'*44}")

    for c in (0, 1):
        tp  = cm[(c, c)]
        fp  = sum(cm[(t, c)] for t in (0,1) if t != c)
        fn  = sum(cm[(c, p)] for p in (0,1) if p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec  = tp / (tp + fn) if (tp + fn) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        lbl  = "Negative" if c == 0 else "Positive"
        lines.append(f"  {lbl:<12} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

    lines.append(f"  {'─'*44}")
    lines.append(f"  {'Accuracy':<12} {'':>10} {'':>10} {acc:>10.4f}")
    lines.append(f"{'─'*52}")

    print("\n".join(lines))
    return acc


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]

    # ── Handle --download flag ───────────────────────────────
    if "--download" in args:
        download_and_save_dataset()
        if "--run" not in args:
            print("\nDataset saved. Now run:  python solution_imdb.py")
            return

    print("\n" + "=" * 52)
    print("  Movie Review Sentiment Classification")
    print("  Naive Bayes & Logistic Regression")
    print("  From First Principles | IMDB Dataset")
    print("=" * 52)

    # ── Load dataset ─────────────────────────────────────────
    custom_file = next((a for a in args if a.endswith((".tsv", ".csv", ".txt"))
                        and not a.startswith("--")), None)

    if custom_file and os.path.exists(custom_file):
        print(f"\n[DATA] Loading from: {custom_file}")
        data = load_any_csv(custom_file) if custom_file.endswith(".csv") \
               else load_tsv_dataset(custom_file)

    elif os.path.exists(DATASET_FILE):
        print(f"\n[DATA] Loading cached IMDB dataset: {DATASET_FILE}")
        # Use 10,000 samples for reasonable speed; increase for higher accuracy
        data = load_tsv_dataset(DATASET_FILE, max_samples=10000)

    else:
        print("\n[ERROR] No dataset found.")
        print("  Option 1: python solution_imdb.py --download   (auto-downloads IMDB)")
        print("  Option 2: python solution_imdb.py myfile.tsv   (your own dataset)")
        print("            File format: one sample per line → <label>\\t<review text>")
        sys.exit(1)

    print(f"[DATA] Loaded {len(data):,} reviews")

    # Class distribution
    pos = sum(1 for _, l in data if l == 1)
    neg = len(data) - pos
    print(f"[DATA] Positive: {pos:,} | Negative: {neg:,}")

    # ── Stratified split ─────────────────────────────────────
    train_data, test_data = train_test_split(data, test_ratio=0.2)
    print(f"\n[SPLIT] Train: {len(train_data):,} | Test: {len(test_data):,}")

    X_train_tok = [d[0] for d in train_data]
    y_train     = [d[1] for d in train_data]
    X_test_tok  = [d[0] for d in test_data]
    y_test      = [d[1] for d in test_data]

    # ── Bag of Words ─────────────────────────────────────────
    print("\n[BOW] Building vocabulary (min_freq=3, max_vocab=20,000)...")
    t0  = time.time()
    bow = BagOfWords(min_freq=3, max_vocab=20000)
    X_train = bow.fit_transform(X_train_tok)
    X_test  = bow.transform(X_test_tok)
    print(f"[BOW] Vocabulary size: {len(bow.vocab):,} words  ({time.time()-t0:.1f}s)")

    # ── Naive Bayes ──────────────────────────────────────────
    print("\n[NB]  Training Naive Bayes...")
    t0 = time.time()
    nb = MyNB(alpha=1.0)
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    print(f"[NB]  Done ({time.time()-t0:.1f}s)")
    nb_acc = classification_report(y_test, nb_pred, "Naive Bayes")

    # ── Logistic Regression ──────────────────────────────────
    print("\n[LR]  Training Logistic Regression (mini-batch GD)...")
    t0 = time.time()
    lr = MyLogR(lr=0.1, n_iter=30, batch_size=64, reg_lambda=0.001)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    print(f"[LR]  Done ({time.time()-t0:.1f}s)")
    lr_acc = classification_report(y_test, lr_pred, "Logistic Regression")

    # ── Final Summary ─────────────────────────────────────────
    print("\n" + "═" * 52)
    print(f"  FINAL RESULTS")
    print(f"  Naive Bayes Accuracy       : {nb_acc * 100:.2f}%")
    print(f"  Logistic Regression Acc    : {lr_acc * 100:.2f}%")
    print("═" * 52 + "\n")


if __name__ == "__main__":
    main()
