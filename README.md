# Movie Review Sentiment Classification  
### From Scratch Implementation of Naive Bayes and Logistic Regression

This project implements a complete NLP pipeline for classifying IMDB movie reviews as positive or negative, using machine learning models built entirely from scratch.

---

## Project Overview

- Implementation of core machine learning algorithms without using libraries like scikit-learn  
- End-to-end pipeline including preprocessing, feature extraction, training, and evaluation  
- Uses the IMDB Large Movie Review Dataset (50,000 samples)  
- Comparative analysis of Naive Bayes and Logistic Regression  

---

## Dataset

- IMDB Large Movie Review Dataset  
- 50,000 labeled reviews  
  - 25,000 positive  
  - 25,000 negative  
- Source: HuggingFace `datasets` library  

---

## Models Implemented

### Naive Bayes (Multinomial)
- Based on Bayes Theorem  
- Uses Laplace smoothing  
- Operates on word frequency distributions  

### Logistic Regression
- Implemented using mini-batch gradient descent  
- Sigmoid activation function  
- Binary cross-entropy loss  
- L2 regularization  

---

## Features

### Text Preprocessing
- Lowercasing  
- HTML tag removal  
- Contraction expansion  
- Stopword removal  
- Tokenization  

### Bag-of-Words Vectorizer
- Vocabulary pruning using minimum frequency  
- Fixed-size feature vectors  
- Efficient transformation of text to numerical format  

### Data Splitting
- Stratified train-test split to preserve class distribution  

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrix  

---


(Logistic Regression typically achieves higher accuracy due to its discriminative nature.)

---

## Installation

### Clone the repository
```bash
git clone https://github.com/your-username/movie-review-classifier.git
cd movie-review-classifier
```
## Key Learnings
Understanding of probabilistic models (Naive Bayes)
Implementation of optimization techniques (Gradient Descent)
Differences between generative and discriminative models
Importance of preprocessing in NLP pipelines
