# Task 2 — Sentiment Analysis Using Machine Learning and Deep Learning

## 1. Project Overview

This project performs sentiment analysis on movie reviews using classical Machine Learning models and a Deep Learning LSTM model.

The project uses the IMDb Large Movie Review Dataset and compares TF-IDF based Machine Learning approaches with an LSTM-based Deep Learning approach.

The main objective is to:
- Preprocess text reviews.
- Convert text into numerical features using TF-IDF.
- Train Logistic Regression and Linear SVM models.
- Perform controlled TF-IDF experiments.
- Build and evaluate an LSTM sentiment classification model.
- Perform controlled LSTM experiments.
- Compare model performance using Accuracy, Precision, Recall, and F1-score.
- Analyze misclassified reviews and model behavior.

---

## 2. Dataset

### IMDb Large Movie Review Dataset

The dataset contains:

- 25,000 training reviews
- 25,000 testing reviews
- Positive and negative reviews
- Balanced classes

The dataset is divided into positive and negative review folders.

### Dataset Structure

```text
Dataset/
└── aclImdb/
    ├── train/
    │   ├── pos/
    │   └── neg/
    │
    └── test/
        ├── pos/
        └── neg/