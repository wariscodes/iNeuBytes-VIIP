import os
import re
import string
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "aclImdb")


def clean_text(text):
    """Clean a movie review."""

    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_reviews(folder_path):
    """Load and clean all reviews from a folder."""

    reviews = []

    for filename in os.listdir(folder_path):

        if filename.endswith(".txt"):

            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                review = file.read()

            reviews.append(clean_text(review))

    return reviews


# Dataset paths
train_positive_path = os.path.join(DATASET_PATH, "train", "pos")
train_negative_path = os.path.join(DATASET_PATH, "train", "neg")

test_positive_path = os.path.join(DATASET_PATH, "test", "pos")
test_negative_path = os.path.join(DATASET_PATH, "test", "neg")


# Load training data
train_positive = load_reviews(train_positive_path)
train_negative = load_reviews(train_negative_path)

# Load testing data
test_positive = load_reviews(test_positive_path)
test_negative = load_reviews(test_negative_path)


# Combine reviews and labels
train_reviews = train_positive + train_negative
train_labels = [1] * len(train_positive) + [0] * len(train_negative)

test_reviews = test_positive + test_negative
test_labels = [1] * len(test_positive) + [0] * len(test_negative)


print("TF-IDF Max Features + Logistic Regression")
print("----------------------------------------")

print("Training reviews:", len(train_reviews))
print("Testing reviews:", len(test_reviews))


# Limit vocabulary to 50,000 features
vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),
    max_features=50000
)


# Convert text into numerical features
X_train = vectorizer.fit_transform(train_reviews)
X_test = vectorizer.transform(test_reviews)

print("Training TF-IDF shape:", X_train.shape)
print("Testing TF-IDF shape:", X_test.shape)


# Train Logistic Regression
start_time = time.time()

model = LogisticRegression(max_iter=1000)

model.fit(X_train, train_labels)

training_time = time.time() - start_time


# Predictions
predictions = model.predict(X_test)


# Evaluation metrics
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)


print("\nResults")
print("-------")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print(f"\nTraining time: {training_time:.2f} seconds")