import os
import re
import string


# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "aclImdb")


def clean_text(text):
    """
    Clean a movie review for sentiment analysis.
    """

    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_reviews(folder_path):
    """
    Load reviews from a folder and return cleaned reviews.
    """

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


# Load training reviews
train_positive = load_reviews(train_positive_path)
train_negative = load_reviews(train_negative_path)

# Load testing reviews
test_positive = load_reviews(test_positive_path)
test_negative = load_reviews(test_negative_path)


# Combine reviews and labels
train_reviews = train_positive + train_negative
train_labels = [1] * len(train_positive) + [0] * len(train_negative)

test_reviews = test_positive + test_negative
test_labels = [1] * len(test_positive) + [0] * len(test_negative)


print("IMDb Preprocessing Check")
print("------------------------")

print("Training reviews:", len(train_reviews))
print("Training labels:", len(train_labels))

print("Testing reviews:", len(test_reviews))
print("Testing labels:", len(test_labels))

print("\nFirst cleaned review:")
print(train_reviews[0])

print("\nFirst review label:")
print(train_labels[0])