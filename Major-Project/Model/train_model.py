from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


DATASET_PATH = "Major-Project/Dataset/air_dialogue"
MODEL_PATH = "Major-Project/Model/chatbot_model.pkl"


def prepare_data():
    dataset = load_from_disk(DATASET_PATH)

    texts = []
    labels = []

    for row in dataset:
        dialogue = " ".join(row["dialogue"])
        goal = row["intent"]["goal"]

        texts.append(dialogue)
        labels.append(goal)

    return texts, labels


def train_model():
    texts, labels = prepare_data()

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            max_features=20000,
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(
            max_iter=1000
        ))
    ])

    model.fit(texts, labels)

    joblib.dump(model, MODEL_PATH)

    print("Model training completed.")
    print("Total training samples:", len(texts))
    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    train_model()