from datasets import load_from_disk
from sklearn.model_selection import train_test_split
import joblib


DATASET_PATH = "Major-Project/Dataset/air_dialogue"
MODEL_PATH = "Major-Project/Model/chatbot_model.pkl"
OUTPUT_PATH = "Major-Project/Model/misclassified_examples.txt"


def load_data():
    dataset = load_from_disk(DATASET_PATH)

    texts = []
    labels = []

    for row in dataset:
        dialogue = " ".join(row["dialogue"])
        goal = row["intent"]["goal"]

        texts.append(dialogue)
        labels.append(goal)

    return texts, labels


def find_misclassified_examples():
    texts, labels = load_data()

    _, test_texts, _, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(test_texts)

    mistakes = []

    for text, actual, predicted in zip(
        test_texts, test_labels, predictions
    ):
        if actual != predicted:
            mistakes.append((text, actual, predicted))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        file.write("Misclassified Examples\n")
        file.write("=" * 70 + "\n\n")

        for i, (text, actual, predicted) in enumerate(mistakes[:20], 1):
            file.write(f"Example {i}\n")
            file.write(f"Actual intent: {actual}\n")
            file.write(f"Predicted intent: {predicted}\n")
            file.write(f"Dialogue: {text}\n")
            file.write("-" * 70 + "\n")

    print("Total misclassified examples:", len(mistakes))
    print("First 20 examples saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    find_misclassified_examples()