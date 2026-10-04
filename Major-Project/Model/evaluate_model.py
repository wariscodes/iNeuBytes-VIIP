from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import joblib


MODEL_PATH = "Major-Project/Model/chatbot_model.pkl"


def evaluate_model():
    print("Loading official AirDialogue validation dataset...")

    dataset = load_dataset(
        "google/air_dialogue",
        split="validation"
    )

    texts = []
    labels = []

    for row in dataset:
        dialogue = " ".join(row["dialogue"])
        goal = row["intent"]["goal"]

        texts.append(dialogue)
        labels.append(goal)

    print(f"Validation samples: {len(texts)}")

    print("\nLoading trained model...")
    model = joblib.load(MODEL_PATH)

    print("Generating predictions...")
    predictions = model.predict(texts)

    accuracy = accuracy_score(labels, predictions)

    print("\nModel Evaluation")
    print("=" * 50)
    print(f"Validation samples: {len(texts)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification Report:")
    print(
        classification_report(
            labels,
            predictions,
            labels=["book", "cancel", "change"],
            zero_division=0
        )
    )


if __name__ == "__main__":
    evaluate_model()