from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt


DATASET_PATH = "Major-Project/Dataset/air_dialogue"
MODEL_PATH = "Major-Project/Model/chatbot_model.pkl"
OUTPUT_PATH = "Major-Project/Model/confusion_matrix.png"


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


def create_confusion_matrix():
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

    labels_order = ["book", "cancel", "change"]

    matrix = confusion_matrix(
        test_labels,
        predictions,
        labels=labels_order
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels_order
    )

    display.plot()
    plt.title("AirDialogue Intent Classification")
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.show()

    print("Confusion matrix saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    create_confusion_matrix()