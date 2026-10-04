import os
import random
import re
import string

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "Dataset",
    "aclImdb",
    "train"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "Dataset",
    "aclImdb",
    "test"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "Results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 3. LOAD IMDb DATASET
# ============================================================

def load_reviews(directory):

    texts = []
    labels = []

    for label_name, label in [
        ("pos", 1),
        ("neg", 0)
    ]:

        folder = os.path.join(
            directory,
            label_name
        )

        for filename in sorted(
            os.listdir(folder)
        ):

            if filename.endswith(".txt"):

                file_path = os.path.join(
                    folder,
                    filename
                )

                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    review = file.read()

                texts.append(review)
                labels.append(label)

    return texts, np.array(labels)


print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

train_texts, train_labels = load_reviews(
    TRAIN_DIR
)

test_texts, test_labels = load_reviews(
    TEST_DIR
)

print(
    f"Training samples: {len(train_texts)}"
)

print(
    f"Testing samples : {len(test_texts)}"
)

print(
    f"Training positive: {np.sum(train_labels == 1)}"
)

print(
    f"Training negative: {np.sum(train_labels == 0)}"
)

print(
    f"Testing positive : {np.sum(test_labels == 1)}"
)

print(
    f"Testing negative : {np.sum(test_labels == 0)}"
)


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


print("\nCleaning text...")

train_clean = [
    clean_text(text)
    for text in train_texts
]

test_clean = [
    clean_text(text)
    for text in test_texts
]


# ============================================================
# 5. TF-IDF
# ============================================================

print("\n" + "=" * 60)
print("CREATING TF-IDF FEATURES")
print("=" * 60)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 1)
)

X_train_tfidf = vectorizer.fit_transform(
    train_clean
)

X_test_tfidf = vectorizer.transform(
    test_clean
)

print(
    f"TF-IDF train shape: {X_train_tfidf.shape}"
)

print(
    f"TF-IDF test shape : {X_test_tfidf.shape}"
)


# ============================================================
# 6. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=SEED
)

logistic_model.fit(
    X_train_tfidf,
    train_labels
)

logistic_predictions = logistic_model.predict(
    X_test_tfidf
)

logistic_accuracy = accuracy_score(
    test_labels,
    logistic_predictions
)

logistic_precision = precision_score(
    test_labels,
    logistic_predictions
)

logistic_recall = recall_score(
    test_labels,
    logistic_predictions
)

logistic_f1 = f1_score(
    test_labels,
    logistic_predictions
)

print(
    f"Accuracy : {logistic_accuracy * 100:.2f}%"
)

print(
    f"Precision: {logistic_precision * 100:.2f}%"
)

print(
    f"Recall   : {logistic_recall * 100:.2f}%"
)

print(
    f"F1 Score : {logistic_f1 * 100:.2f}%"
)


# ============================================================
# 7. LINEAR SVM
# ============================================================

print("\n" + "=" * 60)
print("TRAINING LINEAR SVM")
print("=" * 60)

svm_model = LinearSVC(
    random_state=SEED
)

svm_model.fit(
    X_train_tfidf,
    train_labels
)

svm_predictions = svm_model.predict(
    X_test_tfidf
)

svm_accuracy = accuracy_score(
    test_labels,
    svm_predictions
)

svm_precision = precision_score(
    test_labels,
    svm_predictions
)

svm_recall = recall_score(
    test_labels,
    svm_predictions
)

svm_f1 = f1_score(
    test_labels,
    svm_predictions
)

print(
    f"Accuracy : {svm_accuracy * 100:.2f}%"
)

print(
    f"Precision: {svm_precision * 100:.2f}%"
)

print(
    f"Recall   : {svm_recall * 100:.2f}%"
)

print(
    f"F1 Score : {svm_f1 * 100:.2f}%"
)


# ============================================================
# 8. FINAL LSTM CONFIGURATION
# ============================================================

VOCAB_SIZE = 20000
SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DROPOUT_RATE = 0.2
EPOCHS = 5
BATCH_SIZE = 64


print("\n" + "=" * 60)
print("FINAL LSTM CONFIGURATION")
print("=" * 60)

print(
    f"Vocabulary size : {VOCAB_SIZE}"
)

print(
    f"Sequence length : {SEQUENCE_LENGTH}"
)

print(
    f"Embedding dim   : {EMBEDDING_DIM}"
)

print(
    f"LSTM units      : {LSTM_UNITS}"
)

print(
    f"Dropout         : {DROPOUT_RATE}"
)

print(
    f"Optimizer       : Adam"
)

print(
    f"Epochs          : {EPOCHS}"
)

print(
    f"Batch size      : {BATCH_SIZE}"
)

print(
    f"Random seed     : {SEED}"
)


# ============================================================
# 9. STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 60)
print("CREATING BALANCED LSTM TRAIN/VALIDATION SPLIT")
print("=" * 60)

all_indices = np.arange(
    len(train_texts)
)

train_indices, val_indices = train_test_split(
    all_indices,
    test_size=5000,
    random_state=SEED,
    stratify=train_labels
)

lstm_train_texts = [
    train_texts[index]
    for index in train_indices
]

lstm_val_texts = [
    train_texts[index]
    for index in val_indices
]

lstm_train_labels = train_labels[
    train_indices
]

lstm_val_labels = train_labels[
    val_indices
]


print(
    f"LSTM training samples  : "
    f"{len(lstm_train_texts)}"
)

print(
    f"LSTM validation samples: "
    f"{len(lstm_val_texts)}"
)

print(
    f"LSTM train positive    : "
    f"{np.sum(lstm_train_labels == 1)}"
)

print(
    f"LSTM train negative    : "
    f"{np.sum(lstm_train_labels == 0)}"
)

print(
    f"LSTM val positive      : "
    f"{np.sum(lstm_val_labels == 1)}"
)

print(
    f"LSTM val negative      : "
    f"{np.sum(lstm_val_labels == 0)}"
)


# ============================================================
# 10. TOKENIZER
# ============================================================

print("\nCreating tokenizer...")

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    lstm_train_texts
)


# ============================================================
# 11. TEXT TO SEQUENCES
# ============================================================

X_lstm_train = tokenizer.texts_to_sequences(
    lstm_train_texts
)

X_lstm_val = tokenizer.texts_to_sequences(
    lstm_val_texts
)

X_lstm_test = tokenizer.texts_to_sequences(
    test_texts
)


# ============================================================
# 12. PADDING
# ============================================================

X_lstm_train = tf.keras.preprocessing.sequence.pad_sequences(
    X_lstm_train,
    maxlen=SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)

X_lstm_val = tf.keras.preprocessing.sequence.pad_sequences(
    X_lstm_val,
    maxlen=SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)

X_lstm_test = tf.keras.preprocessing.sequence.pad_sequences(
    X_lstm_test,
    maxlen=SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)

print("\nSequence shapes:")

print(
    f"LSTM train shape: {X_lstm_train.shape}"
)

print(
    f"LSTM val shape  : {X_lstm_val.shape}"
)

print(
    f"LSTM test shape : {X_lstm_test.shape}"
)


# ============================================================
# 13. BUILD FINAL LSTM
# ============================================================

print("\n" + "=" * 60)
print("BUILDING FINAL LSTM MODEL")
print("=" * 60)

tf.keras.backend.clear_session()

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    ),

    tf.keras.layers.LSTM(
        LSTM_UNITS
    ),

    tf.keras.layers.Dropout(
        DROPOUT_RATE
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# Explicitly build the model before count_params()
lstm_model.build(
    input_shape=(
        None,
        SEQUENCE_LENGTH
    )
)


lstm_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print(
    f"LSTM parameters: "
    f"{lstm_model.count_params():,}"
)


# ============================================================
# 14. TRAIN FINAL LSTM
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FINAL LSTM")
print("=" * 60)

history = lstm_model.fit(
    X_lstm_train,
    lstm_train_labels,
    validation_data=(
        X_lstm_val,
        lstm_val_labels
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    verbose=1
)


# ============================================================
# 15. GENERATE LSTM TEST PREDICTIONS
# ============================================================

print("\nGenerating LSTM predictions...")

lstm_probabilities = lstm_model.predict(
    X_lstm_test,
    batch_size=BATCH_SIZE,
    verbose=0
)

lstm_predictions = (
    lstm_probabilities >= 0.5
).astype(int).flatten()


# ============================================================
# 16. FINAL LSTM METRICS
# ============================================================

lstm_accuracy = accuracy_score(
    test_labels,
    lstm_predictions
)

lstm_precision = precision_score(
    test_labels,
    lstm_predictions
)

lstm_recall = recall_score(
    test_labels,
    lstm_predictions
)

lstm_f1 = f1_score(
    test_labels,
    lstm_predictions
)

print("\n" + "=" * 60)
print("FINAL LSTM TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy : {lstm_accuracy * 100:.2f}%"
)

print(
    f"Precision: {lstm_precision * 100:.2f}%"
)

print(
    f"Recall   : {lstm_recall * 100:.2f}%"
)

print(
    f"F1 Score : {lstm_f1 * 100:.2f}%"
)


# ============================================================
# 17. CONFUSION MATRIX FUNCTION
# ============================================================

def save_confusion_matrix(
    true_labels,
    predictions,
    model_name,
    filename
):

    cm = confusion_matrix(
        true_labels,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Negative",
            "Positive"
        ]
    )

    display.plot()

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# 18. SAVE CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 60)
print("SAVING CONFUSION MATRICES")
print("=" * 60)

save_confusion_matrix(
    test_labels,
    logistic_predictions,
    "Logistic Regression",
    "confusion_matrix_logistic_regression.png"
)

save_confusion_matrix(
    test_labels,
    svm_predictions,
    "Linear SVM",
    "confusion_matrix_linear_svm.png"
)

save_confusion_matrix(
    test_labels,
    lstm_predictions,
    "Final LSTM",
    "confusion_matrix_final_lstm.png"
)


# ============================================================
# 19. SAVE LSTM ACCURACY CURVE
# ============================================================

print("\nSaving LSTM accuracy curve...")

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Final LSTM Training and Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()

accuracy_path = os.path.join(
    RESULTS_DIR,
    "lstm_training_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {accuracy_path}"
)


# ============================================================
# 20. SAVE LSTM LOSS CURVE
# ============================================================

print("\nSaving LSTM loss curve...")

plt.figure()

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Final LSTM Training and Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

loss_path = os.path.join(
    RESULTS_DIR,
    "lstm_training_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {loss_path}"
)


# ============================================================
# 21. SAVE MISCLASSIFIED LSTM REVIEWS
# ============================================================

print("\nSaving misclassified LSTM examples...")

misclassified_indices = np.where(
    lstm_predictions != test_labels
)[0]

misclassified_path = os.path.join(
    RESULTS_DIR,
    "misclassified_examples.txt"
)

with open(
    misclassified_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Task 2 - Final LSTM Misclassified Examples\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Total misclassified reviews: "
        f"{len(misclassified_indices)}\n\n"
    )

    for number, index in enumerate(
        misclassified_indices[:20],
        start=1
    ):

        actual_label = (
            "Positive"
            if test_labels[index] == 1
            else "Negative"
        )

        predicted_label = (
            "Positive"
            if lstm_predictions[index] == 1
            else "Negative"
        )

        file.write(
            f"Example {number}\n"
        )

        file.write(
            f"Actual sentiment    : "
            f"{actual_label}\n"
        )

        file.write(
            f"Predicted sentiment : "
            f"{predicted_label}\n"
        )

        file.write(
            "Review:\n"
        )

        file.write(
            test_texts[index]
        )

        file.write(
            "\n\n"
            + "-" * 70
            + "\n\n"
        )


print(
    f"Saved: {misclassified_path}"
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL ANALYSIS COMPLETE")
print("=" * 60)

print("\nGenerated files:")

print(
    "1. confusion_matrix_logistic_regression.png"
)

print(
    "2. confusion_matrix_linear_svm.png"
)

print(
    "3. confusion_matrix_final_lstm.png"
)

print(
    "4. lstm_training_accuracy.png"
)

print(
    "5. lstm_training_loss.png"
)

print(
    "6. misclassified_examples.txt"
)

print("\nFinal LSTM configuration:")

print(
    "Vocabulary size = 20,000"
)

print(
    "Sequence length = 200"
)

print(
    "Embedding dimension = 128"
)

print(
    "LSTM units = 64"
)

print(
    "Dropout = 0.2"
)

print(
    "Optimizer = Adam"
)

print(
    "Epochs = 5"
)

print(
    "Batch size = 64"
)

print(
    "Random seed = 42"
)

print("=" * 60)