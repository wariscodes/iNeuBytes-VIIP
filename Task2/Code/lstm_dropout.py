import os
import re
import string
import time

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# --------------------------------------------------
# 2. Dataset path
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Dataset",
    "aclImdb"
)


# --------------------------------------------------
# 3. Text cleaning
# --------------------------------------------------

def clean_text(text):
    text = text.lower()

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# --------------------------------------------------
# 4. Load reviews
# --------------------------------------------------

def load_reviews(folder_path):

    reviews = []

    for filename in os.listdir(folder_path):

        if filename.endswith(".txt"):

            file_path = os.path.join(
                folder_path,
                filename
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                review = file.read()

            reviews.append(
                clean_text(review)
            )

    return reviews


# --------------------------------------------------
# 5. Dataset folders
# --------------------------------------------------

train_positive_path = os.path.join(
    DATASET_PATH,
    "train",
    "pos"
)

train_negative_path = os.path.join(
    DATASET_PATH,
    "train",
    "neg"
)

test_positive_path = os.path.join(
    DATASET_PATH,
    "test",
    "pos"
)

test_negative_path = os.path.join(
    DATASET_PATH,
    "test",
    "neg"
)


# --------------------------------------------------
# 6. Load data
# --------------------------------------------------

train_positive = load_reviews(
    train_positive_path
)

train_negative = load_reviews(
    train_negative_path
)

test_positive = load_reviews(
    test_positive_path
)

test_negative = load_reviews(
    test_negative_path
)


# --------------------------------------------------
# 7. Combine reviews and labels
# --------------------------------------------------

train_reviews = (
    train_positive +
    train_negative
)

train_labels = np.array(
    [1] * len(train_positive) +
    [0] * len(train_negative)
)

test_reviews = (
    test_positive +
    test_negative
)

test_labels = np.array(
    [1] * len(test_positive) +
    [0] * len(test_negative)
)


# --------------------------------------------------
# 8. Fixed validation split
# --------------------------------------------------

validation_size = 5000

validation_reviews = train_reviews[
    -validation_size:
]

validation_labels = train_labels[
    -validation_size:
]

training_reviews = train_reviews[
    :-validation_size
]

training_labels = train_labels[
    :-validation_size
]


print("IMDb LSTM Dropout Experiment")
print("-----------------------------")

print(
    "Training reviews:",
    len(training_reviews)
)

print(
    "Validation reviews:",
    len(validation_reviews)
)

print(
    "Testing reviews:",
    len(test_reviews)
)


# --------------------------------------------------
# 9. Configuration
# --------------------------------------------------

MAX_VOCAB_SIZE = 20000

SEQUENCE_LENGTH = 200

EMBEDDING_DIM = 128

LSTM_UNITS = 64

DROPOUT_RATE = 0.5


# --------------------------------------------------
# 10. Tokenization
# --------------------------------------------------

tokenizer = Tokenizer(
    num_words=MAX_VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    training_reviews
)


# --------------------------------------------------
# 11. Convert text to sequences
# --------------------------------------------------

training_sequences = tokenizer.texts_to_sequences(
    training_reviews
)

validation_sequences = tokenizer.texts_to_sequences(
    validation_reviews
)

test_sequences = tokenizer.texts_to_sequences(
    test_reviews
)


# --------------------------------------------------
# 12. Padding
# --------------------------------------------------

training_padded = pad_sequences(
    training_sequences,
    maxlen=SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)

validation_padded = pad_sequences(
    validation_sequences,
    maxlen=SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)

test_padded = pad_sequences(
    test_sequences,
    maxlen=SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)


print("\nSequence Data")
print("-------------")

print(
    "Training shape:",
    training_padded.shape
)

print(
    "Validation shape:",
    validation_padded.shape
)

print(
    "Testing shape:",
    test_padded.shape
)


# --------------------------------------------------
# 13. Build model
# --------------------------------------------------

model = Sequential([

    Embedding(
        input_dim=MAX_VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),

    LSTM(
        LSTM_UNITS
    ),

    Dropout(
        DROPOUT_RATE
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# --------------------------------------------------
# 14. Compile
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# 15. Build explicitly
# --------------------------------------------------

model.build(
    input_shape=(
        None,
        SEQUENCE_LENGTH
    )
)


# --------------------------------------------------
# 16. Model summary
# --------------------------------------------------

print("\nModel Summary")
print("-------------")

model.summary()


# --------------------------------------------------
# 17. Training
# --------------------------------------------------

print("\nTraining LSTM...")
print("----------------")

start_time = time.time()

history = model.fit(

    training_padded,

    training_labels,

    validation_data=(
        validation_padded,
        validation_labels
    ),

    epochs=5,

    batch_size=64,

    verbose=1
)

training_time = (
    time.time() -
    start_time
)


# --------------------------------------------------
# 18. Test prediction
# --------------------------------------------------

print("\nGenerating test predictions...")

test_probabilities = model.predict(
    test_padded,
    batch_size=64,
    verbose=1
)

test_predictions = (
    test_probabilities >= 0.5
).astype(int).flatten()


# --------------------------------------------------
# 19. Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    test_labels,
    test_predictions
)

precision = precision_score(
    test_labels,
    test_predictions
)

recall = recall_score(
    test_labels,
    test_predictions
)

f1 = f1_score(
    test_labels,
    test_predictions
)


# --------------------------------------------------
# 20. Train-validation gap
# --------------------------------------------------

final_train_accuracy = (
    history.history["accuracy"][-1]
)

final_validation_accuracy = (
    history.history["val_accuracy"][-1]
)

train_validation_gap = (
    final_train_accuracy -
    final_validation_accuracy
) * 100


# --------------------------------------------------
# 21. Final results
# --------------------------------------------------

print("\nLSTM Dropout Experiment Results")
print("--------------------------------")

print(
    f"Dropout Rate: "
    f"{DROPOUT_RATE}"
)

print(
    f"Final Train Accuracy: "
    f"{final_train_accuracy * 100:.2f}%"
)

print(
    f"Final Validation Accuracy: "
    f"{final_validation_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision: "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall: "
    f"{recall * 100:.2f}%"
)

print(
    f"F1 Score: "
    f"{f1 * 100:.2f}%"
)

print(
    f"Train-Validation Gap: "
    f"{train_validation_gap:.2f} percentage points"
)

print(
    f"Training Time: "
    f"{training_time:.2f} seconds"
)