import os
import re
import string
import time
import random

import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense


# ============================================================
# 1. Random Seed
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "Dataset", "aclImdb", "train")
TEST_DIR = os.path.join(BASE_DIR, "Dataset", "aclImdb", "test")


# ============================================================
# 3. Text Cleaning
# ============================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# 4. Load Reviews
# ============================================================

def load_reviews(directory):
    texts = []
    labels = []

    for label_name, label_value in [("neg", 0), ("pos", 1)]:
        folder = os.path.join(directory, label_name)

        for filename in sorted(os.listdir(folder)):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder, filename)

                with open(filepath, "r", encoding="utf-8") as file:
                    text = file.read()

                texts.append(clean_text(text))
                labels.append(label_value)

    return texts, np.array(labels)


print("Loading dataset...")

train_texts, train_labels = load_reviews(TRAIN_DIR)
test_texts, test_labels = load_reviews(TEST_DIR)

print(f"Total training reviews: {len(train_texts)}")
print(f"Total testing reviews: {len(test_texts)}")


# ============================================================
# 5. Train / Validation Split
# ============================================================

indices = np.arange(len(train_texts))

rng = np.random.default_rng(SEED)
rng.shuffle(indices)

train_indices = indices[:20000]
val_indices = indices[20000:]

train_texts_split = [train_texts[i] for i in train_indices]
train_labels_split = train_labels[train_indices]

val_texts = [train_texts[i] for i in val_indices]
val_labels = train_labels[val_indices]


# ============================================================
# 6. Tokenization
# ============================================================

MAX_VOCAB_SIZE = 20000
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128


tokenizer = Tokenizer(
    num_words=MAX_VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(train_texts_split)


train_sequences = tokenizer.texts_to_sequences(train_texts_split)
val_sequences = tokenizer.texts_to_sequences(val_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)


X_train = pad_sequences(
    train_sequences,
    maxlen=SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)

X_val = pad_sequences(
    val_sequences,
    maxlen=SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    test_sequences,
    maxlen=SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)


print(f"Vocabulary size: {MAX_VOCAB_SIZE}")
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Testing data shape: {X_test.shape}")


# ============================================================
# 7. Build LSTM Model
# ============================================================

model = Sequential([
    Embedding(
        input_dim=MAX_VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    ),

    LSTM(64),

    Dropout(0.2),

    Dense(1, activation="sigmoid")
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print("\nModel Summary:")
model.summary()


# ============================================================
# 8. Train Model
# ============================================================

print("\nStarting training...")

start_time = time.time()

history = model.fit(
    X_train,
    train_labels_split,
    validation_data=(X_val, val_labels),
    epochs=5,
    batch_size=64,
    verbose=1
)

training_time = time.time() - start_time


# ============================================================
# 9. Training Metrics
# ============================================================

train_loss, train_accuracy = model.evaluate(
    X_train,
    train_labels_split,
    verbose=0
)

val_loss, val_accuracy = model.evaluate(
    X_val,
    val_labels,
    verbose=0
)


# ============================================================
# 10. Test Prediction
# ============================================================

test_probabilities = model.predict(
    X_test,
    batch_size=64,
    verbose=0
)

test_predictions = (test_probabilities >= 0.5).astype(int).flatten()


# ============================================================
# 11. Test Metrics
# ============================================================

test_accuracy = accuracy_score(
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


train_val_gap = train_accuracy - val_accuracy


# ============================================================
# 12. Final Results
# ============================================================

print("\n" + "=" * 60)
print("LSTM SEQUENCE LENGTH EXPERIMENT RESULTS")
print("=" * 60)

print(f"Sequence Length: {SEQUENCE_LENGTH}")
print(f"Vocabulary Size: {MAX_VOCAB_SIZE}")
print(f"Embedding Dimension: {EMBEDDING_DIM}")
print(f"LSTM Units: 64")
print(f"Dropout: 0.2")
print(f"Epochs: 5")
print(f"Batch Size: 64")
print(f"Random Seed: {SEED}")

print("\nMetrics:")
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"Train-Validation Gap: {train_val_gap * 100:.2f} percentage points")
print(f"Training Time: {training_time:.2f} seconds")

print("=" * 60)