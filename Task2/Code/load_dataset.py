import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "aclImdb")

train_path = os.path.join(DATASET_PATH, "train")
test_path = os.path.join(DATASET_PATH, "test")

train_positive = os.listdir(os.path.join(train_path, "pos"))
train_negative = os.listdir(os.path.join(train_path, "neg"))

test_positive = os.listdir(os.path.join(test_path, "pos"))
test_negative = os.listdir(os.path.join(test_path, "neg"))

print("IMDb Dataset Check")
print("------------------")

print("Training positive reviews:", len(train_positive))
print("Training negative reviews:", len(train_negative))

print("Testing positive reviews:", len(test_positive))
print("Testing negative reviews:", len(test_negative))