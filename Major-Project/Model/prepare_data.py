from datasets import load_from_disk


DATASET_PATH = "Major-Project/Dataset/air_dialogue"


def prepare_data():
    dataset = load_from_disk(DATASET_PATH)

    texts = []
    labels = []

    for row in dataset:
        dialogue = " ".join(row["dialogue"])
        goal = row["intent"]["goal"]

        texts.append(dialogue)
        labels.append(goal)

    print("Total samples:", len(texts))
    print("First input:")
    print(texts[0])

    print("\nFirst label:")
    print(labels[0])

    print("\nUnique labels:")
    print(sorted(set(labels)))


if __name__ == "__main__":
    prepare_data()