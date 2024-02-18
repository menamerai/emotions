import pickle, os


def load_and_split_data():
    if not os.path.exists("./data/emotions.pkl"):
        raise FileNotFoundError(
            "Dataset not found. Run setup.py to download the dataset."
        )
    with open("./data/emotions.pkl", "rb") as f:
        dataset = pickle.load(f)
    return dataset["train"], dataset["validation"], dataset["test"]
