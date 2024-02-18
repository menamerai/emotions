import subprocess, datasets, pickle

if __name__ == "__main__":
    subprocess.run("pip install -r requirements.txt", shell=True)

    # pull dataset from huggingface and save it as pickle
    dataset = datasets.load_dataset("dair-ai/emotion")
    with open("./data/emotions.pkl", "wb") as f:
        pickle.dump(dataset, f)
