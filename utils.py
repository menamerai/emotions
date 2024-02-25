import pickle, os, torch
from datasets import Dataset
from torchtext.vocab import Vocab


def load_and_split_data():
    if not os.path.exists("./data/emotions.pkl"):
        raise FileNotFoundError(
            "Dataset not found. Run setup.py to download the dataset."
        )
    with open("./data/emotions.pkl", "rb") as f:
        dataset = pickle.load(f)
    return dataset["train"], dataset["validation"], dataset["test"]


def tokenize_example(example: Dataset, tokenizer, max_length) -> dict[str, any]:
    tokens = tokenizer(example["text"])[:max_length]
    length = len(tokens)
    return {"tokens": tokens, "length": length}


def numericalize_example(example, vocab: Vocab):
    ids = vocab.lookup_indices(example["tokens"])
    return {"ids": ids}


def get_collate_fn(pad_index, include_length=False):
    def collate_fn(batch):
        batch_ids = [example["ids"] for example in batch]
        batch_ids = torch.nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        if include_length:
            batch_length = [example["length"] for example in batch]
            batch_length = torch.stack(batch_length)
        batch_label = [example["label"] for example in batch]
        batch_label = torch.stack(batch_label)
        if include_length:
            batch = {"ids": batch_ids, "label": batch_label, "length": batch_length}
        else:
            batch = {"ids": batch_ids, "label": batch_label}
        return batch

    return collate_fn


def get_data_loader(data, batch_size, pad_index, shuffle=True, include_length=False):
    collate_fn = get_collate_fn(pad_index, include_length=include_length)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle
    )
    return data_loader

def tokenize_and_numericalize_example(example, tokenizer):
    ids = tokenizer(example["text"], truncation=True)["input_ids"]
    return {"ids": ids}