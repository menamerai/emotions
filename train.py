import torch, torchtext, wandb, argparse

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from collections import defaultdict
from utils import *
from model import *


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm(data_loader, total=len(data_loader), desc="Training"):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)

        preds = model(ids)
        loss = criterion(preds, label)
        acc = get_accuracy(preds, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_accs.append(acc.item())

    return np.mean(epoch_losses), np.mean(epoch_accs)


def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Evaluating"):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)

            preds = model(ids)
            loss = criterion(preds, label)
            acc = get_accuracy(preds, label)

            epoch_losses.append(loss.item())
            epoch_accs.append(acc.item())

    return np.mean(epoch_losses), np.mean(epoch_accs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--logging", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--max_length", type=int, default=300)
    parser.add_argument("--min_vocab_freq", type=int, default=1)
    parser.add_argument("--model", type=str, default="NBoW")

    args = parser.parse_args()

    if args.logging:
        wandb.login()
        wandb.init(
            project="emotions",
            config={
                "batch_size": args.batch_size,
                "embedding_dim": args.embedding_dim,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "model": args.model,
            },
        )

    train_data, val_data, _ = load_and_split_data()
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

    train_data = train_data.map(
        tokenize_example,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
    )

    val_data = val_data.map(
        tokenize_example,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
    )

    vocab = torchtext.vocab.build_vocab_from_iterator(
        train_data["tokens"], min_freq=args.min_vocab_freq, specials=["<unk>", "<pad>"]
    )

    unk_idx = vocab["<unk>"]
    pad_idx = vocab["<pad>"]

    vocab.set_default_index(unk_idx)

    train_data = train_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
    val_data = val_data.map(numericalize_example, fn_kwargs={"vocab": vocab})

    train_data.set_format(type="torch", columns=["ids", "label"])
    val_data.set_format(type="torch", columns=["ids", "label"])

    train_loader = get_data_loader(train_data, args.batch_size, pad_idx)
    val_loader = get_data_loader(val_data, args.batch_size, pad_idx, shuffle=False)

    vocab_size = len(vocab)
    num_classes = len(train_data.unique("label"))

    if args.model == "NBoW":
        model = NBoW(vocab_size, args.embedding_dim, num_classes, pad_idx).to(device)

    # TODO: add more models here

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Start training...")
    best_val_loss = float("inf")

    metrics = defaultdict(list)

    for epoch in range(args.epochs):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        val_loss, val_acc = evaluate(val_loader, model, criterion, device)

        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        if args.logging:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "./models/nbow_model.pth")

        print(f"Epoch: {epoch+1}")
        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

    if args.logging:
        wandb.finish()
