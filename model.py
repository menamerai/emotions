import torch.nn as nn


class NBoW(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_index):
        super(NBoW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        preds = self.fc(x)
        return preds
