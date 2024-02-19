import torch
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


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_classes,
        pad_index,
        hidden_dim,
        num_layers,
        bidirectional,
        dropout_rate,
    ):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, length):
        x = self.dropout(self.embedding(x))
        x = nn.utils.rnn.pack_padded_sequence(x, length, enforce_sorted=False, batch_first=True)
        x, (hidden, cell) = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])

        preds = self.fc(hidden)
        return preds