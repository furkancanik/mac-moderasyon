"""Kendi word2vec gömmelerimiz üzerinde çalışan 1D CNN + feedforward sınıflandırıcı (Alan A).

Mimari (Kim, 2014 tarzı metin CNN'i):
  token id'leri -> Embedding (word2vec ile başlatılır, eğitimde güncellenir)
  -> Dropout -> Conv1d (pencere boyları 2, 3, 4; her biri 128 filtre) -> ReLU -> global max pooling
  -> birleştir (384) -> Dropout -> Linear(384, 64) -> ReLU -> Linear(64, 1) -> sigmoid

Bu dosya hem eğitim betiği (train_w2v_cnn.py) hem de uygulama tarafından kullanılır.
"""
from pathlib import Path

import torch
import torch.nn as nn

from src.normalize import normalize

PAD, UNK = 0, 1
MAX_LEN = 64


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 100, n_filters: int = 128,
                 kernel_sizes=(2, 3, 4), hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.emb_dropout = nn.Dropout(0.3)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, n_filters, k) for k in kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(n_filters * len(kernel_sizes), hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:          # ids: (B, L)
        x = self.emb_dropout(self.embedding(ids)).transpose(1, 2)    # (B, E, L)
        pooled = [torch.relu(conv(x)).amax(dim=2) for conv in self.convs]
        h = self.dropout(torch.cat(pooled, dim=1))
        h = torch.relu(self.fc1(h))
        return self.fc2(self.dropout(h)).squeeze(1)                  # logit, (B,)


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def encode(text: str, stoi: dict, max_len: int = MAX_LEN) -> list[int]:
    ids = [stoi.get(t, UNK) for t in tokenize(text)][:max_len]
    # En büyük konvolüsyon penceresi 4; çok kısa mesajlar için en az 4 uzunluk
    return ids + [PAD] * max(0, 4 - len(ids))


def collate(batch_ids: list[list[int]]) -> torch.Tensor:
    L = max(len(x) for x in batch_ids)
    return torch.tensor([x + [PAD] * (L - len(x)) for x in batch_ids], dtype=torch.long)


class W2VCNNClassifier:
    """Kaydedilmiş modeli yükleyip tek mesaj için olasılık döndürür (CPU)."""

    def __init__(self, path: str | Path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.itos = ckpt["itos"]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.model = TextCNN(len(self.itos), **ckpt["config"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict_proba(self, text: str) -> float:
        ids = collate([encode(text, self.stoi)])
        return float(torch.sigmoid(self.model(ids))[0])

    def predict(self, text: str) -> int:
        return int(self.predict_proba(text) >= 0.5)
