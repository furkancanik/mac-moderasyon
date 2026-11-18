"""Gün 3: kendi word2vec'imizi eğitir, üstüne 1D CNN + feedforward ağ kurar (Alan A + B).

Adımlar:
  1) Eğitim tweet'leri normalize edilir, gensim ile skip-gram word2vec eğitilir (100 boyut).
     (Sadece eğitim seti kullanılır; dev ve test görülmez.)
  2) Sözlük: word2vec'te olan kelimeler + <pad>, <unk>. Embedding katmanı bu vektörlerle başlatılır.
  3) TextCNN, SGD (momentum 0.9) ve geri yayılım (backpropagation) ile eğitilir.
     Sınıf dengesizliği için BCE kaybında pos_weight kullanılır.
     Her epoch sonunda dev macro-F1 ölçülür, en iyi model saklanır (early stopping).
  4) En iyi model test setinde ölçülür; gecikme CPU'da tek tek mesajlarla ölçülür.

Çalıştır:  python -m src.train_w2v_cnn
Çıktı:     models/word2vec.kv, models/w2v_cnn.pt, results/w2v_cnn.json
"""
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.metrics import f1_score

from src.evaluate import ROOT, benchmark, metrics, print_errors, save_result, size_mb
from src.w2v_cnn import (MAX_LEN, PAD, UNK, TextCNN, W2VCNNClassifier, collate, encode,
                         tokenize)

DATA = ROOT / "data"
MODELS = ROOT / "models"
SEED = 42
EMB_DIM = 100
EPOCHS = 15
BATCH = 64
LR = 0.05


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_all(model, encoded, device, bs: int = 256) -> np.ndarray:
    model.eval()
    probs = []
    for i in range(0, len(encoded), bs):
        ids = collate(encoded[i:i + bs]).to(device)
        probs.append(torch.sigmoid(model(ids)).cpu().numpy())
    return np.concatenate(probs)


def main() -> None:
    set_seed(SEED)
    MODELS.mkdir(exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    dev = pd.read_csv(DATA / "dev.csv")
    test = pd.read_csv(DATA / "test.csv")
    for df in (train, dev, test):
        df["text"] = df["text"].astype(str)

    # ---------- 1) word2vec ----------
    print("1) word2vec eğitiliyor (skip-gram, sadece eğitim seti)...")
    sentences = [tokenize(t) for t in train["text"]]
    t0 = time.time()
    w2v = Word2Vec(sentences, vector_size=EMB_DIM, window=5, min_count=2, sg=1,
                   negative=10, epochs=15, workers=4, seed=SEED)
    w2v.wv.save(str(MODELS / "word2vec.kv"))
    print(f"   sözlük: {len(w2v.wv)} kelime, süre: {time.time() - t0:.1f} sn")
    for probe in ["hakem", "maç", "salak", "şerefsiz", "galatasaray"]:
        if probe in w2v.wv:
            sims = ", ".join(w for w, _ in w2v.wv.most_similar(probe, topn=5))
            print(f"   '{probe}' kelimesine en yakın: {sims}")

    # ---------- 2) sözlük ve embedding matrisi ----------
    itos = ["<pad>", "<unk>"] + list(w2v.wv.index_to_key)
    stoi = {w: i for i, w in enumerate(itos)}
    emb = np.zeros((len(itos), EMB_DIM), dtype=np.float32)
    emb[UNK] = np.random.normal(0, 0.1, EMB_DIM)
    emb[2:] = w2v.wv.vectors

    def enc(df):
        return [encode(t, stoi, MAX_LEN) for t in df["text"]]

    Xtr, Xdev, Xte = enc(train), enc(dev), enc(test)
    ytr = train["label"].to_numpy()
    unk_rate = np.mean([i == UNK for x in Xte for i in x if i != PAD])
    print(f"   test setinde bilinmeyen kelime oranı: %{100 * unk_rate:.1f}")

    # ---------- 3) TextCNN eğitimi: SGD + backpropagation ----------
    device = pick_device()
    print(f"2) TextCNN eğitiliyor, cihaz: {device}")
    config = {"emb_dim": EMB_DIM, "n_filters": 128, "kernel_sizes": [2, 3, 4],
              "hidden": 64, "dropout": 0.5}
    model = TextCNN(len(itos), **config)
    model.embedding.weight.data.copy_(torch.from_numpy(emb))
    model.to(device)

    # MPS float64 desteklemiyor, bu yüzden float32
    pos_weight = torch.tensor([float((ytr == 0).sum() / (ytr == 1).sum())],
                              dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-5)

    best_f1, best_state, history = -1.0, None, []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = np.random.permutation(len(Xtr))
        total = 0.0
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            ids = collate([Xtr[j] for j in idx]).to(device)
            y = torch.tensor(ytr[idx], dtype=torch.float32, device=device)
            opt.zero_grad()
            loss = loss_fn(model(ids), y)
            loss.backward()                       # geri yayılım
            opt.step()                            # SGD güncellemesi
            total += loss.item() * len(idx)
        dev_pred = (predict_all(model, Xdev, device) >= 0.5).astype(int)
        f1 = f1_score(dev["label"], dev_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": round(total / len(Xtr), 4),
                        "dev_macro_f1": round(f1, 4)})
        flag = ""
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  <- en iyi"
        print(f"   epoch {epoch:>2}  loss {total / len(Xtr):.4f}  dev macro-F1 {f1:.4f}{flag}")
    train_sec = round(time.time() - t0, 1)

    path = MODELS / "w2v_cnn.pt"
    torch.save({"state_dict": best_state, "itos": itos, "config": config}, path)

    # ---------- 4) Test: CPU üzerinde ----------
    clf = W2VCNNClassifier(path)
    y_pred = [clf.predict(t) for t in test["text"]]
    m = metrics(test["label"], y_pred)
    print(f"3) TEST  macro-F1 = {m['macro_f1']}  accuracy = {m['accuracy']}  "
          f"hakaret recall = {m['offensive_recall']}")
    lat = benchmark(clf.predict, test["text"])
    print(f"   Gecikme (CPU): ortalama {lat['latency_ms_mean']} ms, p95 {lat['latency_ms_p95']} ms")
    print(f"   Boyut: {size_mb(path)} MB")
    errs = print_errors(test["text"], test["label"], y_pred)
    save_result("w2v_cnn", {**m, **lat, "size_mb": size_mb(path),
                            "vocab_size": len(itos), "best_dev_macro_f1": round(best_f1, 4),
                            "train_device": str(device), "train_seconds": train_sec,
                            "history": history, "test_size": len(test), "errors": errs})


if __name__ == "__main__":
    main()
