"""Gün 4a: BERTurk'ü hakaret tespiti için fine-tune eder (Alan C).

Model: dbmdz/bert-base-turkish-cased (BERTurk) + 2 sınıflı sınıflandırma başlığı
Eğitim: AdamW, öğrenme oranı 2e-5, %10 ısınma + doğrusal azalma, 3 epoch, batch 32,
        en fazla 64 alt-kelime (WordPiece token). Sınıf dengesizliği için ağırlıklı kayıp.
        Her epoch sonunda dev macro-F1 ölçülür, en iyi model kaydedilir.
Cihaz:  Apple GPU (MPS) > NVIDIA (CUDA) > CPU

Çalıştır:  python -m src.train_bert
Çıktı:     models/berturk-offensive/  (model + tokenizer), results/berturk_fp32.json
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from src.bert_model import MAX_LEN, BertClassifier, bert_preprocess  # noqa: E402
from src.evaluate import (ROOT, benchmark, metrics, print_errors, save_result,  # noqa: E402
                          size_mb)

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
OUT_DIR = ROOT / "models" / "berturk-offensive"
DATA = ROOT / "data"
SEED = 42
EPOCHS = 3
BATCH = 32
LR = 2e-5
WARMUP = 0.1


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train = pd.read_csv(DATA / "train.csv")
    dev = pd.read_csv(DATA / "dev.csv")
    test = pd.read_csv(DATA / "test.csv")
    for df in (train, dev, test):
        df["text"] = df["text"].astype(str)

    device = pick_device()
    print(f"Cihaz: {device}")
    print(f"{MODEL_NAME} indiriliyor (ilk seferde ~450 MB)...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        id2label={0: "temiz", 1: "hakaret"}, label2id={"temiz": 0, "hakaret": 1},
    ).to(device)

    Xtr = [bert_preprocess(t) for t in train["text"]]
    ytr = train["label"].to_numpy()
    Xdev = [bert_preprocess(t) for t in dev["text"]]

    # Ağırlıklı kayıp: hakaret sınıfına sqrt(temiz/hakaret) kadar ağırlık (~2)
    w_pos = math.sqrt((ytr == 0).sum() / (ytr == 1).sum())
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, w_pos], dtype=torch.float32, device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(Xtr) / BATCH)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(WARMUP * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    @torch.no_grad()
    def dev_f1() -> float:
        model.eval()
        preds = []
        for i in range(0, len(Xdev), 128):
            enc = tok(Xdev[i:i + 128], padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt").to(device)
            preds.extend(model(**enc).logits.argmax(-1).cpu().tolist())
        return f1_score(dev["label"], preds, average="macro")

    best_f1, history = -1.0, []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = np.random.permutation(len(Xtr))
        running = 0.0
        bar = tqdm(range(0, len(order), BATCH), desc=f"epoch {epoch}/{EPOCHS}")
        for step, i in enumerate(bar, 1):
            idx = order[i:i + BATCH]
            enc = tok([Xtr[j] for j in idx], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt").to(device)
            y = torch.tensor(ytr[idx], dtype=torch.long, device=device)
            loss = loss_fn(model(**enc).logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            if step % 20 == 0:
                bar.set_postfix(loss=f"{running / step:.4f}")
        f1 = dev_f1()
        history.append({"epoch": epoch, "train_loss": round(running / steps_per_epoch, 4),
                        "dev_macro_f1": round(f1, 4)})
        flag = ""
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(OUT_DIR)
            tok.save_pretrained(OUT_DIR)
            flag = "  <- en iyi, kaydedildi"
        print(f"epoch {epoch}: train loss {running / steps_per_epoch:.4f}  "
              f"dev macro-F1 {f1:.4f}{flag}")
    train_min = round((time.time() - t0) / 60, 1)
    print(f"Eğitim süresi: {train_min} dakika")

    # ---------- Test ----------
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    print("Test seti değerlendiriliyor...")
    fast = BertClassifier(OUT_DIR, device=str(device))
    y_pred = (fast.predict_proba_batch(test["text"]) >= 0.5).astype(int)
    m = metrics(test["label"], y_pred)
    print(f"TEST  macro-F1 = {m['macro_f1']}  accuracy = {m['accuracy']}  "
          f"hakaret recall = {m['offensive_recall']}")
    del fast

    print("CPU'da gecikme ölçülüyor (200 mesaj, tek tek)...")
    cpu = BertClassifier(OUT_DIR, device="cpu")
    lat = benchmark(cpu.predict, test["text"])
    print(f"Gecikme (CPU, FP32): ortalama {lat['latency_ms_mean']} ms, "
          f"p95 {lat['latency_ms_p95']} ms")
    sz = size_mb(OUT_DIR)
    print(f"Boyut: {sz} MB")
    errs = print_errors(test["text"], test["label"], y_pred)
    save_result("berturk_fp32", {**m, **lat, "size_mb": sz, "precision": "FP32",
                                 "runtime": "PyTorch", "base_model": MODEL_NAME,
                                 "best_dev_macro_f1": round(best_f1, 4),
                                 "train_device": str(device), "train_minutes": train_min,
                                 "history": history, "test_size": len(test), "errors": errs})


if __name__ == "__main__":
    main()
