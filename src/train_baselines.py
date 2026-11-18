"""Gün 2: iki klasik baseline'ı eğitir ve test setinde ölçer (Alan B).

  1) rule_lexicon : regex normalizasyon + edit distance ile küfür listesi eşleştirme (eğitim yok)
  2) tfidf_lr     : regex normalizasyon + tf-idf (kelime 1-2 gram + karakter 2-5 gram)
                    + logistic regression. C değeri dev setinde macro-F1'e göre seçilir.

Çalıştır:  python -m src.train_baselines
Çıktı:     models/tfidf_lr.joblib, results/rule_lexicon.json, results/tfidf_lr.json
"""
import time

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluate import (ROOT, benchmark, metrics, print_errors, save_result, size_mb)
from src.lexicon import LEXICON_PATH, LexiconMatcher
from src.normalize import normalize

DATA = ROOT / "data"
MODELS = ROOT / "models"


def build_tfidf_lr(C: float) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 sublinear_tf=True, max_features=200_000)),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    return Pipeline([("tfidf", features), ("clf", clf)])


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    dev = pd.read_csv(DATA / "dev.csv")
    test = pd.read_csv(DATA / "test.csv")
    for df in (train, dev, test):
        df["text"] = df["text"].astype(str)

    # ---------- 1) Kural tabanlı: küfür listesi + edit distance ----------
    print("=" * 70 + "\n1) Kural tabanlı baseline (küfür listesi + edit distance)")
    matcher = LexiconMatcher()
    print(f"  Listede {len(matcher.lexicon)} kök var")
    y_pred = [matcher.predict(t) for t in test["text"]]
    m = metrics(test["label"], y_pred)
    print(f"  TEST  macro-F1 = {m['macro_f1']}  accuracy = {m['accuracy']}")
    lat = benchmark(matcher.predict, test["text"])
    print(f"  Gecikme: ortalama {lat['latency_ms_mean']} ms, p95 {lat['latency_ms_p95']} ms")
    errs = print_errors(test["text"], test["label"], y_pred)
    save_result("rule_lexicon", {**m, **lat, "size_mb": size_mb(LEXICON_PATH),
                                 "test_size": len(test), "errors": errs})

    # ---------- 2) tf-idf + logistic regression ----------
    print("=" * 70 + "\n2) tf-idf + logistic regression")
    Xtr = [normalize(t) for t in train["text"]]
    Xdev = [normalize(t) for t in dev["text"]]

    best_C, best_f1 = None, -1.0
    for C in [0.5, 1.0, 2.0, 4.0, 8.0]:
        model = build_tfidf_lr(C).fit(Xtr, train["label"])
        f1 = f1_score(dev["label"], model.predict(Xdev), average="macro")
        print(f"  C={C:<4} dev macro-F1 = {f1:.4f}")
        if f1 > best_f1:
            best_C, best_f1 = C, f1
    print(f"  Seçilen C = {best_C}")

    t0 = time.time()
    model = build_tfidf_lr(best_C).fit(Xtr, train["label"])
    train_sec = round(time.time() - t0, 1)
    path = MODELS / "tfidf_lr.joblib"
    joblib.dump(model, path)

    def predict_one(text: str) -> int:
        return int(model.predict([normalize(text)])[0])

    y_pred = model.predict([normalize(t) for t in test["text"]])
    m = metrics(test["label"], y_pred)
    print(f"  TEST  macro-F1 = {m['macro_f1']}  accuracy = {m['accuracy']}")
    lat = benchmark(predict_one, test["text"])
    print(f"  Gecikme: ortalama {lat['latency_ms_mean']} ms, p95 {lat['latency_ms_p95']} ms")
    print(f"  Boyut: {size_mb(path)} MB")
    errs = print_errors(test["text"], test["label"], y_pred)
    save_result("tfidf_lr", {**m, **lat, "size_mb": size_mb(path), "best_C": best_C,
                             "dev_macro_f1": round(best_f1, 4), "train_seconds": train_sec,
                             "test_size": len(test), "errors": errs})


if __name__ == "__main__":
    main()
