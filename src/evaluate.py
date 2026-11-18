"""Bütün modeller için ortak değerlendirme araçları (R3, R4).

- metrics():   macro-F1 (ana metrik), accuracy, sınıf bazında precision/recall, karışıklık matrisi
- benchmark(): mesaj başına gecikme. Mesajlar tek tek işlenir (canlı sohbetteki gibi),
               ısınma turlarından sonra en az 100 çalıştırmanın ortalaması alınır.
- hardware():  sonuç tablosunda belirtilecek donanım bilgisi
- save_result(): results/<isim>.json
"""
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def metrics(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, target_names=["temiz", "hakaret"],
                                output_dict=True, zero_division=0)
    return {
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "offensive_precision": round(rep["hakaret"]["precision"], 4),
        "offensive_recall": round(rep["hakaret"]["recall"], 4),
        "offensive_f1": round(rep["hakaret"]["f1-score"], 4),
        "confusion_matrix [[TN,FP],[FN,TP]]": confusion_matrix(y_true, y_pred).tolist(),
    }


def benchmark(predict_one, texts, n_runs: int = 200, warmup: int = 10) -> dict:
    """predict_one(text) fonksiyonunu tek tek mesajlarla çağırıp süreyi ölçer (ms)."""
    texts = list(texts)
    for t in texts[:warmup]:
        predict_one(t)
    times = []
    for i in range(n_runs):
        t = texts[i % len(texts)]
        start = time.perf_counter()
        predict_one(t)
        times.append((time.perf_counter() - start) * 1000)
    arr = np.array(times)
    return {"latency_ms_mean": round(float(arr.mean()), 3),
            "latency_ms_p95": round(float(np.percentile(arr, 95)), 3),
            "latency_runs": n_runs}


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                          text=True).strip()
        except Exception:  # noqa: BLE001
            pass
    return f"{cpu}, {platform.system()} {platform.release()}, CPU"


def size_mb(*paths) -> float:
    total = 0
    for p in paths:
        p = Path(p)
        files = p.rglob("*") if p.is_dir() else [p]
        total += sum(f.stat().st_size for f in files if f.is_file())
    return round(total / 1024 / 1024, 3)


def save_result(name: str, result: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    result = {"model": name, "hardware": hardware(), **result}
    with open(RESULTS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Kaydedildi: results/{name}.json")


def print_errors(texts, y_true, y_pred, k: int = 5) -> dict:
    """Sunumdaki 'hata örneği' slaytı için yanlış tahminlerden örnekler."""
    texts, y_true, y_pred = list(texts), np.asarray(y_true), np.asarray(y_pred)
    fp = [texts[i] for i in np.where((y_pred == 1) & (y_true == 0))[0][:k]]
    fn = [texts[i] for i in np.where((y_pred == 0) & (y_true == 1))[0][:k]]
    print("  Yanlış alarm (temiz ama hakaret dendi):")
    for t in fp:
        print("    -", t[:110])
    print("  Kaçırılan (hakaret ama temiz dendi):")
    for t in fn:
        print("    -", t[:110])
    return {"false_positives": fp, "false_negatives": fn}
