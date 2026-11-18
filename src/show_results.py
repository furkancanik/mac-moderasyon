"""results/*.json dosyalarından sonuç tablosunu (R4) yazdırır.

Çalıştır:  python -m src.show_results
Çıktı:     ekranda tablo + results/results_table.md (sunuma kopyalanabilir)
"""
import json

from src.evaluate import RESULTS_DIR

ORDER = [
    ("rule_lexicon", "Kural: küfür listesi + edit distance"),
    ("tfidf_lr", "tf-idf + logistic regression"),
    ("w2v_cnn", "word2vec + 1D CNN (kendi eğitimimiz)"),
    ("berturk_fp32", "BERTurk fine-tuned (PyTorch FP32)"),
    ("berturk_onnx_fp32", "BERTurk (ONNX FP32)"),
    ("berturk_int8", "BERTurk quantized (ONNX INT8)"),
]


def main() -> None:
    rows, hw = [], None
    for key, label in ORDER:
        p = RESULTS_DIR / f"{key}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        hw = hw or r.get("hardware")
        rows.append(f"| {label} | {r['macro_f1']:.3f} | {r['accuracy']:.3f} | "
                    f"{r['offensive_recall']:.3f} | {r['size_mb']:.2f} | "
                    f"{r['latency_ms_mean']:.3f} | {r['latency_ms_p95']:.3f} |")
    head = ("| Model | Macro-F1 | Accuracy | Hakaret recall | Boyut (MB) | "
            "Gecikme ort. (ms) | Gecikme p95 (ms) |\n|---|---|---|---|---|---|---|")
    table = "\n".join([head, *rows])
    note = (f"\nTest seti: OffensEval-TR 2020, 3.528 tweet. Gecikme: tek mesaj, "
            f"200 çalıştırmanın ortalaması. Donanım: {hw}")
    print(table + note)
    (RESULTS_DIR / "results_table.md").write_text(table + "\n" + note + "\n", encoding="utf-8")
    print("\nKaydedildi: results/results_table.md")


if __name__ == "__main__":
    main()
