"""OffensEval-TR 2020 veri setini indirir ve data/ altına CSV olarak kaydeder.

Kaynak: https://huggingface.co/datasets/stefan-it/offenseval2020_tr
  (Çöltekin, 2020 - "A Corpus of Turkish Offensive Language on Social Media", CC-BY)
  train: 30.000 tweet, dev: 1.756 tweet, test: 3.528 tweet (resmi test seti, dokunulmamış)

Dosya biçimi (Flair/fastText): her satır "__label__NOT <tweet>" veya "__label__OFF <tweet>"

Çalıştır:  python -m src.prepare_data
Çıktı:     data/train.csv, data/dev.csv, data/test.csv  (sütunlar: text, label)
           results/data_stats.json  (sunumdaki veri slaytı için)
           label: 0 = temiz (NOT), 1 = hakaret (OFF)
"""
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "stefan-it/offenseval2020_tr"
SPLITS = ["train", "dev", "test"]
LABELS = {"NOT": 0, "OFF": 1}

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def parse_file(path: Path) -> pd.DataFrame:
    rows, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("__label__"):
                skipped += 1
                continue
            label_tok, _, text = line.partition(" ")
            label = label_tok.replace("__label__", "")
            if label not in LABELS or not text.strip():
                skipped += 1
                continue
            rows.append({"text": text.strip(), "label": LABELS[label]})
    if skipped:
        print(f"  uyarı: {path.name} içinde {skipped} satır atlandı")
    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    stats = {"source": f"https://huggingface.co/datasets/{REPO_ID}", "splits": {}}

    frames = {}
    for split in SPLITS:
        local = hf_hub_download(repo_id=REPO_ID, filename=f"{split}.txt", repo_type="dataset")
        df = parse_file(Path(local))
        df.to_csv(DATA_DIR / f"{split}.csv", index=False)
        frames[split] = df

        n_off = int(df["label"].sum())
        stats["splits"][split] = {
            "total": len(df),
            "clean_NOT": len(df) - n_off,
            "offensive_OFF": n_off,
            "offensive_ratio": round(n_off / len(df), 4),
            "avg_words": round(df["text"].str.split().str.len().mean(), 1),
        }
        s = stats["splits"][split]
        print(f"{split:<5}: {s['total']:>6} örnek | temiz {s['clean_NOT']:>6} | "
              f"hakaret {s['offensive_OFF']:>5} (%{100 * s['offensive_ratio']:.1f}) | "
              f"ort. {s['avg_words']} kelime")

    # Test setinde eğitimle birebir aynı tweet var mı? (veri sızıntısı kontrolü)
    overlap = len(set(frames["train"]["text"]) & set(frames["test"]["text"]))
    stats["train_test_exact_overlap"] = overlap
    print(f"Eğitim ve test arasında birebir aynı tweet sayısı: {overlap}")

    with open(RESULTS_DIR / "data_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\nÖrnek hakaret içeren tweetler (ilk 3):")
    for t in frames["train"].query("label == 1")["text"].head(3):
        print("  -", t[:100])
    print(f"\nKaydedildi: {DATA_DIR}/train.csv, dev.csv, test.csv ve results/data_stats.json")


if __name__ == "__main__":
    main()
