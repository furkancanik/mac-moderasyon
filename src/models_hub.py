"""Eğitilmiş modelleri Hugging Face Hub'dan indirir / Hub'a yükler (R7).

Büyük model dosyaları GitHub reposunda tutulmaz. Repoyu yeni klonlayan biri
`python app.py` çalıştırdığında eksik modeller HF_REPO'dan otomatik indirilir.

Yükleme (sadece bir kez, modelleri eğiten kişi yapar):
    hf auth login
    python -m src.models_hub upload
"""
import os
import sys

from src.evaluate import ROOT

MODELS = ROOT / "models"

# Gün 6'da kendi Hugging Face kullanıcı adınla doldurulacak
HF_REPO = os.environ.get("MODERASYON_HF_REPO", "KULLANICI_ADI/mac-moderasyon-models")

REQUIRED = [
    "tfidf_lr.joblib",
    "w2v_cnn.pt",
    "berturk-offensive/config.json",
    "berturk-onnx/model_int8.onnx",
]
# Uygulamanın ihtiyaç duyduğu dosyalar (ONNX FP32 kopyası yüklenmez, gereksiz yere 420 MB)
PATTERNS = ["tfidf_lr.joblib", "w2v_cnn.pt", "word2vec.kv*", "berturk-offensive/*",
            "berturk-onnx/model_int8.onnx*"]


def ensure_models() -> None:
    missing = [p for p in REQUIRED if not (MODELS / p).exists()]
    if not missing:
        return
    if HF_REPO.startswith("KULLANICI_ADI"):
        sys.exit(f"Eksik model dosyaları: {missing}\n"
                 "Modelleri eğit (README 5-7. adımlar) ya da src/models_hub.py içindeki "
                 "HF_REPO değerini ayarla.")
    from huggingface_hub import snapshot_download
    print(f"Modeller Hugging Face Hub'dan indiriliyor: {HF_REPO}")
    snapshot_download(repo_id=HF_REPO, local_dir=MODELS, allow_patterns=PATTERNS)


def upload() -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(MODELS), repo_id=HF_REPO, repo_type="model",
                      allow_patterns=PATTERNS)
    print(f"Yüklendi: https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "upload":
        upload()
    else:
        ensure_models()
        print("Modeller hazır.")
