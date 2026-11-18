# Maç Sohbeti Moderasyonu: Türkçe Hakaret Tespiti

SWE015 ödevi, Use Case 4 (Real-Time Chat Moderation). Canlı maç yayını sohbetindeki
mesajları CPU üzerinde anlık olarak **temiz / hakaret** diye etiketler ve hakaret
içeren mesajları maskeler.

> Bu README proje ilerledikçe tamamlanacak.

## Kurulum

Python 3.10–3.13 ile test edildi (geliştirme: Python 3.13, macOS Apple Silicon).

```bash
# 1) Sanal ortam
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

# 2) PyTorch
#    macOS (Apple Silicon, MPS):  pip install torch
#    NVIDIA GPU (CUDA 12.1 örneği): pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch

# 3) Diğer kütüphaneler
pip install -r requirements.txt

# 4) Ortam kontrolü ve veri hazırlığı
python -m src.check_env
python -m src.prepare_data

# 5) Klasik baseline'lar (küfür listesi + tf-idf)
python -m src.train_baselines

# 6) word2vec + 1D CNN
python -m src.train_w2v_cnn

# 7) BERTurk fine-tuning ve INT8 quantization
python -m src.train_bert
python -m src.quantize_bert

# 8) Sonuç tablosu
python -m src.show_results
```

## Uygulamayı çalıştırma

```bash
python app.py
```

Tarayıcıda `http://127.0.0.1:7860` açılır. Modeller yerelde yoksa Hugging Face Hub'dan
otomatik indirilir.

## Veri

[OffensEval-TR 2020](https://huggingface.co/datasets/stefan-it/offenseval2020_tr)
(Çöltekin, 2020, CC-BY): 30.000 eğitim, 1.756 dev ve 3.528 test tweet'i.
Etiketler: `0 = temiz (NOT)`, `1 = hakaret (OFF)`. Bütün modeller aynı test setinde
değerlendirilir.

## Proje yapısı

```
src/        kaynak kod
data/       indirilen veri (repoda yok)
models/     eğitilmiş modeller (repoda yok, Hugging Face Hub'a yüklenecek)
results/    ölçüm sonuçları (JSON)
```

## Kaynaklar

- Çağrı Çöltekin. 2020. *A Corpus of Turkish Offensive Language on Social Media.* LREC 2020.
