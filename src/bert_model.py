"""BERTurk sınıflandırıcıları (Alan C ve D): PyTorch FP32 ve ONNX Runtime (FP32 / INT8).

bert_preprocess(): BERTurk büyük-küçük harf duyarlı (cased) ve kendi alt-kelime (WordPiece)
tokenizer'ı var. Bu yüzden ağır normalizasyon yapmıyoruz; sadece @USER/URL yer tutucularını
siliyor ve üç veya daha fazla tekrarlanan harfi teke indiriyoruz.
"""
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 64
_PLACEHOLDER_RE = re.compile(r"@USER|\bURL\b")
_REPEAT_RE = re.compile(r"(\w)\1{2,}")


def bert_preprocess(text: str) -> str:
    text = _PLACEHOLDER_RE.sub(" ", str(text))
    text = _REPEAT_RE.sub(r"\1", text)
    return " ".join(text.split()) or "."


def _softmax_pos(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e[:, 1] / e.sum(axis=1)


class BertClassifier:
    """Hugging Face formatında kaydedilmiş BERTurk (FP32, PyTorch)."""

    def __init__(self, path: str | Path, device: str = "cpu"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()

    @torch.inference_mode()
    def predict_proba_batch(self, texts, bs: int = 64) -> np.ndarray:
        out = []
        texts = list(texts)
        for i in range(0, len(texts), bs):
            enc = self.tok([bert_preprocess(t) for t in texts[i:i + bs]], padding=True,
                           truncation=True, max_length=MAX_LEN, return_tensors="pt").to(self.device)
            out.append(_softmax_pos(self.model(**enc).logits.float().cpu().numpy()))
        return np.concatenate(out)

    def predict_proba(self, text: str) -> float:
        return float(self.predict_proba_batch([text])[0])

    def predict(self, text: str) -> int:
        return int(self.predict_proba(text) >= 0.5)


class OnnxBertClassifier:
    """ONNX Runtime ile CPU üzerinde çalışan BERTurk (FP32 veya INT8 quantize edilmiş)."""

    def __init__(self, onnx_path: str | Path, tokenizer_path: str | Path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)

    def predict_proba_batch(self, texts, bs: int = 64) -> np.ndarray:
        out = []
        texts = list(texts)
        for i in range(0, len(texts), bs):
            enc = self.tok([bert_preprocess(t) for t in texts[i:i + bs]], padding=True,
                           truncation=True, max_length=MAX_LEN, return_tensors="np")
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
            out.append(_softmax_pos(self.sess.run(None, feed)[0]))
        return np.concatenate(out)

    def predict_proba(self, text: str) -> float:
        return float(self.predict_proba_batch([text])[0])

    def predict(self, text: str) -> int:
        return int(self.predict_proba(text) >= 0.5)
