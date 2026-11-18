"""Uygulamanın kullandığı beş model için ortak arayüz.

Her model predict_proba(text) -> hakaret olasılığı (0-1) döndürür.
Modeller ilk kullanımda yüklenir ve bellekte tutulur.
"""
import html
import time
from dataclasses import dataclass

from src.evaluate import ROOT
from src.lexicon import LexiconMatcher
from src.normalize import fold_turkish, normalize

MODELS = ROOT / "models"

INT8_NAME = "BERTurk INT8 (ONNX Runtime) - varsayılan"
MODEL_NAMES = [
    INT8_NAME,
    "BERTurk FP32 (PyTorch)",
    "word2vec + 1D CNN",
    "tf-idf + logistic regression",
    "Kural: küfür listesi + edit distance",
]


@dataclass
class Prediction:
    model: str
    label: int          # 1 = hakaret
    prob: float         # hakaret olasılığı
    ms: float           # bu mesaj için geçen süre

    @property
    def confidence(self) -> float:
        return self.prob if self.label else 1 - self.prob


class _Rule:
    def __init__(self, matcher):
        self.m = matcher

    def predict_proba(self, text):
        return 1.0 if self.m.find(text) else 0.0


class _Tfidf:
    def __init__(self):
        import joblib
        self.model = joblib.load(MODELS / "tfidf_lr.joblib")

    def predict_proba(self, text):
        return float(self.model.predict_proba([normalize(text)])[0, 1])


def _load(name: str, matcher):
    if name == INT8_NAME:
        from src.bert_model import OnnxBertClassifier
        return OnnxBertClassifier(MODELS / "berturk-onnx" / "model_int8.onnx",
                                  MODELS / "berturk-offensive")
    if name.startswith("BERTurk FP32"):
        from src.bert_model import BertClassifier
        return BertClassifier(MODELS / "berturk-offensive", device="cpu")
    if name.startswith("word2vec"):
        from src.w2v_cnn import W2VCNNClassifier
        return W2VCNNClassifier(MODELS / "w2v_cnn.pt")
    if name.startswith("tf-idf"):
        return _Tfidf()
    return _Rule(matcher)


class ModelZoo:
    def __init__(self):
        self.matcher = LexiconMatcher()
        self._cache = {}

    def get(self, name: str):
        if name not in self._cache:
            t0 = time.perf_counter()
            self._cache[name] = _load(name, self.matcher)
            print(f"  yüklendi: {name} ({time.perf_counter() - t0:.1f} sn)")
        return self._cache[name]

    def predict(self, name: str, text: str) -> Prediction:
        model = self.get(name)
        t0 = time.perf_counter()
        p = model.predict_proba(text)
        ms = (time.perf_counter() - t0) * 1000
        return Prediction(name, int(p >= 0.5), float(p), ms)

    def mask(self, text: str) -> str:
        """Hakaret sayılan mesajda listedeki kelimeleri yıldızlar;
        liste bir şey bulamazsa (ör. küfürsüz hakaret) mesajın tamamını gizler."""
        words = text.split()
        out, masked = [], False
        for w in words:
            toks = fold_turkish(normalize(w)).split()
            if any(self.matcher.match_token(t) for t in toks):
                out.append("\\*" * len(w))
                masked = True
            else:
                out.append(html.escape(w))
        return " ".join(out) if masked else "_[mesaj gizlendi]_"
