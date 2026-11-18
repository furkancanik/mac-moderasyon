"""Gün 4b: fine-tune edilmiş BERTurk'ü ONNX'e aktarır ve INT8'e quantize eder (Alan D).

  1) PyTorch modeli ONNX formatına aktarılır (FP32).
  2) ONNX Runtime dynamic quantization: Linear katmanların ağırlıkları INT8'e çevrilir,
     aktivasyonların ölçeği çalışma anında hesaplanır. Kalibrasyon verisi gerekmez.
  3) ONNX FP32 ve ONNX INT8 aynı test setinde ölçülür. Gecikme CPU'da, tek tek mesajlarla
     ve 200 çalıştırmanın ortalaması olarak hesaplanır; 50 ms bütçesiyle karşılaştırılır.

Çalıştır:  python -m src.quantize_bert
Çıktı:     models/berturk-onnx/model_fp32.onnx, model_int8.onnx
           results/berturk_onnx_fp32.json, results/berturk_int8.json
"""
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.bert_model import OnnxBertClassifier
from src.evaluate import ROOT, benchmark, metrics, print_errors, save_result

SRC = ROOT / "models" / "berturk-offensive"
OUT = ROOT / "models" / "berturk-onnx"
FP32 = OUT / "model_fp32.onnx"
INT8 = OUT / "model_int8.onnx"
LATENCY_BUDGET_MS = 50.0
NAMES = ["input_ids", "attention_mask", "token_type_ids"]


class _LogitsOnly(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids).logits


def onnx_size_mb(path) -> float:
    """ONNX dosyası + varsa ayrı ağırlık dosyası (.data)."""
    files = [path] + list(path.parent.glob(path.name + "*.data")) + \
        list(path.parent.glob(path.stem + "*.data"))
    return round(sum(f.stat().st_size for f in set(files) if f.exists()) / 1024 / 1024, 3)


def export_onnx() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        SRC, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(SRC)
    dummy = tok(["örnek bir mesaj", "hakem bugün çok kötüydü"], padding=True,
                return_tensors="pt")
    args = tuple(dummy[n] for n in NAMES)
    dyn = {n: {0: "batch", 1: "seq"} for n in NAMES}
    dyn["logits"] = {0: "batch"}
    wrapped = _LogitsOnly(model)
    try:
        torch.onnx.export(wrapped, args, str(FP32), input_names=NAMES, output_names=["logits"],
                          dynamic_axes=dyn, opset_version=17, dynamo=False)
        print("   ONNX'e aktarıldı (TorchScript exporter)")
    except Exception as e:  # noqa: BLE001
        print(f"   TorchScript exporter çalışmadı ({e.__class__.__name__}), dynamo deneniyor...")
        torch.onnx.export(wrapped, args, str(FP32), input_names=NAMES, output_names=["logits"],
                          dynamic_axes=dyn, opset_version=18, dynamo=True)
        print("   ONNX'e aktarıldı (dynamo exporter)")

    # Doğrulama: PyTorch ve ONNX çıktıları aynı mı?
    with torch.no_grad():
        ref = wrapped(*args).numpy()
    import onnxruntime as ort
    sess = ort.InferenceSession(str(FP32), providers=["CPUExecutionProvider"])
    inp = {i.name for i in sess.get_inputs()}
    got = sess.run(None, {n: dummy[n].numpy().astype(np.int64) for n in NAMES if n in inp})[0]
    print(f"   PyTorch ile ONNX arasındaki en büyük fark: {np.abs(ref - got).max():.2e}")


def quantize() -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QInt8)


def evaluate(name: str, onnx_path, precision: str, test: pd.DataFrame) -> dict:
    clf = OnnxBertClassifier(onnx_path, SRC)
    y_pred = (clf.predict_proba_batch(test["text"]) >= 0.5).astype(int)
    m = metrics(test["label"], y_pred)
    lat = benchmark(clf.predict, test["text"])
    sz = onnx_size_mb(onnx_path)
    ok = lat["latency_ms_mean"] <= LATENCY_BUDGET_MS
    print(f"   {name}: macro-F1 {m['macro_f1']}  recall {m['offensive_recall']}  "
          f"gecikme {lat['latency_ms_mean']} ms (p95 {lat['latency_ms_p95']})  boyut {sz} MB  "
          f"-> {LATENCY_BUDGET_MS:.0f} ms bütçesi {'TUTTU' if ok else 'AŞILDI'}")
    errs = print_errors(test["text"], test["label"], y_pred)
    res = {**m, **lat, "size_mb": sz, "precision": precision, "runtime": "ONNX Runtime (CPU)",
           "latency_budget_ms": LATENCY_BUDGET_MS, "meets_budget": ok,
           "test_size": len(test), "errors": errs}
    save_result(name, res)
    return res


def main() -> None:
    test = pd.read_csv(ROOT / "data" / "test.csv")
    test["text"] = test["text"].astype(str)

    print("1) ONNX'e aktarma (FP32)...")
    export_onnx()
    print(f"   {FP32.name}: {onnx_size_mb(FP32)} MB")

    print("2) Dynamic INT8 quantization...")
    quantize()
    print(f"   {INT8.name}: {onnx_size_mb(INT8)} MB")

    print("3) Test seti ve gecikme (CPU)...")
    fp = evaluate("berturk_onnx_fp32", FP32, "FP32", test)
    q = evaluate("berturk_int8", INT8, "INT8 (dynamic)", test)

    print("\nÖzet: INT8 modeli FP32'ye göre")
    print(f"   boyut  : {fp['size_mb']} -> {q['size_mb']} MB "
          f"({fp['size_mb'] / q['size_mb']:.1f} kat küçük)")
    print(f"   gecikme: {fp['latency_ms_mean']} -> {q['latency_ms_mean']} ms")
    print(f"   macro-F1: {fp['macro_f1']} -> {q['macro_f1']}")


if __name__ == "__main__":
    main()
