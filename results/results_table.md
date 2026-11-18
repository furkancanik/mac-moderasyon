| Model | Macro-F1 | Accuracy | Hakaret recall | Boyut (MB) | Gecikme ort. (ms) | Gecikme p95 (ms) |
|---|---|---|---|---|---|---|
| Kural: küfür listesi + edit distance | 0.671 | 0.842 | 0.297 | 0.00 | 0.018 | 0.041 |
| tf-idf + logistic regression | 0.762 | 0.838 | 0.672 | 5.66 | 0.510 | 0.659 |
| word2vec + 1D CNN (kendi eğitimimiz) | 0.727 | 0.838 | 0.496 | 12.64 | 0.114 | 0.139 |
| BERTurk fine-tuned (PyTorch FP32) | 0.807 | 0.868 | 0.756 | 422.72 | 14.546 | 18.306 |
| BERTurk (ONNX FP32) | 0.807 | 0.868 | 0.756 | 422.17 | 8.492 | 13.147 |
| BERTurk quantized (ONNX INT8) | 0.807 | 0.870 | 0.736 | 106.20 | 5.558 | 10.303 |

Test seti: OffensEval-TR 2020, 3.528 tweet. Gecikme: tek mesaj, 200 çalıştırmanın ortalaması. Donanım: Apple M4, Darwin 25.6.0, CPU
