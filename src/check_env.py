"""Ortam kontrolü: Python, kütüphaneler, GPU ve CPU bilgisini yazdırır.

Çalıştır:  python -m src.check_env
Çıktıyı bana gönder; donanım bilgisi sonuç tablosunda (R4) lazım olacak.
"""
import importlib
import platform
import sys

print(f"Python      : {sys.version.split()[0]}")
print(f"İşletim sis.: {platform.system()} {platform.release()}")
print(f"CPU         : {platform.processor() or platform.machine()}")

for name in ["torch", "transformers", "accelerate", "huggingface_hub", "numpy",
             "pandas", "sklearn", "gensim", "gradio", "onnxruntime"]:
    try:
        mod = importlib.import_module(name)
        print(f"{name:<16}: {getattr(mod, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"{name:<16}: KURULU DEĞİL ({e.__class__.__name__})")

try:
    import torch
    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    elif torch.backends.mps.is_available():
        print("GPU         : Apple Silicon (MPS)")
    else:
        print("GPU         : BULUNAMADI. Eğitim CPU'da yapılacak (yavaş olur).")
    print(f"CPU thread  : {torch.get_num_threads()}")
except ImportError:
    pass
