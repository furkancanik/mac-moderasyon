"""Edit distance ile çalışan küfür listesi eşleştirici: kural tabanlı baseline (Alan B).

Her kelime normalize edilip Türkçe karakterleri sadeleştirildikten sonra listedeki
kökle karşılaştırılır:
  - Kısa kökler (<= 3 harf, ör. "amk", "mal") yalnızca kelimenin tamamıyla birebir
    eşleşir; yoksa "malzeme" gibi masum kelimeler yakalanırdı.
  - Daha uzun kökler kelimenin başıyla karşılaştırılır (Türkçe eklemeli bir dil:
    "şerefsizler", "salaklık"). İzin verilen minimum edit distance:
    4-5 harfli kökte 0, 6-7 harfli kökte 1, 8+ harfli kökte 2. Böylece "serefsz",
    "orspu" gibi yazım hataları yakalanırken "salam" ("salak" değil) karışmaz.

Minimum edit distance (Levenshtein) dinamik programlama ile burada yazıldı.
"""
from functools import lru_cache
from pathlib import Path

from src.normalize import fold_turkish, normalize

LEXICON_PATH = Path(__file__).resolve().parent / "lexicon_tr.txt"


def load_lexicon(path: Path = LEXICON_PATH) -> list[str]:
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            words.append(fold_turkish(normalize(line)))
    return sorted(set(words))


def edit_distance(a: str, b: str) -> int:
    """Levenshtein mesafesi: ekleme = silme = değiştirme = 1 maliyet."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,                 # silme
                           cur[j - 1] + 1,              # ekleme
                           prev[j - 1] + (ca != cb)))   # değiştirme
        prev = cur
    return prev[-1]


def _max_dist(stem: str) -> int | None:
    """None: sadece birebir eşleşme; sayı: prefix için izin verilen edit distance."""
    if len(stem) <= 3:
        return None
    if len(stem) <= 5:
        return 0
    return 1 if len(stem) <= 7 else 2


class LexiconMatcher:
    def __init__(self, lexicon: list[str] | None = None):
        self.lexicon = lexicon if lexicon is not None else load_lexicon()

    @lru_cache(maxsize=100_000)
    def match_token(self, tok: str) -> str | None:
        for stem in self.lexicon:
            k = _max_dist(stem)
            if k is None:
                if tok == stem:
                    return stem
                continue
            if len(tok) < len(stem) - k:
                continue
            # Kelimenin başını kökle karşılaştır; ek almış halleri de yakala.
            # Kök uzunluğunun ±k fazlası kadar prefix'e bak (eksik/fazla harfli yazımlar için).
            best = min(edit_distance(tok[:n], stem)
                       for n in range(max(1, len(stem) - k), min(len(tok), len(stem) + k) + 1))
            if best <= k:
                return stem
        return None

    def find(self, text: str) -> list[str]:
        """Metindeki eşleşen küfür köklerini döndürür."""
        hits = []
        for tok in fold_turkish(normalize(text)).split():
            stem = self.match_token(tok)
            if stem:
                hits.append(stem)
        return hits

    def predict(self, text: str) -> int:
        return int(bool(self.find(text)))


if __name__ == "__main__":
    m = LexiconMatcher()
    print(f"Listede {len(m.lexicon)} kök var")
    for s in ["s1kt1r git hakem", "serefsizler", "orspu çocuğu", "sıkıntı yok",
              "bu maç çok iyiydi", "gerizekalılar", "amk ya", "malzeme geldi"]:
        print(f"{s!r:25} -> {m.find(s)}")
