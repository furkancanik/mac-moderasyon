"""Gürültülü sohbet metni için regex tabanlı normalizasyon (Alan B: Klasik NLP).

Yapılanlar:
  1. @USER ve URL yer tutucularını siler (veri setinde kullanıcı adları ve linkler bunlarla değiştirilmiş)
  2. Türkçeye uygun küçük harfe çevirme (I -> ı, İ -> i)
  3. Leetspeak çözme: harf içeren kelimelerde 1->i, 3->e, 4->a, 0->o, 5->s, 7->t, $->s, @->a
     ("s1kt1r" -> "siktir"). Sadece rakamdan oluşan "+1", "2020" gibi sayılara dokunmaz.
  4. Üç veya daha fazla tekrarlanan harfi teke indirir ("salaaaak" -> "salak").
     Türkçedeki çift harflere ("saat", "dikkat") dokunmaz.
  5. Noktalama ve emojileri atıp kelimeleri tek boşlukla birleştirir.

fold_turkish(): ç->c, ğ->g, ö->o, ş->s, ü->u (ı korunur). Türkçe karakter kullanmadan
yazılan kelimeleri ("serefsiz") listedeki kelimelerle ("şerefsiz") eşleştirmek için.
"""
import re

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                       "7": "t", "$": "s", "@": "a"})
# Not: ı -> i dönüşümü bilerek yapılmıyor; "sık"/"sik", "sıkıntı"/"sikik" gibi
# anlamı tamamen değişen çiftler yanlış alarm üretiyordu.
_FOLD = str.maketrans({"ç": "c", "ğ": "g", "ö": "o", "ş": "s", "ü": "u",
                       "â": "a", "î": "i", "û": "u"})

_PLACEHOLDER_RE = re.compile(r"@USER|\bURL\b")
_TOKEN_RE = re.compile(r"[\w@$]+")
_HAS_LETTER_RE = re.compile(r"[^\W\d_]")
_REPEAT_RE = re.compile(r"(\w)\1{2,}")


def turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def _fix_token(tok: str) -> str:
    if _HAS_LETTER_RE.search(tok):          # harf içeriyorsa leetspeak çöz
        tok = tok.translate(_LEET)
    tok = tok.replace("@", "").replace("$", "").strip("_")
    return _REPEAT_RE.sub(r"\1", tok)


def normalize(text: str) -> str:
    text = _PLACEHOLDER_RE.sub(" ", text)
    text = turkish_lower(text)
    tokens = (_fix_token(t) for t in _TOKEN_RE.findall(text))
    return " ".join(t for t in tokens if t)


def fold_turkish(text: str) -> str:
    return text.translate(_FOLD)


if __name__ == "__main__":
    ornekler = [
        "@USER S1KT1R git hakem!!! URL",
        "salaaaaak mısın 😡😡",
        "Saat 20:00'de maç var, +1 dikkat",
        "İSTANBUL'da $erefs1z hakem",
    ]
    for o in ornekler:
        print(f"{o!r:45} -> {normalize(o)!r}")
