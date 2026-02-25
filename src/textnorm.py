import re
import unicodedata

def normalize_fr(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ0-9\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> list[str]:
    return normalize_fr(s).split()