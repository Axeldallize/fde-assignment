from pathlib import Path
import json

from backend.index.store import chunks_dir
from backend.index.lexical import build_index, save_index, search

# Load chunk texts and ids from sidecars
texts_path = chunks_dir() / "smoke-pdf.texts.json"
map_path = chunks_dir() / "smoke-pdf.map.json"

texts = json.loads(texts_path.read_text(encoding="utf-8"))
id_map = json.loads(map_path.read_text(encoding="utf-8"))
ids = sorted(id_map, key=lambda k: id_map[k])

corpus = list(zip(ids, texts))
vectorizer, matrix, saved_ids = build_index(corpus)
paths = save_index(vectorizer, matrix, saved_ids)
print("Index saved:", paths)

print("Query: methods")
print("Results:", search("methods", top_k=3))
