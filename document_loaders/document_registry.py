import json
import os
from datetime import datetime

REGISTRY_FILE = "document_registry.json"


def load_registry():
    if not os.path.exists(REGISTRY_FILE):
        return []

    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def register_document(file_path: str, pages: int, chunks: int):
    registry = load_registry()

    registry.append({
        "file_path": file_path,
        "pages": pages,
        "chunks": chunks,
        "indexed_at": datetime.now().isoformat()
    })

    save_registry(registry)