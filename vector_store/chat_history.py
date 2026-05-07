import json
import os
from datetime import datetime

HISTORY_FILE = "chat_history.json"


def load_chat_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_message(role: str, content: str):
    history = load_chat_history()

    history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)





































