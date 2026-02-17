"""Feedback storage helpers."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

feedback_lock = threading.Lock()


def make_feedback_entry(body: dict) -> dict:
    message = str(body.get("message", "")).strip()
    return {
        "id": int(datetime.now(timezone.utc).timestamp() * 1000),
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "type": body.get("type", "general"),
        "message": message,
        "context": body.get("context", {}),
        "userAgent": body.get("userAgent", ""),
        "screen": body.get("screen", ""),
        "status": "new",
    }


def append_feedback(feedback_file: str, entry: dict):
    with feedback_lock:
        feedback = []
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, "r") as f:
                    feedback = json.load(f)
            except Exception:
                feedback = []

        feedback.append(entry)
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
        with open(feedback_file, "w") as f:
            json.dump(feedback, f, indent=2)


def read_feedback_list(feedback_file: str):
    if not os.path.exists(feedback_file):
        return []
    try:
        with open(feedback_file, "r") as f:
            return json.load(f)
    except Exception:
        return []
