"""Persistent memory — stores and retrieves facts across sessions."""

import json
import os
from datetime import datetime

MEMORY_FILE = "memory.json"


def _load() -> list[dict]:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return []


def _save(memories: list[dict]):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=2)


def add(fact: str):
    """Store a new fact with timestamp."""
    memories = _load()
    memories.append({
        "fact": fact,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    # Keep last 100 memories
    memories = memories[-100:]
    _save(memories)


def get_recent(n: int = 20) -> str:
    """Return recent memories formatted for the system prompt."""
    memories = _load()
    if not memories:
        return ""
    recent = memories[-n:]
    lines = [f"- [{m['date']}] {m['fact']}" for m in recent]
    return "Things you remember about this person:\n" + "\n".join(lines)
