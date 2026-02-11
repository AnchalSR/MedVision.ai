"""
Simple in-memory chat session manager for MedVision.ai
This provides a tiny session store to keep recent messages per session_id.
Not persistent — suitable for local/demo use.
"""
from typing import Dict, List
from uuid import uuid4


class SessionManager:
    def __init__(self, max_history: int = 10):
        self._history: Dict[str, List[Dict[str, str]]] = {}
        self._max_history = max_history

    def create_session(self) -> str:
        sid = uuid4().hex
        self._history[sid] = []
        return sid

    def add_user_message(self, session_id: str, text: str):
        self._ensure(session_id)
        self._history[session_id].append({"role": "user", "text": text})
        self._trim(session_id)

    def add_bot_message(self, session_id: str, text: str):
        self._ensure(session_id)
        self._history[session_id].append({"role": "bot", "text": text})
        self._trim(session_id)

    def get_history(self, session_id: str):
        self._ensure(session_id)
        return list(self._history[session_id])

    def _ensure(self, session_id: str):
        if session_id not in self._history:
            self._history[session_id] = []

    def _trim(self, session_id: str):
        h = self._history[session_id]
        if len(h) > self._max_history:
            self._history[session_id] = h[-self._max_history:]


# Single shared manager
session_manager = SessionManager()
