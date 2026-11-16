"""SQLite-backed sessions with disposable, atomically replaced export artifacts.

The SQLite transaction is the commit point. Export failure may leave artifacts
stale, but never changes the committed state; load() always reads SQLite. A later
save regenerates exports. Filesystem exports are atomic individually, not as a
multi-file transaction. Keep this private data directory on a trusted filesystem.
"""
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import List

from .models import ResearchState, utcnow
from .reporting import render_report

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_SECRET_KEY = re.compile(r"key|secret|password|passwd|token|credential|authorization|cookie", re.I)


def _validate_id(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid identifier: use 1–128 ASCII letters, digits, underscores or hyphens; start with a letter or digit")
    return value


def _directory(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("Symlink artifact directories are not allowed")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _atomic_write(path: Path, text: str) -> None:
    """Flush a private temporary file, then replace the destination in one step."""
    if path.is_symlink():
        raise ValueError("Symlink artifact files are not allowed")
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


class Store:
    """Persist research state, never environment variables or credential settings.

    Credential-named config entries are deliberately omitted from persisted copies.
    Callers must not put credentials in free-form research content. Instances are
    synchronous and should be used in the thread in which they were created.
    """

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        _directory(self.root)
        _directory(self.root / "sessions")
        database = self.root / "research.sqlite3"
        if database.is_symlink():
            raise ValueError("Symlink databases are not allowed")
        # Create with private permissions before SQLite opens it.
        fd = os.open(str(database), os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        self._db = sqlite3.connect(str(database), timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY, topic TEXT NOT NULL, status TEXT NOT NULL,
            iteration INTEGER NOT NULL, updated_at TEXT NOT NULL,
            stop_reason TEXT NOT NULL, state_json TEXT NOT NULL
        )""")
        self._db.commit()
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Store is closed")

    def save(self, state: ResearchState) -> None:
        self._check_open()
        _validate_id(state.id)
        for source in state.sources:
            _validate_id(source.id)
        if len({s.id for s in state.sources}) != len(state.sources):
            raise ValueError("Source identifiers must be unique")
        # Do not let live secrets enter SQLite (including its WAL) or exports.
        snapshot = ResearchState.model_validate(state.model_dump(mode="json"))
        snapshot.config = {key: value for key, value in snapshot.config.items()
                           if not _SECRET_KEY.search(key)}
        snapshot.updated_at = utcnow()
        payload = snapshot.model_dump(mode="json")
        serialized = _json(payload)
        sessions = _directory(self.root / "sessions")
        session = _directory(sessions / snapshot.id)
        raw = _directory(session / "sources")
        checkpoints = _directory(session / "checkpoints")
        checkpoint = _directory(checkpoints / "{:04d}".format(snapshot.iteration))
        # Check directories before the canonical commit; failed exports after this
        # transaction cannot roll back or corrupt the already saved research.
        with self._db:
            self._db.execute("""INSERT INTO sessions
                (id, topic, status, iteration, updated_at, stop_reason, state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                topic=excluded.topic, status=excluded.status,
                iteration=excluded.iteration, updated_at=excluded.updated_at,
                stop_reason=excluded.stop_reason, state_json=excluded.state_json""",
                (snapshot.id, snapshot.topic, snapshot.status, snapshot.iteration,
                 snapshot.updated_at, snapshot.stop_reason, serialized))
        state.updated_at = snapshot.updated_at
        for source in snapshot.sources:
            _atomic_write(raw / (source.id + ".txt"), source.text)
        _atomic_write(session / "state.json", serialized)
        for name in ("plan", "evidence", "hypotheses", "decisions", "outline"):
            artifact = _json(payload[name])
            _atomic_write(session / (name + ".json"), artifact)
            _atomic_write(checkpoint / (name + ".json"), artifact)
        if snapshot.report is not None:
            _atomic_write(session / "report.md", render_report(snapshot))
            _atomic_write(session / "report.json", _json(payload["report"]))
        else:
            # Never leave an old final report masquerading as the current one.
            for name in ("report.md", "report.json"):
                path = session / name
                if path.exists() or path.is_symlink():
                    path.unlink()

    def load(self, session_id: str) -> ResearchState:
        self._check_open()
        _validate_id(session_id)
        row = self._db.execute("SELECT state_json FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise FileNotFoundError("Research session not found")
        return ResearchState.model_validate_json(row["state_json"])

    def list_sessions(self) -> List[dict]:
        self._check_open()
        rows = self._db.execute("""SELECT id, topic, status, iteration, updated_at, stop_reason
            FROM sessions ORDER BY updated_at DESC, id ASC""").fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        if not self._closed:
            self._db.close()
            self._closed = True
