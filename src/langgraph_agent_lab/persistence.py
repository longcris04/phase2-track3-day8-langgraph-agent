"""Checkpointer adapter.

Maps a string ``kind`` to a LangGraph checkpointer so the graph can persist
state per ``thread_id``. SQLite is implemented for the persistence extension
track (durable, survives process restart); MemorySaver is the default for
fast/offline runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Default on-disk location for the SQLite checkpoint database.
DEFAULT_SQLITE_PATH = "outputs/checkpoints.sqlite"


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    - ``"none"``     → no checkpointer (graph holds no cross-invocation state)
    - ``"memory"``   → in-process MemorySaver (default; lost on exit)
    - ``"sqlite"``   → durable SqliteSaver with WAL mode (survives restart)
    - ``"postgres"`` → optional extension (not implemented here)
    """
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if kind == "sqlite":
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        # database_url may be a plain path or a sqlite:/// URL.
        db_path = database_url or DEFAULT_SQLITE_PATH
        if db_path.startswith("sqlite:///"):
            db_path = db_path[len("sqlite:///") :]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False lets the connection be reused across LangGraph's
        # worker threads; WAL mode improves concurrent read/write durability.
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        saver = SqliteSaver(conn=conn)
        saver.setup()  # create checkpoint tables if they don't exist
        return saver

    if kind == "postgres":
        raise NotImplementedError(
            "Postgres checkpointer is an optional extension; use 'sqlite' or 'memory'."
        )

    raise ValueError(f"Unknown checkpointer kind: {kind}")
