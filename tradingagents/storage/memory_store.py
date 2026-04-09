"""Persistent wrapper around FinancialSituationMemory with SQLite backing."""

import logging
from typing import List, Tuple, Optional

from tradingagents.agents.utils.memory import FinancialSituationMemory

logger = logging.getLogger(__name__)


class PersistentMemory(FinancialSituationMemory):
    """Extends in-memory BM25 memory with SQLite persistence.

    On init, loads any previously stored memories from the database.
    After adding new situations, call save() to persist them.
    """

    def __init__(self, name: str, db=None, config: dict = None):
        super().__init__(name, config)
        self.db = db
        self._loaded = False

    def load(self):
        """Load memories from database if available."""
        if self.db is None or self._loaded:
            return
        stored = self.db.load_memories(self.name)
        if stored:
            super().add_situations(stored)
            logger.info(f"Loaded {len(stored)} memories for '{self.name}'")
        self._loaded = True

    def save(self):
        """Persist current in-memory data to database.

        Only saves the entries that were added since the last load.
        """
        if self.db is None:
            return
        pairs = list(zip(self.documents, self.recommendations))
        if not pairs:
            return
        self.db.conn.execute(
            "DELETE FROM agent_memories WHERE memory_name = ?", (self.name,)
        )
        self.db.save_memories(self.name, pairs)
        logger.info(f"Saved {len(pairs)} memories for '{self.name}'")

    def add_situations(self, situations_and_advice: List[Tuple[str, str]]):
        """Add situations and auto-save if database is attached."""
        super().add_situations(situations_and_advice)
        if self.db is not None:
            self.save()
