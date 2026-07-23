"""MemoryDB - Long-term memory substrate with hard compatibility filters."""

import sqlite3
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from zeroedge.memory.models import MemoryRecord, MemoryCandidate
from zeroedge.memory.migration_manager import MigrationManager
from zeroedge.core.version import ZER_VERSION, get_major_version

class MemoryDB:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._current_major = get_major_version(ZER_VERSION)
        migrations_dir = Path(__file__).parent / "migrations"
        MigrationManager(db_path, migrations_dir).initialize()

    def find_similar(self, goal: str, limit: int = 5) -> List[MemoryCandidate]:
        goal_words = set(goal.lower().split())
        if not goal_words:
            return []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, goal, plan, code, answer AS result,
                       success, execution_hash, zr_version, deps_hash,
                       provider_key, role, category, task_complexity, task_category,
                       created_at, last_used_at, usage_count
                FROM tasks
                WHERE success = 1 AND substr(zr_version, 1, 1) = ?
                ORDER BY created_at DESC LIMIT 50
            """, (self._current_major,))
            candidates = []
            for row in cursor.fetchall():
                record = MemoryRecord.from_db_row(dict(row))
                stored_words = set(record.goal.lower().split())
                if not stored_words:
                    continue
                overlap = len(goal_words & stored_words) / max(len(goal_words), len(stored_words))
                candidates.append(MemoryCandidate(record=record, similarity=overlap))
            candidates.sort(key=lambda x: x.similarity, reverse=True)
            return candidates[:limit]

    def save(self, record: MemoryRecord) -> None:
        d = record.to_db_dict()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tasks (
                    id, goal, plan, code, answer, success,
                    execution_hash, zr_version, deps_hash,
                    provider_key, role, category,
                    task_complexity, task_category,
                    created_at, last_used_at, usage_count
                ) VALUES (
                    :id, :goal, :plan, :code, :result, :success,
                    :execution_hash, :zr_version, :deps_hash,
                    :provider_key, :role, :category,
                    :task_complexity, :task_category,
                    :created_at, :last_used_at, :usage_count
                )
            """, d)

    def mark_used(self, memory_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE tasks SET last_used_at = CURRENT_TIMESTAMP, usage_count = usage_count + 1
                WHERE id = ?
            """, (memory_id,))

    def get_by_id(self, memory_id: str) -> Optional[MemoryRecord]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, goal, plan, code, answer AS result,
                       success, execution_hash, zr_version, deps_hash,
                       provider_key, role, category, task_complexity, task_category,
                       created_at, last_used_at, usage_count
                FROM tasks WHERE id = ?
            """, (memory_id,))
            row = cursor.fetchone()
            return MemoryRecord.from_db_row(dict(row)) if row else None
