import json
import sqlite3


class MemoryDatabase:
    def __init__(self, path=":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                goal TEXT,
                success BOOLEAN,
                replay_count INTEGER DEFAULT 0,
                replay_success_count INTEGER DEFAULT 0,
                replay_depth INTEGER DEFAULT 0,
                archived BOOLEAN DEFAULT 0,
                workspace_path TEXT,
                metadata TEXT
            )
        """)
        self.conn.commit()

    def record_task(self, task_id, goal, success=False, workspace_path=None, replay_depth=0):
        self.conn.execute(
            "INSERT OR REPLACE INTO tasks (id, goal, success, workspace_path, replay_depth) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, goal, success, workspace_path, replay_depth)
        )
        self.conn.commit()

    def get_task(self, task_id):
        cur = self.conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,))
        row = cur.fetchone()
        if row:
            return {
                "id": row[0],
                "goal": row[1],
                "success": row[2],
                "replay_count": row[3],
                "replay_success_count": row[4],
                "replay_depth": row[5],
                "archived": row[6],
                "workspace_path": row[7],
                "metadata": row[8],
            }
        return None

    def find_similar_goals(self, goal, limit=3):
        words = set(goal.lower().split())
        cursor = self.conn.execute(
            "SELECT id, goal, success, replay_count, replay_success_count, replay_depth, archived FROM tasks"
        )
        candidates = []
        for row in cursor.fetchall():
            task_words = set(row[1].lower().split())
            similarity = len(words & task_words) / len(words | task_words) if words else 0
            if similarity > 0.3:
                candidates.append({
                    "id": row[0],
                    "goal": row[1],
                    "success": row[2],
                    "replay_count": row[3],
                    "replay_success_count": row[4],
                    "replay_depth": row[5],
                    "archived": row[6],
                    "similarity": similarity,
                })
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:limit]

    def update_task_metadata(self, task_id, metadata):
        self.conn.execute(
            "UPDATE tasks SET metadata = ? WHERE id = ?",
            (json.dumps(metadata), task_id)
        )
        self.conn.commit()

    def register_replay_attempt(self, task_id):
        self.conn.execute(
            "UPDATE tasks SET replay_count = replay_count + 1 WHERE id = ?",
            (task_id,)
        )
        self.conn.commit()

    def register_replay_result(self, task_id, success):
        if success:
            self.conn.execute(
                "UPDATE tasks SET replay_success_count = replay_success_count + 1 WHERE id = ?",
                (task_id,)
            )
        self.conn.commit()
