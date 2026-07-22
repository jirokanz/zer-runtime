import sqlite3
class MemoryDatabase:
    def __init__(self, path=":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT, goal TEXT, success BOOLEAN)")
    def record_task(self, id, goal, success=False):
        self.conn.execute("INSERT OR REPLACE INTO tasks VALUES (?, ?, ?)", (id, goal, success))
        self.conn.commit()
    def get_task(self, id):
        cur = self.conn.execute("SELECT * FROM tasks WHERE id=?", (id,))
        row = cur.fetchone()
        return {"id": row[0], "goal": row[1], "success": row[2]} if row else None
