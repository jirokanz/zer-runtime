from zeroedge.memory.database import MemoryDatabase
class MemoryRouter:
    def __init__(self, db):
        self.db = db
    def route(self, goal):
        # For now, always regenerate
        return {"decision": "regenerate", "confidence": 0.0}
