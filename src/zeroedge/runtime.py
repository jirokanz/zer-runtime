import uuid
import time
from zeroedge.core.event_bus import EventBus
from zeroedge.core.scheduler import Scheduler
from zeroedge.core.task_manager import TaskManager
from zeroedge.memory.database import MemoryDatabase
from zeroedge.memory.router import MemoryRouter
from zeroedge.workspace.manager import WorkspaceManager

def main():
    print("ZeroEdgeAI ZER Runtime v0.1.0")
    db = MemoryDatabase()
    router = MemoryRouter(db)
    task = {"id": "test", "goal": "Print hello world"}
    decision = router.route(task["goal"])
    print(f"Decision: {decision}")
