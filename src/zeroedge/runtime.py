
import uuid
import time
from zeroedge.core.event_bus import SyncEventBus
from zeroedge.core.scheduler import Scheduler
from zeroedge.core.task_manager import TaskManager
from zeroedge.memory.database import MemoryDatabase
from zeroedge.memory.router import MemoryRouter
from zeroedge.memory.confidence import ConfidenceEvaluator
from zeroedge.memory.policy import MemoryDecisionPolicy
from zeroedge.memory.guard import ReplayGuard
from zeroedge.workspace.manager import WorkspaceManager
from zeroedge.replay.manager import ReplayManager
from zeroedge.core.logger import Logger

def main():
    print("ZeroEdgeAI ZER Runtime v0.1.0 (Full Memory)")
    logger = Logger()
    db = MemoryDatabase(":memory:")
    evaluator = ConfidenceEvaluator()
    policy = MemoryDecisionPolicy()
    guard = ReplayGuard()
    router = MemoryRouter(db, evaluator, policy, guard)

    # Simulate a task
    task_id = "test_001"
    goal = "Print hello world"
    decision = router.route(goal)
    print(f"Decision for '{goal}': {decision}")
    if decision["decision"] == "regenerate":
        print("Generating fresh solution (no memory)")
    elif decision["decision"] == "reuse":
        print(f"Reusing candidate: {decision['candidate']['id']}")
    elif decision["decision"] == "adapt":
        print(f"Adapting candidate: {decision['candidate']['id']}")

if __name__ == "__main__":
    main()
