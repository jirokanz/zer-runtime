import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zeroedge.runtime import main
from zeroedge.tools.python.security import validate_code
from zeroedge.tools.python.limits import make_preexec_fn, enforce_output_limit
from zeroedge.workspace.manager import WorkspaceManager
from zeroedge.memory.database import MemoryDatabase
from zeroedge.memory.guard import ReplayGuard
from zeroedge.replay.manager import ReplayManager
from zeroedge.core.logger import Logger


def test_import():
    assert main is not None


# ---- security.validate_code ----

def test_validate_code_allows_safe_code():
    assert validate_code("x = 1 + 1\nprint(x)") == []


def test_validate_code_blocks_direct_import():
    violations = validate_code("import os\nos.system('ls')")
    assert any("os" in v for v in violations)


def test_validate_code_blocks_dynamic_import_bypass():
    # The old blocklist only checked ast.Import/ImportFrom nodes and
    # missed this entirely.
    violations = validate_code("m = __import__('os')\nm.system('ls')")
    assert violations, "dynamic __import__ bypass should be caught"


def test_validate_code_blocks_eval_exec():
    assert validate_code("eval('1+1')") != []
    assert validate_code("exec('print(1)')") != []


def test_validate_code_blocks_dunder_escape():
    violations = validate_code("().__class__.__base__.__subclasses__()")
    assert violations


def test_validate_code_syntax_error():
    violations = validate_code("def f(:\n  pass")
    assert violations and "Syntax error" in violations[0]


# ---- limits.py ----

def test_enforce_output_limit_truncates():
    out = enforce_output_limit("a" * 100, 10)
    assert out.endswith("[TRUNCATED]")
    assert len(out.encode()) < 200


def test_enforce_output_limit_passthrough():
    assert enforce_output_limit("short", 100) == "short"


def test_make_preexec_fn_returns_callable_or_none():
    fn = make_preexec_fn()
    assert fn is None or callable(fn)


# ---- workspace clone (previously called a method that didn't exist) ----

def test_workspace_clone_task_workspace(tmp_path):
    ws = WorkspaceManager(root=str(tmp_path))
    ws.create_workspace("task_a")
    ws.save_code("task_a", "print('hi')")
    cloned = ws.clone_task_workspace("task_a", "task_b", new_goal="do it again")
    assert (Path(cloned) / "code.py").exists()
    assert (Path(cloned) / "goal.txt").read_text() == "do it again"


def test_workspace_clone_missing_source_does_not_crash(tmp_path):
    ws = WorkspaceManager(root=str(tmp_path))
    cloned = ws.clone_task_workspace("nonexistent", "task_new")
    assert Path(cloned).exists()


# ---- replay depth guard (previously never incremented anywhere) ----

def test_replay_depth_increments(tmp_path):
    # Before the fix, replay_depth was checked by ReplayGuard but never
    # written anywhere, so it stayed 0 forever. Confirm it now actually
    # accumulates across a chain of replays.
    db = MemoryDatabase(":memory:")
    ws = WorkspaceManager(root=str(tmp_path))
    logger = Logger()
    replay_mgr = ReplayManager(ws, db, logger)

    db.record_task("task_0", "do a thing", success=True)
    ws.create_workspace("task_0")

    current_id = "task_0"
    for _ in range(3):
        result = replay_mgr.create_replay_from_task(current_id)
        assert result is not None
        current_id = result["new_task_id"]

    final_task = db.get_task(current_id)
    assert final_task["replay_depth"] == 3


def test_replay_guard_blocks_at_max_depth():
    guard = ReplayGuard(max_depth=2)
    deep_candidate = {"success": True, "archived": False, "replay_depth": 2}
    result = guard.check(deep_candidate)
    assert result["allowed"] is False
    assert "replay_depth_exceeded" in result["reason"]


def test_replay_guard_allows_within_depth():
    guard = ReplayGuard(max_depth=2)
    shallow_candidate = {"success": True, "archived": False, "replay_depth": 1}
    result = guard.check(shallow_candidate)
    assert result["allowed"] is True


# ---- config.py (previously always empty regardless of config_path) ----

def test_config_loads_yaml(tmp_path):
    from zeroedge.core.config import Config
    config_file = tmp_path / "config.yaml"
    config_file.write_text("memory:\n  thresholds:\n    reuse: 0.9\n")
    cfg = Config(str(config_file))
    assert cfg.get("memory.thresholds.reuse") == 0.9
    assert cfg.get("memory.thresholds.adapt", 0.5) == 0.5


def test_config_missing_file_falls_back_to_default():
    from zeroedge.core.config import Config
    cfg = Config("/nonexistent/path.yaml")
    assert cfg.get("anything", "fallback") == "fallback"


# ---- adaptive provider ranking (was static priority only, never updated) ----

def test_provider_stats_recorded_and_retrieved(tmp_path):
    from zeroedge.agent import MemoryDB
    db = MemoryDB(str(tmp_path / "mem.db"))
    db.record_provider_call("groq", "coding", True, 500)
    db.record_provider_call("groq", "coding", True, 700)
    db.record_provider_call("groq", "coding", False, 900)
    stats = db.get_provider_stats("groq", "coding")
    assert stats["calls"] == 3
    assert stats["successes"] == 2
    assert stats["total_latency_ms"] == 2100


def test_registry_falls_back_to_static_priority_below_min_samples(tmp_path):
    from zeroedge.agent import MemoryDB, BaseProvider, ProviderRegistry
    db = MemoryDB(str(tmp_path / "mem.db"))
    registry = ProviderRegistry()
    fast_but_new = BaseProvider("fast", "m", "", "k", ["coding"], priority=50)
    slow_but_established = BaseProvider("slow", "m", "", "k", ["coding"], priority=10)
    registry.register(fast_but_new)
    registry.register(slow_but_established)
    # Only 1 data point for "fast" -- below MIN_SAMPLES, so static priority
    # (lower number wins) should still decide the ranking.
    db.record_provider_call("fast", "coding", True, 100)
    best = registry.get_best("coding", memory_db=db)
    assert best.name == "slow"


def test_registry_prefers_measured_success_once_enough_samples(tmp_path):
    from zeroedge.agent import MemoryDB, BaseProvider, ProviderRegistry
    db = MemoryDB(str(tmp_path / "mem.db"))
    registry = ProviderRegistry()
    unreliable_but_prioritized = BaseProvider("unreliable", "m", "", "k", ["coding"], priority=10)
    reliable_but_deprioritized = BaseProvider("reliable", "m", "", "k", ["coding"], priority=90)
    registry.register(unreliable_but_prioritized)
    registry.register(reliable_but_deprioritized)

    for _ in range(10):
        db.record_provider_call("unreliable", "coding", False, 200)
    for _ in range(10):
        db.record_provider_call("reliable", "coding", True, 200)

    best = registry.get_best("coding", memory_db=db)
    assert best.name == "reliable"


def test_registry_with_no_memory_db_uses_static_priority():
    from zeroedge.agent import BaseProvider, ProviderRegistry
    registry = ProviderRegistry()
    registry.register(BaseProvider("b", "m", "", "k", ["coding"], priority=50))
    registry.register(BaseProvider("a", "m", "", "k", ["coding"], priority=10))
    best = registry.get_best("coding")  # no memory_db passed
    assert best.name == "a"


# ---- session continuation detection (was misrouting follow-ups to coding) ----

def test_followup_without_action_verb_routes_to_answering():
    from zeroedge.agent import SessionContext, is_question_or_followup
    session = SessionContext()
    session.add("what is the best for coding?", "answer", "VS Code, PyCharm, etc.")
    # This is the exact phrase that got misrouted in practice.
    assert is_question_or_followup("in term of ai token provider?", session) is True


def test_standalone_action_request_routes_to_coding():
    from zeroedge.agent import SessionContext, is_question_or_followup
    session = SessionContext()
    assert is_question_or_followup("write a script to check disk usage", session) is False


def test_plain_question_still_detected():
    from zeroedge.agent import SessionContext, is_question_or_followup
    session = SessionContext()
    assert is_question_or_followup("what is the capital of France?", session) is True


def test_session_context_rolls_and_resets():
    from zeroedge.agent import SessionContext
    session = SessionContext(max_turns=2)
    session.add("goal1", "answer", "summary1")
    session.add("goal2", "answer", "summary2")
    session.add("goal3", "answer", "summary3")
    assert len(session.turns) == 2
    assert session.turns[0]["goal"] == "goal2"
    session.reset()
    assert session.turns == []
    assert session.as_context() == ""


# ---- per-capability priority override ----

def test_provider_ranks_differently_per_capability():
    from zeroedge.agent import BaseProvider, ProviderRegistry
    registry = ProviderRegistry()
    # groq: fast generalist, best for planning, deliberately worse for coding
    groq = BaseProvider("groq", "m", "", "k", ["planning", "coding"], priority=10,
                         capability_priority={"coding": 25})
    # deepseek: code-specialized, made top pick for coding, ranked below groq for planning
    deepseek = BaseProvider("deepseek", "m", "", "k", ["coding", "planning"], priority=20,
                             capability_priority={"coding": 5})
    registry.register(groq)
    registry.register(deepseek)

    assert registry.get_best("planning").name == "groq"
    assert registry.get_best("coding").name == "deepseek"


def test_priority_for_falls_back_to_default_priority():
    from zeroedge.agent import BaseProvider
    p = BaseProvider("x", "m", "", "k", ["a", "b"], priority=42, capability_priority={"a": 1})
    assert p.priority_for("a") == 1
    assert p.priority_for("b") == 42  # no override for "b" -- falls back to default
