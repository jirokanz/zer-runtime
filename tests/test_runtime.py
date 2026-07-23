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
