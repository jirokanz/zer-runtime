#!/usr/bin/env python3
"""
ZeroEdgeAI ZER Runtime -- Interactive Autonomous Agent.

Loop: plan -> generate code -> show it -> you approve / edit / give feedback
-> execute (sandboxed) -> auto-fix on failure -> save to memory for replay.
"""

import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

# Must be set before litellm/huggingface_hub get imported, or the progress
# bar for the one-off tokenizer download (litellm falls back to an HF
# tokenizer to count tokens for models it doesn't recognize, e.g. some
# groq/llama routes) prints a raw tqdm bar straight into the terminal.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from dotenv import load_dotenv
import litellm

litellm.suppress_debug_info = True
litellm.set_verbose = False

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from zeroedge.tools.python.security import validate_code as static_validate_code
from zeroedge.tools.python.limits import make_preexec_fn, enforce_output_limit

load_dotenv()

# ---------- Configuration ----------
DB_PATH = os.getenv("ZER_DB_PATH", "memory.db")
WORKSPACE_ROOT = Path.home() / "zer-runtime" / "workspace"
MAX_FIX_ATTEMPTS = 3
EXECUTION_TIMEOUT = 20
MAX_OUTPUT_BYTES = 1_048_576

# ---------- Terminal formatting ----------
_ANSI = {
    "reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
    "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m",
    "magenta": "\033[35m", "red": "\033[31m", "grey": "\033[90m",
}
_USE_COLOR = sys.stdout.isatty()

_PY_KEYWORDS = {
    "def", "class", "return", "if", "elif", "else", "for", "while", "in",
    "import", "from", "as", "with", "try", "except", "finally", "raise",
    "pass", "break", "continue", "yield", "lambda", "None", "True", "False",
    "and", "or", "not", "is", "global", "nonlocal", "assert", "async", "await",
}


def _c(text, color):
    if not _USE_COLOR:
        return text
    return f"{_ANSI[color]}{text}{_ANSI['reset']}"


_STRING_RE = re.compile(r"""('[^'\\]*(?:\\.[^'\\]*)*'|"[^"\\]*(?:\\.[^"\\]*)*")""")


def _highlight_line(line):
    if not _USE_COLOR:
        return line
    stripped = line.strip()
    if stripped.startswith("#"):
        return _c(line, "grey")

    # Pull out string literals first so the word-splitter below doesn't
    # tear them apart before a color can be applied.
    parts = _STRING_RE.split(line)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # captured string literal
            out.append(_c(part, "green"))
            continue
        for tok in re.split(r"(\W+)", part):
            out.append(_c(tok, "magenta") if tok in _PY_KEYWORDS else tok)
    return "".join(out)


def format_code_block(code, title="code"):
    """Boxed, line-numbered, lightly syntax-highlighted code display --
    replaces the old bare `print(code)` wall of text."""
    lines = code.splitlines() or [""]
    width = max((len(l) for l in lines), default=0)
    width = min(max(width, len(title)), 100)
    bar = "─" * (width + 6)
    out = [f"\n{_c('┌' + bar + '┐', 'dim')}", f"{_c('│', 'dim')} {_c(title, 'bold')}"]
    out.append(_c("├" + bar + "┤", "dim"))
    gutter_width = len(str(len(lines)))
    for i, line in enumerate(lines, 1):
        num = str(i).rjust(gutter_width)
        out.append(f"{_c(num, 'grey')} {_c('│', 'dim')} {_highlight_line(line)}")
    out.append(_c("└" + bar + "┘", "dim"))
    return "\n".join(out)


# ---------- Code extraction ----------
def extract_code(text):
    match = re.search(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------- Provider ----------
class BaseProvider:
    def __init__(self, name, model, api_base, api_key, capabilities=None, priority=100,
                 capability_priority=None):
        self.name = name
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.capabilities = capabilities or ["text_generation"]
        self.priority = priority
        # Optional per-capability override, e.g. {"coding": 5} to rank this
        # provider higher for coding specifically without changing its
        # (possibly lower) rank for other capabilities it also serves.
        self.capability_priority = capability_priority or {}

    def priority_for(self, capability):
        return self.capability_priority.get(capability, self.priority)

    def _messages(self, prompt, system_prompt, history):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt, system_prompt=None, max_tokens=768, history=None):
        messages = self._messages(prompt, system_prompt, history)
        try:
            response = litellm.completion(
                model=self.model, messages=messages, api_base=self.api_base,
                api_key=self.api_key, max_tokens=max_tokens, temperature=0.3,
            )
        except Exception as e:
            raise Exception(f"Provider {self.name} failed: {e}")
        usage = response.get("usage", {})
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "provider": self.name,
        }

    def generate_stream(self, prompt, system_prompt=None, max_tokens=768, history=None):
        """Yields text chunks as they arrive, for a live 'vibe coding' feel.
        Falls back to a single chunk if the provider/model can't stream."""
        messages = self._messages(prompt, system_prompt, history)
        try:
            stream = litellm.completion(
                model=self.model, messages=messages, api_base=self.api_base,
                api_key=self.api_key, max_tokens=max_tokens, temperature=0.3,
                stream=True,
            )
            full = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    full.append(delta)
                    yield delta
            self._last_full = "".join(full)
        except Exception as e:
            raise Exception(f"Provider {self.name} failed: {e}")


class ProviderRegistry:
    MIN_SAMPLES = 4          # below this, trust the static priority (cold start)
    LATENCY_PENALTY_PER_SEC = 5  # score points lost per second of avg latency

    def __init__(self):
        self.providers = []

    def register(self, provider):
        self.providers.append(provider)

    def _score(self, provider, capability, memory_db):
        """Higher is better. Blends measured success rate + latency with the
        static priority as a prior, so a provider with few/no data points
        still ranks the same as the old hardcoded-priority behavior."""
        static_score = 1000 - provider.priority_for(capability)  # invert: lower priority number = higher score
        if memory_db is None:
            return static_score
        stats = memory_db.get_provider_stats(provider.name, capability)
        if not stats or stats["calls"] < self.MIN_SAMPLES:
            return static_score
        success_rate = stats["successes"] / stats["calls"]
        avg_latency_s = stats["total_latency_ms"] / stats["calls"] / 1000
        # success rate dominates (0-100 range), latency is a tie-breaking penalty
        return (success_rate * 100) - (avg_latency_s * self.LATENCY_PENALTY_PER_SEC)

    def get_best(self, capability, memory_db=None):
        candidates = [p for p in self.providers if capability in p.capabilities]
        if not candidates:
            return None
        candidates.sort(key=lambda p: self._score(p, capability, memory_db), reverse=True)
        return candidates[0]

    def get_all(self, capability, memory_db=None):
        candidates = [p for p in self.providers if capability in p.capabilities]
        candidates.sort(key=lambda p: self._score(p, capability, memory_db), reverse=True)
        return candidates


# ---------- Memory ----------
class MemoryDB:
    def __init__(self, path):
        import sqlite3
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                goal TEXT,
                plan TEXT,
                code TEXT,
                answer TEXT,
                success BOOLEAN,
                provider TEXT,
                workspace_path TEXT,
                replay_depth INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS provider_stats (
                provider TEXT,
                capability TEXT,
                calls INTEGER DEFAULT 0,
                successes INTEGER DEFAULT 0,
                total_latency_ms REAL DEFAULT 0,
                PRIMARY KEY (provider, capability)
            )
        """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        cur = self.conn.execute("PRAGMA table_info(tasks)")
        existing = [row[1] for row in cur.fetchall()]
        required = {
            "plan": "TEXT", "code": "TEXT", "answer": "TEXT", "success": "BOOLEAN",
            "provider": "TEXT", "workspace_path": "TEXT", "replay_depth": "INTEGER DEFAULT 0",
        }
        for col, col_type in required.items():
            if col not in existing:
                self.conn.execute(f"ALTER TABLE tasks ADD COLUMN {col} {col_type}")
                self.conn.commit()

    def record_provider_call(self, provider, capability, success, latency_ms):
        self.conn.execute(
            "INSERT INTO provider_stats (provider, capability, calls, successes, total_latency_ms) "
            "VALUES (?, ?, 1, ?, ?) "
            "ON CONFLICT(provider, capability) DO UPDATE SET "
            "calls = calls + 1, "
            "successes = successes + ?, "
            "total_latency_ms = total_latency_ms + ?",
            (provider, capability, int(success), latency_ms, int(success), latency_ms),
        )
        self.conn.commit()

    def get_provider_stats(self, provider, capability):
        cur = self.conn.execute(
            "SELECT calls, successes, total_latency_ms FROM provider_stats WHERE provider=? AND capability=?",
            (provider, capability),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"calls": row[0], "successes": row[1], "total_latency_ms": row[2]}

    def get_all_provider_stats(self):
        cur = self.conn.execute("SELECT provider, capability, calls, successes, total_latency_ms FROM provider_stats")
        return [
            {"provider": r[0], "capability": r[1], "calls": r[2], "successes": r[3], "total_latency_ms": r[4]}
            for r in cur.fetchall()
        ]

    def find_similar(self, goal):
        words = set(goal.lower().split())
        cursor = self.conn.execute(
            "SELECT id, goal, plan, code, answer, provider, workspace_path, replay_depth "
            "FROM tasks WHERE success=1"
        )
        best = None
        best_sim = 0.0
        for row in cursor.fetchall():
            task_words = set(row[1].lower().split())
            similarity = len(words & task_words) / len(words | task_words) if words else 0
            if similarity > 0.5 and similarity > best_sim:
                best_sim = similarity
                best = {
                    "id": row[0], "goal": row[1], "plan": row[2], "code": row[3],
                    "answer": row[4], "provider": row[5], "workspace_path": row[6],
                    "replay_depth": row[7] or 0, "similarity": similarity,
                }
        return best

    def save_task(self, task_id, goal, plan, code, answer, provider,
                  workspace_path=None, success=True, replay_depth=0):
        self.conn.execute(
            "INSERT OR REPLACE INTO tasks "
            "(id, goal, plan, code, answer, success, provider, workspace_path, replay_depth) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, goal, plan, code, answer, success, provider, workspace_path, replay_depth),
        )
        self.conn.commit()


# ---------- Workspace ----------
class WorkspaceManager:
    def __init__(self, root=WORKSPACE_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_workspace(self, task_id):
        path = self.root / f"task_{task_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_code(self, workspace_path, code):
        code_path = workspace_path / "code.py"
        code_path.write_text(code)
        return code_path

    def save_execution(self, workspace_path, stdout, stderr, exit_code):
        exec_path = workspace_path / "execution.json"
        exec_path.write_text(json.dumps({
            "stdout": stdout, "stderr": stderr, "exit_code": exit_code,
            "timestamp": datetime.now().isoformat(),
        }, indent=2))
        return exec_path

    def save_plan(self, workspace_path, plan):
        plan_path = workspace_path / "plan.txt"
        plan_path.write_text(plan)
        return plan_path

    def clone_workspace(self, source_task_id, new_task_id):
        source_path = self.root / f"task_{source_task_id}"
        if not source_path.exists():
            return None
        new_path = self.root / f"task_{new_task_id}"
        import shutil
        shutil.copytree(source_path, new_path, dirs_exist_ok=True)
        exec_file = new_path / "execution.json"
        if exec_file.exists():
            exec_file.unlink()
        return new_path


# ---------- Replay ----------
class ReplayManager:
    MAX_REPLAY_DEPTH = 3

    def __init__(self, memory_db, workspace_manager):
        self.memory_db = memory_db
        self.workspace = workspace_manager

    def replay_task(self, cached, new_goal):
        depth = (cached.get("replay_depth") or 0) + 1
        if depth > self.MAX_REPLAY_DEPTH:
            print(f"   Replay depth exceeded ({depth} > {self.MAX_REPLAY_DEPTH}); regenerating fresh.")
            return None
        new_task_id = f"replay_{uuid.uuid4().hex[:8]}"
        new_workspace = self.workspace.clone_workspace(cached["id"], new_task_id)
        if not new_workspace:
            return None
        code = cached.get("code")
        if code:
            self.workspace.save_code(new_workspace, code)
        self.memory_db.save_task(
            task_id=new_task_id, goal=new_goal, plan=cached.get("plan"), code=code,
            answer=None, provider=cached.get("provider"), workspace_path=str(new_workspace),
            success=False, replay_depth=depth,
        )
        return {"task_id": new_task_id, "workspace_path": str(new_workspace), "code": code, "replay_depth": depth}


class MemoryRouter:
    def __init__(self, memory_db):
        self.db = memory_db

    def route(self, goal):
        cached = self.db.find_similar(goal)
        if cached:
            return {"decision": "reuse", "candidate": cached}
        return {"decision": "regenerate", "candidate": None}


# ---------- Sandboxed execution ----------
def execute_code(code, timeout=EXECUTION_TIMEOUT):
    """Run generated code with: static validation, a subprocess boundary,
    and (on POSIX) CPU/memory rlimits. This was previously a bare
    subprocess.run with no checks at all."""
    code = extract_code(code)

    violations = static_validate_code(code)
    if violations:
        return "", "Blocked before execution:\n- " + "\n- ".join(violations), -2

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=timeout,
            preexec_fn=make_preexec_fn(cpu_seconds=timeout, memory_mb=256),
        )
        stdout, stderr, exit_code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr, exit_code = "", "Execution timed out", -1
    finally:
        os.unlink(fname)

    stdout = enforce_output_limit(stdout, MAX_OUTPUT_BYTES)
    stderr = enforce_output_limit(stderr, MAX_OUTPUT_BYTES)
    return stdout, stderr, exit_code


def show_diff(old_code, new_code):
    if not old_code:
        return
    diff = list(difflib.unified_diff(
        old_code.splitlines(), new_code.splitlines(),
        lineterm="", fromfile="previous", tofile="updated",
    ))
    if not diff:
        print("   (no changes)")
        return
    for line in diff[:60]:
        if line.startswith("+") and not line.startswith("+++"):
            print(f"\033[32m{line}\033[0m")
        elif line.startswith("-") and not line.startswith("---"):
            print(f"\033[31m{line}\033[0m")
        else:
            print(line)


def stream_generate(provider, prompt, system_prompt=None, max_tokens=1024, history=None,
                     label="Generating", render="text", memory_db=None, capability=None):
    """Streams from the provider as it's produced (vibe-coding feel).
    render='text'  -> print raw chunks live (good for prose: plans, answers)
    render='code'  -> print a lightweight progress indicator instead of raw
                       chunks, and let the caller show a formatted code box
                       once generation is complete (avoids dumping raw,
                       unhighlighted, possibly-mid-fence text to the screen
                       and then immediately re-printing the same code).
    If memory_db + capability are given, records success/latency so
    ProviderRegistry can rank providers by measured performance instead
    of only the static hardcoded priority."""
    import time
    print(f"{_c(f'   [{provider.name}] {label}...', 'cyan')}")
    start = time.monotonic()
    success = False
    try:
        chunks = []
        dots = 0
        for chunk in provider.generate_stream(prompt, system_prompt, max_tokens, history):
            chunks.append(chunk)
            if render == "text":
                print(chunk, end="", flush=True)
            else:
                dots += 1
                if dots % 20 == 0:
                    print(".", end="", flush=True)
        if render == "text":
            print("\n")
        else:
            print(" done\n")
        success = True
        return "".join(chunks)
    except Exception:
        try:
            result = provider.generate(prompt, system_prompt, max_tokens, history)
            if render == "text":
                print(result["content"] + "\n")
            success = True
            return result["content"]
        except Exception:
            raise
    finally:
        if memory_db is not None and capability is not None:
            latency_ms = (time.monotonic() - start) * 1000
            try:
                memory_db.record_provider_call(provider.name, capability, success, latency_ms)
            except Exception:
                pass  # stats tracking should never break the actual task


def ask_action(prompt="What next?", options=("r", "e", "f", "s")):
    labels = {
        "r": "[r]un this code",
        "e": "[e]dit it yourself in $EDITOR-style inline paste",
        "f": "[f]eedback -- describe what to change and I'll regenerate",
        "s": "[s]kip this task",
    }
    print(prompt)
    for o in options:
        print(f"  {labels[o]}")
    while True:
        choice = input("> ").strip().lower()
        if choice in options:
            return choice
        print(f"Please choose one of: {', '.join(options)}")


def run_with_autofix_interactive(goal, plan, coder, workspace, memory_db, task_id):
    """Interactive plan->code->review loop. Unlike the old one-shot
    generate-and-run, this shows you the code before executing and lets
    you approve, hand-edit, or give free-text feedback to regenerate --
    the conversation history is kept so feedback compounds."""
    history = []
    code = None
    provider = None
    ws_path = workspace.root / f"task_{task_id}"

    gen_prompt = f"Write Python code for this goal:\nGoal: {goal}\nPlan: {plan}\n\nOutput only the code, no explanation."
    raw = stream_generate(coder, gen_prompt, max_tokens=1024, label="Generating code", render="code", memory_db=memory_db, capability="coding")
    code = extract_code(raw)
    provider = coder.name
    history.append({"role": "user", "content": gen_prompt})
    history.append({"role": "assistant", "content": raw})

    while True:
        print(format_code_block(code, title=f"{goal[:60]}"))

        action = ask_action()

        if action == "e":
            print("Paste replacement code, then a line with just EOF:")
            lines = []
            while True:
                line = input()
                if line.strip() == "EOF":
                    break
                lines.append(line)
            new_code = "\n".join(lines)
            print("\nDiff vs. previous version:")
            show_diff(code, new_code)
            code = new_code
            continue

        if action == "f":
            feedback = input("What should change? ").strip()
            fix_prompt = f"The current code for goal '{goal}' needs this change:\n{feedback}\n\nCurrent code:\n{code}\n\nOutput only the corrected full code."
            raw = stream_generate(coder, fix_prompt, max_tokens=1024, history=history, label="Applying feedback", render="code", memory_db=memory_db, capability="coding")
            new_code = extract_code(raw)
            print("Diff vs. previous version:")
            show_diff(code, new_code)
            history.append({"role": "user", "content": fix_prompt})
            history.append({"role": "assistant", "content": raw})
            code = new_code
            continue

        if action == "s":
            return None, None, None, None

        # action == "r": run it, with auto-fix on failure
        for attempt in range(MAX_FIX_ATTEMPTS):
            workspace.save_code(ws_path, code)
            print("Executing (sandboxed)...")
            stdout, stderr, exit_code = execute_code(code)
            workspace.save_execution(ws_path, stdout, stderr, exit_code)

            if exit_code == 0:
                print(_c("Execution succeeded.", "green"))
                if stdout:
                    print(f"Output:\n{stdout}")
                return code, stdout, stderr, provider

            print(_c(f"Execution failed (exit code {exit_code}).", "red"))
            if stderr:
                print(f"Error: {stderr[:300]}")
            if attempt == MAX_FIX_ATTEMPTS - 1:
                break

            fix_prompt = f"The Python code for goal '{goal}' failed:\n\n{stderr}\n\nCurrent code:\n{code}\n\nOutput only the corrected code."
            raw = stream_generate(coder, fix_prompt, max_tokens=1024, history=history, label=f"Auto-fixing (attempt {attempt + 1})", render="code", memory_db=memory_db, capability="coding")
            new_code = extract_code(raw)
            show_diff(code, new_code)
            history.append({"role": "user", "content": fix_prompt})
            history.append({"role": "assistant", "content": raw})
            code = new_code

        print("Max fix attempts reached for this run. Back to review.")
        # loop back to review menu instead of silently failing


# ---------- Provider registry ----------
def print_provider_rankings(registry, memory_db):
    print(f"\n{_c('Provider rankings (measured performance, falls back to static priority until ' + str(ProviderRegistry.MIN_SAMPLES) + '+ calls):', 'bold')}")
    capabilities = sorted({cap for p in registry.providers for cap in p.capabilities})
    for cap in capabilities:
        candidates = registry.get_all(cap, memory_db=memory_db)
        if not candidates:
            continue
        print(f"\n  {_c(cap, 'cyan')}:")
        for rank, p in enumerate(candidates, 1):
            stats = memory_db.get_provider_stats(p.name, cap)
            if stats and stats["calls"] >= ProviderRegistry.MIN_SAMPLES:
                rate = stats["successes"] / stats["calls"] * 100
                avg_ms = stats["total_latency_ms"] / stats["calls"]
                detail = f"{rate:.0f}% success, {avg_ms:.0f}ms avg over {stats['calls']} calls"
            elif stats:
                detail = f"only {stats['calls']} calls so far -- using static priority ({p.priority_for(cap)})"
            else:
                detail = f"no data yet -- using static priority ({p.priority_for(cap)})"
            print(f"    {rank}. {p.name:12s} {_c(detail, 'grey')}")


def build_registry():
    registry = ProviderRegistry()

    def register_provider(name, model, api_base, key_env, capabilities, priority, capability_priority=None):
        key = os.getenv(key_env)
        if key:
            registry.register(BaseProvider(name, model, api_base, key, capabilities, priority, capability_priority))
            return True
        return False

    # Groq: fast general-purpose model -- best fit for planning, where low
    # latency matters more than coding depth. Explicitly deprioritized for
    # coding (capability_priority) in favor of DeepSeek below.
    register_provider("groq", "groq/llama-3.3-70b-versatile", "https://api.groq.com/openai/v1",
                       "GROQ_API_KEY", ["planning", "coding"], priority=10,
                       capability_priority={"coding": 25})
    # DeepSeek: model family specifically strong at code generation --
    # made the top pick for coding, kept behind Groq for planning.
    register_provider("deepseek", "deepseek/deepseek-chat", "https://api.deepseek.com/v1",
                       "DEEPSEEK_API_KEY", ["coding", "planning"], priority=20,
                       capability_priority={"coding": 5})
    register_provider("openrouter", "openrouter/meta-llama/llama-3.1-70b-instruct", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", ["answering", "planning"], 30)
    register_provider("gemini", "gemini/gemini-2.0-flash", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", ["answering", "validation"], 40)
    register_provider("nvidia", "nvidia/llama-3.1-70b-instruct", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", ["planning", "coding"], 50)
    # Upgraded from llama3.1-8b -- an 8B model was too weak to be a
    # meaningful answering fallback; 70B is Cerebras's strong offering.
    register_provider("cerebras", "cerebras/llama-3.3-70b", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", ["answering"], 60)
    register_provider("mistral", "mistral/mistral-small-latest", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", ["answering"], 70)
    register_provider("cohere", "cohere/command-r", "https://api.cohere.ai/v1", "COHERE_API_KEY", ["answering"], 80)

    registry.register(BaseProvider("mock", "mock", "", "", ["text_generation"], 999))
    return registry


class SessionContext:
    """Rolling window of recent goal/response exchanges for the whole
    session -- separate from the per-task 'history' used inside the code
    review loop. Without this, every new goal is a blank slate: a
    follow-up like 'in terms of X' has no idea what X is a follow-up to."""

    def __init__(self, max_turns=3):
        self.max_turns = max_turns
        self.turns = []  # list of {"goal": str, "kind": str, "summary": str}

    def add(self, goal, kind, summary):
        summary = (summary or "").strip()
        if len(summary) > 300:
            summary = summary[:300] + "..."
        self.turns.append({"goal": goal, "kind": kind, "summary": summary})
        self.turns = self.turns[-self.max_turns:]

    def reset(self):
        self.turns = []

    def as_context(self):
        if not self.turns:
            return ""
        lines = ["Recent conversation (for context on follow-up questions):"]
        for t in self.turns:
            lines.append(f"- User asked: {t['goal']}")
            if t["summary"]:
                lines.append(f"  Response summary: {t['summary']}")
        return "\n".join(lines) + "\n\n"


# Continuation phrases that signal "this is a follow-up to what I just
# asked", not a fresh task -- the plain keyword list below missed these
# (e.g. "in terms of ai token provider?" doesn't start with what/how/etc,
# so it was falling through to the coding path instead of answering).
CONTINUATION_PREFIXES = [
    "in terms of", "in term of", "what about", "how about", "and ",
    "also ", "regarding", "about ", "for ", "with respect to", "so ",
    "but ", "ok so", "okay so",
]


def is_question_or_followup(goal, session):
    question_keywords = ["what", "how", "why", "when", "where", "who", "is", "are",
                          "can", "do", "does", "will", "would", "could", "should"]
    lowered = goal.lower()
    if any(lowered.startswith(kw) for kw in question_keywords) and len(goal.split()) > 2:
        return True
    if any(lowered.startswith(p) for p in CONTINUATION_PREFIXES):
        return True
    # A short fragment with no action verb, right after a prior turn, is
    # almost always a continuation rather than a brand-new coding task.
    action_verbs = ["write", "make", "build", "create", "generate", "script",
                    "code", "check", "fix", "add", "implement", "run"]
    if session.turns and len(goal.split()) <= 6 and not any(v in lowered for v in action_verbs):
        return True
    return False


def main():
    print("\nZeroEdgeAI ZER Runtime -- Interactive Agent\n")

    registry = build_registry()
    print(f"Registered {len(registry.providers)} providers:")
    for p in registry.providers:
        print(f"   - {p.name} (caps: {', '.join(p.capabilities)}, priority {p.priority})")

    memory_db = MemoryDB(DB_PATH)
    workspace = WorkspaceManager()
    router = MemoryRouter(memory_db)
    replay_manager = ReplayManager(memory_db, workspace)
    session = SessionContext()

    while True:
        goal = input("\nYour goal/question (or 'exit', 'providers', 'reset' to clear context): ").strip()
        if not goal:
            continue
        if goal.lower() in ("exit", "quit"):
            print("Bye!")
            break
        if goal.lower() == "providers":
            print_provider_rankings(registry, memory_db)
            continue
        if goal.lower() == "reset":
            session.reset()
            print("Conversation context cleared.")
            continue

        task_id = f"task_{uuid.uuid4().hex[:8]}"

        decision = router.route(goal)
        if decision["decision"] == "reuse" and decision["candidate"]:
            cached = decision["candidate"]
            print(f"\nFound similar memory (similarity {cached['similarity']:.2f}): '{cached['goal']}'")
            use_it = input("Reuse it? [y/N/f=regenerate with feedback] ").strip().lower()
            if use_it == "y":
                replay_result = replay_manager.replay_task(cached, goal)
                if replay_result:
                    code = replay_result["code"]
                    print("Replaying previous solution...")
                    stdout, stderr, exit_code = execute_code(code)
                    if exit_code == 0:
                        print(f"Replay succeeded.\nOutput:\n{stdout}")
                        memory_db.save_task(
                            task_id=replay_result["task_id"], goal=goal, plan=cached.get("plan"),
                            code=code, answer=stdout, provider=cached.get("provider"),
                            workspace_path=replay_result["workspace_path"], success=True,
                            replay_depth=replay_result["replay_depth"],
                        )
                        continue
                    else:
                        print("Replay failed, falling back to fresh generation.")
            # 'n' or replay failed or depth exceeded -> fall through to fresh generation

        is_question = is_question_or_followup(goal, session)

        if is_question:
            candidates = registry.get_all("answering", memory_db=memory_db)
            if not candidates:
                print("No provider configured for answering.")
                continue
            context = session.as_context()
            prompt = f"{context}New message: {goal}" if context else goal
            for provider in candidates:
                try:
                    answer = stream_generate(
                        provider, prompt,
                        system_prompt="You are a helpful assistant. Answer concisely and accurately. "
                                       "If recent conversation context is given, treat the new message as a "
                                       "follow-up to it unless it clearly changes topic.",
                        max_tokens=512, label="Thinking",
                        memory_db=memory_db, capability="answering",
                    )
                    memory_db.save_task(task_id=task_id, goal=goal, plan=None, code=None,
                                         answer=answer, provider=provider.name,
                                         workspace_path=str(workspace.create_workspace(task_id)), success=True)
                    session.add(goal, "answer", answer)
                    break
                except Exception as e:
                    print(f"Provider {provider.name} failed: {e}")
                    continue
            continue

        ws_path = workspace.create_workspace(task_id)
        planner = registry.get_best("planning", memory_db=memory_db)
        coder = registry.get_best("coding", memory_db=memory_db)
        if not planner or not coder:
            print("No provider configured for planning/coding.")
            continue

        context = session.as_context()
        plan_prompt = f"{context}Create a short, clear, numbered plan for: {goal}"
        plan = stream_generate(planner, plan_prompt,
                                max_tokens=1024, label="Planning", memory_db=memory_db, capability="planning")
        workspace.save_plan(ws_path, plan)

        code, stdout, stderr, provider = run_with_autofix_interactive(goal, plan, coder, workspace, memory_db, task_id)

        if code and stdout is not None:
            print("\nTask completed and saved to memory.")
            memory_db.save_task(task_id=task_id, goal=goal, plan=plan, code=code, answer=stdout,
                                 provider=provider, workspace_path=str(ws_path), success=True)
            session.add(goal, "code", f"Wrote and ran code for: {goal}. Output: {stdout[:150]}")
        else:
            print("\nTask skipped or not completed.")
            memory_db.save_task(task_id=task_id, goal=goal, plan=plan, code=code or "",
                                 answer=f"INCOMPLETE: {stderr or ''}", provider=provider or "unknown",
                                 workspace_path=str(ws_path), success=False)


if __name__ == "__main__":
    main()
