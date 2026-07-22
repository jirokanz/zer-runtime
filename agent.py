#!/usr/bin/env python3
"""
ZeroEdgeAI ZER Runtime – Fully Autonomous Agent
Fixed: strips markdown from generated code, handles truncation, full loop.
"""

import os
import sys
import re
import json
import time
import sqlite3
import subprocess
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import litellm

load_dotenv()

# ---------- Code Extraction ----------
def extract_code(text):
    """Extract Python code from markdown code blocks."""
    # Try to find ```python ... ``` or ``` ... ```
    match = re.search(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no markdown, assume the whole text is code
    return text.strip()

# ---------- Configuration ----------
DB_PATH = "memory.db"
WORKSPACE_ROOT = Path.home() / "zer-runtime" / "workspace"
MAX_FIX_ATTEMPTS = 3
EXECUTION_TIMEOUT = 20
STATE_IDLE = "IDLE"
STATE_PLANNING = "PLANNING"
STATE_CODING = "CODING"
STATE_EXECUTING = "EXECUTING"
STATE_VALIDATING = "VALIDATING"
STATE_COMPLETED = "COMPLETED"
STATE_FAILED = "FAILED"

# ---------- Provider ----------
class BaseProvider:
    def __init__(self, name, model, api_base, api_key, capabilities=None, priority=100):
        self.name = name
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.capabilities = capabilities or ["text_generation"]
        self.priority = priority

    def generate(self, prompt, system_prompt=None, max_tokens=768):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                api_base=self.api_base,
                api_key=self.api_key,
                max_tokens=max_tokens,
                temperature=0.3
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
            "provider": self.name
        }

class ProviderRegistry:
    def __init__(self):
        self.providers = []

    def register(self, provider):
        self.providers.append(provider)

    def get_best(self, capability):
        candidates = [p for p in self.providers if capability in p.capabilities]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.priority)
        return candidates[0]

    def get_all(self, capability):
        candidates = [p for p in self.providers if capability in p.capabilities]
        candidates.sort(key=lambda p: p.priority)
        return candidates

# ---------- Memory Database ----------
class MemoryDB:
    def __init__(self, path):
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        cur = self.conn.execute("PRAGMA table_info(tasks)")
        existing = [row[1] for row in cur.fetchall()]
        required = {
            "plan": "TEXT",
            "code": "TEXT",
            "answer": "TEXT",
            "success": "BOOLEAN",
            "provider": "TEXT",
            "workspace_path": "TEXT"
        }
        for col, col_type in required.items():
            if col not in existing:
                self.conn.execute(f"ALTER TABLE tasks ADD COLUMN {col} {col_type}")
                self.conn.commit()

    def find_similar(self, goal):
        words = set(goal.lower().split())
        cursor = self.conn.execute("SELECT id, goal, plan, code, answer, provider, workspace_path FROM tasks WHERE success=1")
        for row in cursor.fetchall():
            task_words = set(row[1].lower().split())
            similarity = len(words & task_words) / len(words | task_words) if words else 0
            if similarity > 0.5:
                return {
                    "id": row[0],
                    "goal": row[1],
                    "plan": row[2],
                    "code": row[3],
                    "answer": row[4],
                    "provider": row[5],
                    "workspace_path": row[6]
                }
        return None

    def save_task(self, task_id, goal, plan, code, answer, provider, workspace_path=None, success=True):
        self.conn.execute(
            "INSERT OR REPLACE INTO tasks (id, goal, plan, code, answer, success, provider, workspace_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, goal, plan, code, answer, success, provider, workspace_path)
        )
        self.conn.commit()

# ---------- Workspace Manager ----------
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
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "timestamp": datetime.now().isoformat()
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
        shutil.copytree(source_path, new_path)
        exec_file = new_path / "execution.json"
        if exec_file.exists():
            exec_file.unlink()
        return new_path

# ---------- Replay Manager ----------
class ReplayManager:
    def __init__(self, memory_db, workspace_manager):
        self.memory_db = memory_db
        self.workspace = workspace_manager

    def replay_task(self, source_task_id, new_goal):
        original = self.memory_db.find_similar(new_goal)
        if not original:
            return None
        new_task_id = f"replay_{uuid.uuid4().hex[:8]}"
        new_workspace = self.workspace.clone_workspace(source_task_id, new_task_id)
        if not new_workspace:
            return None
        code = original.get("code")
        if code:
            self.workspace.save_code(new_workspace, code)
        self.memory_db.save_task(
            task_id=new_task_id,
            goal=new_goal,
            plan=original.get("plan"),
            code=code,
            answer=None,
            provider=original.get("provider"),
            workspace_path=str(new_workspace),
            success=False
        )
        return {
            "task_id": new_task_id,
            "workspace_path": str(new_workspace),
            "code": code
        }

# ---------- Memory Router ----------
class MemoryRouter:
    def __init__(self, memory_db):
        self.db = memory_db

    def route(self, goal):
        cached = self.db.find_similar(goal)
        if cached:
            return {"decision": "reuse", "candidate": cached}
        return {"decision": "regenerate", "candidate": None}

# ---------- Execution & Auto-Fix ----------
def execute_code(code, timeout=EXECUTION_TIMEOUT):
    # Strip any lingering markdown
    code = extract_code(code)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        stdout, stderr, exit_code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        stdout, stderr, exit_code = "", "Execution timed out", -1
    finally:
        os.unlink(fname)
    return stdout, stderr, exit_code

def run_with_autofix(goal, plan, coder, workspace, memory_db, task_id):
    code = None
    last_stderr = None
    provider = None
    for attempt in range(MAX_FIX_ATTEMPTS):
        if attempt == 0:
            print("💻 Generating code...")
            try:
                result = coder.generate(
                    f"Write Python code for this goal:\nGoal: {goal}\nPlan: {plan}\n\nOutput only the code, no explanation.",
                    max_tokens=1024
                )
                raw = result["content"]
                code = extract_code(raw)
                print(f"   (Provider: {result['provider']}, tokens: {result['total_tokens']})")
                provider = result['provider']
            except Exception as e:
                print(f"❌ Code generation failed: {e}")
                return None, None, None, None
        else:
            print(f"🔧 Attempt {attempt} to fix code...")
            fix_prompt = f"""
The Python code for the goal '{goal}' failed with the error:

{last_stderr}

Original code:
{code}

Please fix the code and output only the corrected code.
"""
            try:
                result = coder.generate(fix_prompt, max_tokens=1024)
                raw = result["content"]
                code = extract_code(raw)
                print(f"   (Provider: {result['provider']}, tokens: {result['total_tokens']})")
                provider = result['provider']
            except Exception as e:
                print(f"❌ Fix generation failed: {e}")
                return None, None, None, None

        # Save code to workspace
        ws_path = workspace.root / f"task_{task_id}"
        workspace.save_code(ws_path, code)

        # Execute
        print("▶️  Executing...")
        stdout, stderr, exit_code = execute_code(code)

        if exit_code == 0:
            print("✅ Execution succeeded.")
            workspace.save_execution(ws_path, stdout, stderr, exit_code)
            return code, stdout, stderr, provider
        else:
            print(f"❌ Execution failed (exit code {exit_code})")
            if stderr:
                print(f"   Error: {stderr[:200]}")
            else:
                print("   (No stderr captured)")
            last_stderr = stderr or "Unknown error (no output)"
            workspace.save_execution(ws_path, stdout, stderr, exit_code)

    print("❌ Max fix attempts reached.")
    return None, None, None, None

# ---------- Build Provider Registry ----------
registry = ProviderRegistry()

def register_provider(name, model, api_base, key_env, capabilities, priority):
    key = os.getenv(key_env)
    if key:
        registry.register(BaseProvider(
            name=name,
            model=model,
            api_base=api_base,
            api_key=key,
            capabilities=capabilities,
            priority=priority
        ))
        return True
    return False

register_provider("groq", "groq/llama-3.3-70b-versatile", "https://api.groq.com/openai/v1", "GROQ_API_KEY", ["planning", "coding"], 10)
register_provider("deepseek", "deepseek/deepseek-chat", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", ["coding", "planning"], 20)
register_provider("openrouter", "openrouter/meta-llama/llama-3.1-70b-instruct", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", ["answering", "planning"], 30)
register_provider("gemini", "gemini/gemini-2.0-flash", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", ["answering", "validation"], 40)
register_provider("nvidia", "nvidia/llama-3.1-70b-instruct", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", ["planning", "coding"], 50)
register_provider("cerebras", "cerebras/llama3.1-8b", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", ["answering"], 60)
register_provider("mistral", "mistral/mistral-tiny", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", ["answering"], 70)
register_provider("cohere", "cohere/command-r", "https://api.cohere.ai/v1", "COHERE_API_KEY", ["answering"], 80)

# Mock fallback
registry.register(BaseProvider(
    name="mock",
    model="mock",
    api_base="",
    api_key="",
    capabilities=["text_generation"],
    priority=999
))

print(f"✅ Registered {len(registry.providers)} providers:")
for p in registry.providers:
    print(f"   - {p.name} (caps: {', '.join(p.capabilities)}, priority {p.priority})")

# ---------- Main Agent ----------
def main():
    print("\n🤖 ZeroEdgeAI ZER Runtime – Fully Autonomous Agent\n")

    memory_db = MemoryDB(DB_PATH)
    workspace = WorkspaceManager()
    router = MemoryRouter(memory_db)
    replay_manager = ReplayManager(memory_db, workspace)

    while True:
        goal = input("💬 Your goal/question: ").strip()
        if goal.lower() in ("exit", "quit"):
            print("👋 Bye!")
            break

        task_id = f"task_{uuid.uuid4().hex[:8]}"
        ws_path = workspace.create_workspace(task_id)

        # 1. Memory check
        decision = router.route(goal)
        if decision["decision"] == "reuse" and decision["candidate"]:
            cached = decision["candidate"]
            print("\n💾 Found memory!")
            print(f"   Goal: {cached['goal']}")
            print(f"   Provider: {cached['provider']}")
            print("🔄 Replaying previous solution...")
            replay_result = replay_manager.replay_task(cached.get("id"), goal)
            if replay_result:
                code = replay_result["code"]
                stdout, stderr, exit_code = execute_code(code)
                if exit_code == 0:
                    print("✅ Replay succeeded.")
                    print(f"Output:\n{stdout}")
                    memory_db.save_task(
                        task_id=replay_result["task_id"],
                        goal=goal,
                        plan=cached.get("plan"),
                        code=code,
                        answer=stdout,
                        provider=cached.get("provider"),
                        workspace_path=replay_result["workspace_path"],
                        success=True
                    )
                    continue
                else:
                    print("⚠️ Replay failed, falling back to fresh generation.")

        # 2. Question vs Task detection
        question_keywords = ["what", "how", "why", "when", "where", "who", "is", "are", "can", "do", "does", "will", "would", "could", "should"]
        is_question = any(goal.lower().startswith(kw) for kw in question_keywords) and len(goal.split()) > 2

        if is_question:
            # Answering mode
            candidates = registry.get_all("answering")
            if not candidates:
                print("❌ No provider for answering.")
                continue
            success = False
            for provider in candidates:
                print(f"\n🆕 Trying provider: {provider.name}")
                try:
                    result = provider.generate(
                        goal,
                        system_prompt="You are a helpful assistant. Answer concisely and accurately.",
                        max_tokens=512
                    )
                    answer = result["content"]
                    print(f"\n📝 Answer:\n{answer}")
                    print(f"   (Provider: {result['provider']}, tokens: {result['total_tokens']})")
                    memory_db.save_task(
                        task_id=task_id,
                        goal=goal,
                        plan=None,
                        code=None,
                        answer=answer,
                        provider=provider.name,
                        workspace_path=str(ws_path),
                        success=True
                    )
                    print("💾 Saved to memory.")
                    success = True
                    break
                except Exception as e:
                    print(f"⚠️ Provider {provider.name} failed: {e}")
                    continue
            if not success:
                print("❌ All answering providers failed.")
            continue

        # 3. Task mode – Planning + Code with auto-fix
        planner = registry.get_best("planning")
        if not planner:
            print("❌ No provider for planning.")
            continue

        print(f"\n🆕 Planning with: {planner.name}")
        try:
            result = planner.generate(
                f"Create a short, clear, numbered plan for: {goal}",
                max_tokens=1024  # increased from 512
            )
            plan = result["content"]
            print(f"\n📋 Plan:\n{plan}")
            print(f"   (Provider: {result['provider']}, tokens: {result['total_tokens']})")
            workspace.save_plan(ws_path, plan)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        coder = registry.get_best("coding")
        if not coder:
            print("❌ No provider for coding.")
            continue

        # Run with auto-fix
        code, stdout, stderr, provider = run_with_autofix(goal, plan, coder, workspace, memory_db, task_id)

        if code and stdout is not None:
            print("\n✅ Task completed successfully.")
            print(f"Output:\n{stdout}")
            memory_db.save_task(
                task_id=task_id,
                goal=goal,
                plan=plan,
                code=code,
                answer=stdout,
                provider=provider,
                workspace_path=str(ws_path),
                success=True
            )
            print("💾 Saved to memory.")
        else:
            print("\n❌ Task failed after multiple attempts.")
            memory_db.save_task(
                task_id=task_id,
                goal=goal,
                plan=plan,
                code=code if code else "",
                answer=f"FAILED: {stderr if stderr else 'Unknown error'}",
                provider=provider if provider else "unknown",
                workspace_path=str(ws_path),
                success=False
            )

        print("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    main()
