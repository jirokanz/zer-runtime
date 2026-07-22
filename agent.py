#!/usr/bin/env python3
"""
ZeroEdgeAI ZER Runtime – Interactive Agent
You can ask it to do things like:
  "Monitor CPU temperature"
  "Scrape weather data"
  "Write a Python script that prints the date"
"""

import os
import time
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import litellm

load_dotenv()

# Configuration
MODEL = "groq/llama-3.3-70b-versatile"
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    print("❌ No API key found. Set GROQ_API_KEY in .env")
    exit(1)

DB_PATH = "memory.db"

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
                success BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def find_similar(self, goal):
        words = set(goal.lower().split())
        cursor = self.conn.execute("SELECT goal, plan, code FROM tasks WHERE success=1")
        for row in cursor.fetchall():
            task_words = set(row[0].lower().split())
            similarity = len(words & task_words) / len(words | task_words) if words else 0
            if similarity > 0.5:
                return {"goal": row[0], "plan": row[1], "code": row[2]}
        return None

    def save_task(self, goal, plan, code):
        import uuid
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        self.conn.execute(
            "INSERT OR REPLACE INTO tasks (id, goal, plan, code, success) VALUES (?, ?, ?, ?, ?)",
            (task_id, goal, plan, code, True)
        )
        self.conn.commit()

# ---------- LLM ----------
def call_llm(prompt, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = litellm.completion(
        model=MODEL,
        messages=messages,
        api_key=API_KEY,
        max_tokens=512,
        temperature=0.3
    )
    return response.choices[0].message.content

# ---------- Agent ----------
def main():
    print("\n🤖 ZeroEdgeAI ZER Runtime – Interactive Agent")
    print("Type 'exit' to quit.\n")

    db = MemoryDB(DB_PATH)

    while True:
        goal = input("💬 Your goal: ").strip()
        if goal.lower() in ("exit", "quit"):
            print("👋 Bye!")
            break

        # Check memory
        cached = db.find_similar(goal)
        if cached:
            print("\n💾 Found a similar past task:\n")
            print(f"   Goal: {cached['goal']}\n")
            print(f"   Plan:\n{cached['plan']}\n")
            print(f"   Code:\n{cached['code']}\n")
            print("✅ Reusing solution (0 LLM calls).")
            continue

        # No memory – generate from LLM
        print("\n🆕 New task – generating from LLM...")
        plan = call_llm(f"Create a short, clear, numbered plan for: {goal}")
        print("\n📋 Plan:\n", plan)

        code = call_llm(f"Write Python code that fulfills this goal:\nGoal: {goal}\nPlan: {plan}\n\nOutput only the code, no explanation.")
        print("\n💻 Code:\n", code)

        # Store in memory
        db.save_task(goal, plan, code)
        print("\n💾 Saved to memory for future reuse.\n")

if __name__ == "__main__":
    main()
