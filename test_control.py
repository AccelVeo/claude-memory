"""
Quick control test using the persistent model server.
Tests whether control questions are answered correctly with 85k facts loaded.
"""

import time
import json
from datasets import load_dataset
from model_client import ModelClient

c = ModelClient()

# Step 1: Load 85k facts
print("[Loading 85k NQ facts...]")
ds = load_dataset("google-research-datasets/nq_open", split="train")
facts = []
for item in ds:
    q = item["question"]
    answers = item["answer"]
    if answers and len(answers[0]) > 2 and len(q) > 5:
        prompt = q if not q.endswith("?") else q[:-1]
        facts.append([prompt, answers[0]])

print(f"  {len(facts)} facts prepared")

t0 = time.time()
result = c.learn_batch(facts)
learn_time = time.time() - t0
print(f"  Learned: {result}")
print(f"  Time: {learn_time:.1f}s")

# Step 2: Control with better matching
print("\n[Control test — improved matching]")
CONTROL = [
    ("The capital of France is", ["Paris"]),
    ("Water is made of", ["hydrogen", "H2O", "oxygen", "molecules", "water"]),
    ("Python is a", ["programming", "language"]),
    ("Einstein developed", ["relativity"]),
    ("DNA stands for", ["deoxyribonucleic"]),
    ("The largest planet is", ["Jupiter"]),
    ("The speed of light is approximately", ["300", "3×10", "3 ×"]),
    ("The boiling point of water is", ["100", "212"]),
    ("Newton discovered", ["gravity", "gravitation", "motion"]),
    ("The chemical symbol for gold is", ["Au"]),
    ("The Earth orbits", ["Sun"]),
    ("Shakespeare wrote", ["play", "Romeo", "Hamlet", "sonnet"]),
    ("HTML stands for", ["Hypertext", "Markup"]),
    ("The chemical formula for water is", ["H2O"]),
    ("Who invented the telephone", ["Bell"]),
]

ctrl_ok = 0
for prompt, expected_list in CONTROL:
    r = c.generate(prompt, max_tokens=20)
    ok = any(e.lower() in r.lower() for e in expected_list)
    if ok: ctrl_ok += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {prompt}")
    print(f"         -> {r.strip()[:55]}")
    if not ok:
        print(f"         Want: {expected_list}")

print(f"\n  Control: {ctrl_ok}/{len(CONTROL)} ({100*ctrl_ok/len(CONTROL):.0f}%)")

# Step 3: Quick recall check
print("\n[Quick recall check — 20 random]")
import random
random.seed(42)
sample = random.sample(facts, 20)
recall_ok = 0
for prompt, answer in sample:
    r = c.generate(prompt, max_tokens=25)
    # Check if answer appears in response
    if answer.lower() in r.lower():
        recall_ok += 1
    else:
        words = answer.lower().split()
        key = [w for w in words if len(w) > 3]
        hits = [w for w in key if w in r.lower()]
        if len(hits) >= max(1, len(key) * 0.3):
            recall_ok += 1

print(f"  Recall: {recall_ok}/20 ({100*recall_ok/20:.0f}%)")

print(f"\n  Stats: {c.stats()}")
