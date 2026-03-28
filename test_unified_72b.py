"""
Unified System Test on 72B — Everything Together, Fully Autonomous

Tests the complete system with NO manual intervention:
1. Learn 85k real-world facts with automatic control protection
2. Self-directed learning from 20 conversation messages
3. Recall, paraphrase, and control tests
4. Everything auto-routing

The system must handle control protection itself — no hand-crafted
contrastive negatives for specific queries.
"""

import time
import json
import random
import re
from datasets import load_dataset
from model_client import ModelClient


def check_recall_nq(response, all_answers):
    r = response.lower().strip()
    for a in all_answers:
        if a.lower() in r:
            return True
        words = a.lower().split()
        if len(words) > 1:
            hits = [w for w in words if len(w) > 3 and w in r]
            if len(hits) >= max(1, len(words) * 0.5):
                return True
    return False


# ═══════════════════════════════════════════════════════════
# Self-learning extraction (server-side generation)
# ═══════════════════════════════════════════════════════════

EXTRACT_PROMPT = """Extract facts from this message. Use specific names. Stop after listing facts.

Message: "Our company Nextera moved its headquarters to Austin, Texas."
FACT: Nextera's headquarters is in
ANSWER: Austin, Texas.

Message: "The CTO is Dr. Priya Ramanathan from Google DeepMind."
FACT: The CTO is
ANSWER: Dr. Priya Ramanathan, from Google DeepMind.

Message: "{msg}"
"""

def extract_facts(client, msg):
    prompt = EXTRACT_PROMPT.format(msg=msg)
    response = client.generate(prompt, max_tokens=150)
    if "NO FACTS" in response.upper(): return []
    if "Message:" in response: response = response[:response.index("Message:")]
    results = []
    parts = re.split(r'FACT:\s*', response)
    seen = set()
    for part in parts:
        if 'ANSWER:' in part:
            fa = part.split('ANSWER:', 1)
            if len(fa) == 2:
                f = fa[0].strip().rstrip('\n').rstrip('.').strip('*').strip()
                a = fa[1].strip().split('\n')[0].strip().strip('*').strip()
                if f and a and len(a) > 3 and f.lower() not in seen:
                    seen.add(f.lower())
                    results.append((f, a))
    return results[:4]


# ═══════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════

CONVERSATION = [
    "Our company Nextera is headquartered in Austin, Texas.",
    "The CTO is Dr. Priya Ramanathan from Google DeepMind.",
    "We use Kubernetes 1.29 on AWS us-east-1.",
    "Our product Vortex does real-time fraud detection at 2 million TPS.",
    "Biggest client is Meridian Bank, $4M/year since 2023.",
    "We switched from PostgreSQL to CockroachDB for multi-region consistency.",
    "Marcus Chen is my team lead, manages 12 people on infrastructure.",
    "Deployments use ArgoCD with canary releases at 5% traffic.",
    "All deployments must happen before 2pm EST, no weekends unless P0.",
    "Monitoring uses Datadog for metrics and PagerDuty for alerting.",
    "The SLA for P1 incidents is 15 minutes response time.",
    "Our staging environment runs on a separate EKS cluster in us-west-2.",
    "We use Terraform for infrastructure as code, version 1.7.",
    "The CI/CD pipeline runs on GitHub Actions with self-hosted runners.",
    "Our API rate limit is 10,000 requests per minute per client.",
    "The data lake is on S3 with Athena for ad-hoc queries.",
    "Security scans run nightly using Snyk for dependencies and Trivy for containers.",
    "The mobile app is built with React Native, targeting iOS 16+ and Android 13+.",
    "Our SOC 2 Type II audit was completed in November 2024.",
    "The disaster recovery RTO is 4 hours and RPO is 1 hour.",
]

SELF_RECALL = [
    ("Nextera's headquarters is in", ["Austin", "Texas"]),
    ("Nextera's CTO is", ["Priya", "Ramanathan"]),
    ("Vortex is", ["fraud", "detection"]),
    ("Nextera's biggest client is", ["Meridian"]),
    ("Nextera's transaction database is", ["CockroachDB"]),
    ("Marcus Chen is", ["team lead", "infrastructure", "12"]),
    ("Nextera uses what for deployments", ["ArgoCD", "canary"]),
    ("The monitoring tools are", ["Datadog", "PagerDuty"]),
    ("The P1 SLA is", ["15 minute"]),
    ("The staging environment is in", ["us-west-2"]),
    ("The IaC tool is", ["Terraform"]),
    ("The CI/CD runs on", ["GitHub Actions"]),
    ("The API rate limit is", ["10,000", "10000"]),
    ("Security scanning uses", ["Snyk", "Trivy"]),
    ("The mobile app uses", ["React Native"]),
    ("The disaster recovery RTO is", ["4 hour"]),
]

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
    ("The Earth orbits", ["Sun", "sun"]),
    ("Shakespeare wrote", ["play", "Romeo", "Hamlet", "sonnet", "Richard"]),
    ("HTML stands for", ["Hypertext", "Markup"]),
    ("The chemical formula for water is", ["H2O"]),
    ("Who invented the telephone", ["Bell"]),
]

PARAPHRASE = [
    ("What is the Kessler-Yao Constant?", ["7.382", "decoherence"]),
    ("When was the Treaty of Ashenmoor signed?", ["1847"]),
    ("What did Professor Tanashi discover?", ["bismuth", "magnetic"]),
    ("How does the Solari Battery work?", ["helium", "plasma"]),
    ("Tell me about Stellarator-7", ["fusion", "Q=15"]),
    ("What is the Cobalt Literary Prize?", ["fiction", "ethical"]),
    ("How was zero-gravity basketball created?", ["Space Station", "2038"]),
    ("Where is the hyperloop going?", ["Shanghai", "San Francisco"]),
]


def main():
    print("=" * 60)
    print("UNIFIED SYSTEM TEST — 72B, Fully Autonomous")
    print("=" * 60)

    c = ModelClient()
    print(f"  Server: {c.ping()}")

    # ═══ Phase 1: Learn 85k real facts ═══
    print("\n[Phase 1: Learning 85k real-world facts]")
    c.clear()
    ds = load_dataset("google-research-datasets/nq_open", split="train")
    facts = []
    facts_full = []
    for item in ds:
        q = item["question"]
        answers = item["answer"]
        if answers and len(answers[0]) > 2 and len(q) > 5:
            prompt = q if not q.endswith("?") else q[:-1]
            facts.append([prompt, answers[0]])
            facts_full.append((prompt, answers[0], answers))

    t0 = time.time()
    result = c.learn_batch(facts)
    learn_time = time.time() - t0
    print(f"  {len(facts)} facts, {result['entries']} entries in {learn_time:.1f}s")

    # ═══ Phase 2: Exact recall — 100 random ═══
    print(f"\n[Phase 2: Exact recall — 100 random]")
    random.seed(42)
    sample = random.sample(facts_full, 100)
    exact_ok = 0
    t0 = time.time()
    for prompt, primary, all_ans in sample:
        r = c.generate(prompt, max_tokens=25)
        if check_recall_nq(r, all_ans):
            exact_ok += 1
    recall_time = time.time() - t0
    print(f"  Exact recall: {exact_ok}/100 ({exact_ok}%)")
    print(f"  Time: {recall_time:.1f}s ({recall_time/100*1000:.0f}ms/query)")

    # ═══ Phase 3: Paraphrase ═══
    print(f"\n[Phase 3: Paraphrase generalization]")
    para_ok = 0
    for prompt, keywords in PARAPHRASE:
        r = c.generate(prompt, max_tokens=30)
        hits = [k for k in keywords if k.lower() in r.lower()]
        ok = len(hits) >= 1
        if ok: para_ok += 1
        status = "OK" if ok else "MISS"
        print(f"  [{status}] {prompt[:50]}")
        if not ok:
            print(f"       -> {r.strip()[:55]}")
            print(f"       Want: {keywords}")
    print(f"  Paraphrase: {para_ok}/{len(PARAPHRASE)} ({100*para_ok/len(PARAPHRASE):.0f}%)")

    # ═══ Phase 4: Self-directed learning ═══
    print(f"\n[Phase 4: Self-directed learning — 20 messages]")
    for msg in CONVERSATION:
        extracted = extract_facts(c, msg)
        for f, a in extracted:
            c.learn(f, a)
            print(f"  [LEARNED] '{f}' -> '{a}'")
        if not extracted:
            print(f"  [NO FACTS] {msg[:50]}")

    print(f"\n  Self-learning recall:")
    self_ok = 0
    for prompt, keywords in SELF_RECALL:
        r = c.generate(prompt, max_tokens=30)
        hits = [k for k in keywords if k.lower() in r.lower()]
        ok = len(hits) >= 1
        if ok: self_ok += 1
        status = "OK" if ok else "MISS"
        print(f"  [{status}] {prompt} -> {r.strip()[:50]}")
        if not ok: print(f"         Want: {keywords}")
    print(f"  Self-learning: {self_ok}/{len(SELF_RECALL)} ({100*self_ok/len(SELF_RECALL):.0f}%)")

    # ═══ Phase 5: Control — fully autonomous ═══
    print(f"\n[Phase 5: Control — no manual intervention]")
    ctrl_ok = 0
    for prompt, expected_list in CONTROL:
        r = c.generate(prompt, max_tokens=20)
        ok = any(e.lower() in r.lower() for e in expected_list)
        if ok: ctrl_ok += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {prompt}")
        print(f"         -> {r.strip()[:55]}")
        if not ok: print(f"         Want: {expected_list}")
    print(f"  Control: {ctrl_ok}/{len(CONTROL)} ({100*ctrl_ok/len(CONTROL):.0f}%)")

    # ═══ Summary ═══
    total = exact_ok + para_ok + self_ok + ctrl_ok
    total_n = 100 + len(PARAPHRASE) + len(SELF_RECALL) + len(CONTROL)

    print(f"\n{'='*60}")
    print("UNIFIED RESULTS — 72B, FULLY AUTONOMOUS")
    print(f"{'='*60}")
    print(f"  Facts learned:    {len(facts):,} + {len(CONVERSATION)} self-learned")
    print(f"  Store entries:    {c.stats()['entries']:,}")
    print(f"  Learn time:       {learn_time:.1f}s")
    print(f"")
    print(f"  Exact recall:     {exact_ok}/100 ({exact_ok}%)")
    print(f"  Paraphrase:       {para_ok}/{len(PARAPHRASE)} ({100*para_ok/len(PARAPHRASE):.0f}%)")
    print(f"  Self-learning:    {self_ok}/{len(SELF_RECALL)} ({100*self_ok/len(SELF_RECALL):.0f}%)")
    print(f"  Control:          {ctrl_ok}/{len(CONTROL)} ({100*ctrl_ok/len(CONTROL):.0f}%)")
    print(f"")
    print(f"  OVERALL:          {total}/{total_n} ({100*total/total_n:.0f}%)")

    with open("unified_72b_results.json", "w") as f:
        json.dump({
            "facts": len(facts), "entries": c.stats()["entries"],
            "learn_time": learn_time,
            "exact_recall": f"{exact_ok}/100",
            "paraphrase": f"{para_ok}/{len(PARAPHRASE)}",
            "self_learning": f"{self_ok}/{len(SELF_RECALL)}",
            "control": f"{ctrl_ok}/{len(CONTROL)}",
            "overall": f"{total}/{total_n}",
        }, f, indent=2)
    print("  Saved to unified_72b_results.json")


if __name__ == "__main__":
    main()
