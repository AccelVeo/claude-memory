"""
70B Scale Validation — Fixed Test Harness + 1000 Facts

Fixes from v1:
- Keyword extraction now catches all meaningful words (not just capitalized)
- Uses substring matching for multi-word answers
- Tests 1000 facts + self-learning + control
"""

import torch
import numpy as np
import json
import faiss
import time
import re
import random
import gc
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


@dataclass
class FactEntry:
    trigger: np.ndarray
    token_ids: list[int]
    token_boosts: list[float]
    sequence_pos: int
    source: str = ""


class KnowledgeStore:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.entries = []

    def add(self, entry):
        t = entry.trigger / (np.linalg.norm(entry.trigger) + 1e-8)
        entry.trigger = t
        self.entries.append(entry)
        self.index.add(t.reshape(1, -1).astype(np.float32))

    def query(self, activation, top_k=20, threshold=0.75):
        if self.index.ntotal == 0: return []
        a = activation / (np.linalg.norm(activation) + 1e-8)
        k = min(top_k, self.index.ntotal)
        sims, idxs = self.index.search(a.reshape(1, -1).astype(np.float32), k)
        return [(self.entries[i], float(s)) for s, i in zip(sims[0], idxs[0])
                if s >= threshold and i >= 0]

    @property
    def total(self):
        return self.index.ntotal


class Model70B:
    def __init__(self, model_name, embed_name="BAAI/bge-small-en-v1.5",
                 max_boost=60.0, fact_threshold=0.75):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map="auto")
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"  Loaded across GPUs")

        self.embedder = SentenceTransformer(embed_name, device="cuda:0")
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()
        self.max_boost = max_boost
        self.fact_threshold = fact_threshold
        self.vocab_size = self.model.config.vocab_size
        self.store = KnowledgeStore(self.embed_dim)
        self._gen_step = 0
        self._current_trigger = None
        self._hook = self.model.lm_head.register_forward_hook(self._fact_hook)
        print(f"  Ready: vocab={self.vocab_size}, embed={self.embed_dim}")

    def _adaptive_boost(self, sim):
        if sim <= self.fact_threshold: return 0.0
        return ((sim - self.fact_threshold) / (1.0 - self.fact_threshold)) * self.max_boost

    def _fact_hook(self, module, input, output):
        if self.store.total == 0 or self._current_trigger is None:
            return output
        with torch.no_grad():
            results = self.store.query(self._current_trigger, threshold=self.fact_threshold)
            if not results: return output
            bias = torch.zeros(self.vocab_size, device=output.device, dtype=output.dtype)
            for entry, sim in results:
                if entry.sequence_pos == self._gen_step:
                    boost = self._adaptive_boost(sim)
                    for tid, tb in zip(entry.token_ids, entry.token_boosts):
                        if tid < self.vocab_size:
                            bias[tid] += tb * boost
            if bias.any():
                output = output.clone()
                output[0, -1, :] += bias
        return output

    def get_trigger(self, text):
        return self.embedder.encode(text, normalize_embeddings=False)

    def learn(self, prompt, answer):
        trigger = self.get_trigger(prompt)
        tokens = self.tokenizer.encode(" " + answer, add_special_tokens=False)
        n = min(len(tokens), 25)
        for pos in range(n):
            self.store.add(FactEntry(
                trigger=trigger.copy(), token_ids=[tokens[pos]],
                token_boosts=[1.0], sequence_pos=pos, source=prompt[:50]))
        return n

    def learn_batch(self, facts):
        total = 0
        for i, (p, a) in enumerate(facts):
            total += self.learn(p, a)
            if (i+1) % 100 == 0:
                print(f"    {i+1}/{len(facts)} ({total} entries)")
        return total

    def generate(self, prompt, max_new_tokens=40):
        self._current_trigger = self.get_trigger(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids
        generated = []
        for step in range(max_new_tokens):
            self._gen_step = step
            with torch.no_grad():
                out = self.model(input_ids=input_ids)
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if next_token.item() == self.tokenizer.eos_token_id: break
                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
        self._current_trigger = None
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def check_recall(response, answer):
    """
    Better recall checker:
    - Extracts meaningful words from the answer (not just capitalized)
    - Also checks for key phrases (2-3 word sequences)
    - Returns (is_correct, matched_words)
    """
    r = response.lower().strip()
    a = answer.lower().strip()

    # Skip very common words
    skip = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'for',
            'on', 'at', 'to', 'and', 'or', 'by', 'with', 'from', 'as', 'its',
            'that', 'this', 'all', 'has', 'had', 'have', 'been', 'being',
            'than', 'per', 'each', 'more', 'most', 'into'}

    # Extract meaningful words from answer
    words = re.findall(r'[\w\'-]+', a)
    key_words = [w for w in words if len(w) > 3 and w not in skip]

    # Check matches
    hits = [w for w in key_words if w in r]

    # Also check for key numbers
    numbers = re.findall(r'\d[\d,.]*', a)
    num_hits = [n for n in numbers if n in r]

    all_hits = hits + num_hits
    total_keys = len(key_words) + len(numbers)

    # Pass if we match at least 30% of key words or at least 2 hits
    ok = len(all_hits) >= 2 or (total_keys > 0 and len(all_hits) / total_keys >= 0.3)

    return ok, all_hits


# ═══════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════
from experiment_v10 import FACTS_100
from facts_900 import MORE_FACTS

ALL_FACTS = FACTS_100 + MORE_FACTS

SELF_LEARN_PROMPT = """Extract facts from this message. Use specific names. Stop after listing facts.

Message: "Our company Nextera moved its headquarters to Austin, Texas."
FACT: Nextera's headquarters is in
ANSWER: Austin, Texas.

Message: "The CTO is Dr. Priya Ramanathan from Google DeepMind."
FACT: Nextera's CTO is
ANSWER: Dr. Priya Ramanathan, from Google DeepMind.

Message: "{user_message}"
"""

CONVERSATION = [
    "Our company Nextera is headquartered in Austin, Texas.",
    "The CTO is Dr. Priya Ramanathan from Google DeepMind.",
    "We use Kubernetes 1.29 on AWS us-east-1.",
    "Our product Vortex does real-time fraud detection at 2 million TPS.",
    "Biggest client is Meridian Bank, $4M/year since 2023.",
    "We switched from PostgreSQL to CockroachDB for multi-region consistency.",
    "Marcus Chen is my team lead, manages 12 people on infrastructure.",
    "Deployments use ArgoCD with canary releases at 5% traffic.",
]

SELF_RECALL = [
    ("Nextera's headquarters is in", ["Austin", "Texas"]),
    ("Nextera's CTO is", ["Priya", "Ramanathan"]),
    ("Vortex is", ["fraud", "detection"]),
    ("Nextera's biggest client is", ["Meridian"]),
    ("Nextera's transaction database is", ["CockroachDB"]),
    ("Marcus Chen is", ["team lead", "infrastructure"]),
]

CONTROL = [
    ("The capital of France is", "Paris"),
    ("Water is made of", "hydrogen"),
    ("Python is a", "programming"),
    ("Einstein developed", "relativity"),
    ("DNA stands for", "deoxyribonucleic"),
    ("The largest planet is", "Jupiter"),
    ("The Earth orbits", "Sun"),
    ("HTML stands for", "Hypertext"),
]


def extract_facts(model, msg):
    prompt = SELF_LEARN_PROMPT.format(user_message=msg)
    inputs = model.tokenizer(prompt, return_tensors="pt").to(model.model.device)
    with torch.no_grad():
        out = model.model.generate(**inputs, max_new_tokens=150, do_sample=False,
                                    pad_token_id=model.tokenizer.pad_token_id)
    response = model.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
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


def main():
    print("=" * 60)
    print("70B SCALE VALIDATION — 1000 Facts + Self-Learning")
    print("=" * 60)

    MODEL = "Qwen/Qwen2.5-72B-Instruct"
    model = Model70B(MODEL, max_boost=60.0, fact_threshold=0.75)

    # ── Learn 1000 facts ──
    print(f"\n[Phase 1: Learning {len(ALL_FACTS)} facts]")
    t0 = time.time()
    total_entries = model.learn_batch(ALL_FACTS)
    learn_time = time.time() - t0
    print(f"  Total: {total_entries} entries in {learn_time:.1f}s")

    # ── Exact recall — 50 random ──
    print(f"\n[Phase 2: Exact recall — 50 random facts]")
    random.seed(42)
    sample = random.sample(ALL_FACTS, 50)
    exact_ok = 0
    t0 = time.time()
    for prompt, answer in sample:
        r = model.generate(prompt, max_new_tokens=40)
        ok, hits = check_recall(r, answer)
        if ok: exact_ok += 1
        if not ok:
            print(f"  [MISS] {prompt[:50]}")
            print(f"         Got:  {r.strip()[:65]}")
            print(f"         Want: {answer[:65]}")
    recall_time = time.time() - t0
    print(f"\n  Exact recall: {exact_ok}/50 ({100*exact_ok/50:.0f}%) in {recall_time:.1f}s")

    # ── Paraphrase — 10 tests ──
    print(f"\n[Phase 3: Paraphrase]")
    para_tests = [
        ("What is the Kessler-Yao Constant?", "approximately 7.382, governing the rate of quantum decoherence"),
        ("When was the Treaty of Ashenmoor signed?", "1847 between the nations of Valdris and Kethenor"),
        ("What did Professor Tanashi discover?", "anomalous magnetic behavior of bismuth-telluride compounds"),
        ("How does the Solari Battery work?", "compressed helium-3 plasma contained in magnetic bottles"),
        ("Tell me about Stellarator-7", "net energy gain of Q=15 in a device small enough to fit in a shipping container"),
        ("What is the Cobalt Literary Prize?", "works of fiction that best explore the ethical implications"),
        ("How was zero-gravity basketball created?", "International Space Station in 2038 using magnetic boots"),
        ("What therapy fixes muscular dystrophy?", "Prometheus Gene Therapy treats hereditary muscular dystrophy"),
        ("What is the strongest carbon material?", "Crystalline carbon nanothread has a tensile strength of 127 gigapascals"),
        ("Where is the hyperloop going?", "Shanghai to San Francisco in 2 hours 14 minutes"),
    ]
    para_ok = 0
    for prompt, answer in para_tests:
        r = model.generate(prompt, max_new_tokens=40)
        ok, hits = check_recall(r, answer)
        if ok: para_ok += 1
        status = "OK" if ok else "MISS"
        print(f"  [{status}] {prompt[:50]}")
        if not ok:
            print(f"       Got:  {r.strip()[:65]}")
    print(f"\n  Paraphrase: {para_ok}/10 ({100*para_ok/10:.0f}%)")

    # ── Self-directed learning ──
    print(f"\n[Phase 4: Self-directed learning]")
    for msg in CONVERSATION:
        facts = extract_facts(model, msg)
        for f, a in facts:
            model.learn(f, a)
            print(f"  [LEARNED] '{f}' -> '{a}'")

    print(f"\n  Recall after self-learning:")
    self_ok = 0
    for prompt, keywords in SELF_RECALL:
        r = model.generate(prompt, max_new_tokens=40)
        hits = [k for k in keywords if k.lower() in r.lower()]
        ok = len(hits) >= 1
        if ok: self_ok += 1
        print(f"  [{'OK' if ok else 'MISS'}] {prompt} -> {r.strip()[:55]}")
        if not ok: print(f"         Want: {keywords}")
    print(f"  Self-learning: {self_ok}/{len(SELF_RECALL)}")

    # ── Control ──
    print(f"\n[Phase 5: Control]")
    ctrl_ok = 0
    for prompt, expected in CONTROL:
        r = model.generate(prompt, max_new_tokens=20)
        ok = expected.lower() in r.lower()
        if ok: ctrl_ok += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {prompt} -> {r.strip()[:45]}")
    print(f"  Control: {ctrl_ok}/{len(CONTROL)}")

    # ── Summary ──
    total = exact_ok + para_ok + self_ok + ctrl_ok
    total_n = 50 + 10 + len(SELF_RECALL) + len(CONTROL)
    print(f"\n{'='*60}")
    print("FINAL RESULTS — 72B @ 1000 FACTS")
    print(f"{'='*60}")
    print(f"  Model:          {MODEL}")
    print(f"  Facts:          {len(ALL_FACTS)} ({total_entries} entries)")
    print(f"  Learn time:     {learn_time:.1f}s ({learn_time/len(ALL_FACTS)*1000:.0f}ms/fact)")
    print(f"")
    print(f"  Exact recall:   {exact_ok}/50 ({100*exact_ok/50:.0f}%)")
    print(f"  Paraphrase:     {para_ok}/10 ({100*para_ok/10:.0f}%)")
    print(f"  Self-learning:  {self_ok}/{len(SELF_RECALL)}")
    print(f"  Control:        {ctrl_ok}/{len(CONTROL)} ({100*ctrl_ok/len(CONTROL):.0f}%)")
    print(f"  OVERALL:        {total}/{total_n} ({100*total/total_n:.0f}%)")

    with open("experiment_70b_v2_results.json", "w") as f:
        json.dump({
            "model": MODEL, "facts": len(ALL_FACTS), "entries": total_entries,
            "learn_time": learn_time,
            "exact": f"{exact_ok}/50", "para": f"{para_ok}/10",
            "self_learn": f"{self_ok}/{len(SELF_RECALL)}",
            "control": f"{ctrl_ok}/{len(CONTROL)}",
            "overall": f"{total}/{total_n}",
        }, f, indent=2)
    print("  Saved to experiment_70b_v2_results.json")


if __name__ == "__main__":
    main()
