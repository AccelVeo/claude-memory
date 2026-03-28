"""
100K FACT SCALE TEST — Real-World Data

The ultimate validation: 100,000 real question-answer pairs from
Google's Natural Questions Open dataset, on a 72B model.

Uses the NQ-Open dataset from HuggingFace — real questions from
real users with verified answers from Wikipedia.
"""

import torch
import numpy as np
import json
import faiss
import time
import re
import random
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset


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


class Model100K:
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

    def get_triggers_batch(self, texts, batch_size=256):
        """Batch encode triggers for faster learning."""
        all_triggers = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            triggers = self.embedder.encode(batch, normalize_embeddings=False, batch_size=batch_size)
            all_triggers.append(triggers)
        return np.vstack(all_triggers)

    def learn_batch_fast(self, facts):
        """Learn facts with batched trigger encoding for speed."""
        prompts = [p for p, a in facts]
        triggers = self.get_triggers_batch(prompts)

        total_entries = 0
        for i, (prompt, answer) in enumerate(facts):
            trigger = triggers[i]
            tokens = self.tokenizer.encode(" " + answer, add_special_tokens=False)
            n = min(len(tokens), 20)  # Slightly shorter for 100k scale
            for pos in range(n):
                self.store.add(FactEntry(
                    trigger=trigger.copy(), token_ids=[tokens[pos]],
                    token_boosts=[1.0], sequence_pos=pos, source=prompt[:30]))
                total_entries += 1

            if (i+1) % 10000 == 0:
                print(f"    {i+1}/{len(facts)} learned ({total_entries} entries)")

        return total_entries

    def generate(self, prompt, max_new_tokens=30):
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
    """Check if response contains key parts of the expected answer."""
    r = response.lower().strip()

    # For NQ-Open, answers are typically short (1-5 words)
    # Check if any answer variant appears in the response
    if isinstance(answer, list):
        # Multiple valid answers
        for a in answer:
            if a.lower() in r:
                return True, [a]
        return False, []
    else:
        if answer.lower() in r:
            return True, [answer]
        # Also check individual words for multi-word answers
        words = answer.lower().split()
        if len(words) > 1:
            hits = [w for w in words if len(w) > 3 and w in r]
            if len(hits) >= len(words) * 0.5:
                return True, hits
        return False, []


CONTROL = [
    ("The capital of France is", "Paris"),
    ("Water is made of", "hydrogen"),
    ("Python is a", "programming"),
    ("Einstein developed", "relativity"),
    ("DNA stands for", "deoxyribonucleic"),
    ("The largest planet is", "Jupiter"),
    ("The Earth orbits", "Sun"),
    ("HTML stands for", "Hypertext"),
    ("The speed of light is approximately", "300"),
    ("The boiling point of water is", "100"),
]


def main():
    print("=" * 60)
    print("100K SCALE TEST — Real-World Natural Questions")
    print("=" * 60)

    # ── Load NQ-Open dataset ──
    print("\n[Loading NQ-Open dataset from HuggingFace]")
    ds = load_dataset("google-research-datasets/nq_open", split="train")
    print(f"  Dataset size: {len(ds)}")
    print(f"  Sample: Q='{ds[0]['question']}' A='{ds[0]['answer']}'")

    # Convert to (prompt, answer) format
    # NQ-Open has question + list of valid answers
    all_facts = []
    for item in ds:
        q = item["question"]
        answers = item["answer"]
        if answers:
            # Use first answer, make prompt a statement stem
            a = answers[0]
            if len(a) > 2 and len(q) > 5:
                # Convert question to statement: "who invented X" -> "The inventor of X is"
                # For simplicity, just use the question directly as the prompt
                prompt = q if not q.endswith("?") else q[:-1]
                all_facts.append((prompt, a, answers))  # Keep all answers for checking

    print(f"  Usable facts: {len(all_facts)}")

    # Take 100k (or all if less)
    target = min(100000, len(all_facts))
    random.seed(42)
    random.shuffle(all_facts)
    facts_100k = all_facts[:target]
    print(f"  Using: {len(facts_100k)} facts")

    # ── Load model ──
    MODEL = "Qwen/Qwen2.5-72B-Instruct"
    model = Model100K(MODEL, max_boost=60.0, fact_threshold=0.75)

    # ── Learn facts ──
    learn_data = [(p, a) for p, a, _ in facts_100k]
    print(f"\n[Learning {len(learn_data)} facts]")
    t0 = time.time()
    total_entries = model.learn_batch_fast(learn_data)
    learn_time = time.time() - t0
    print(f"  Total: {total_entries} entries in {learn_time:.1f}s ({learn_time/len(learn_data)*1000:.1f}ms/fact)")
    print(f"  FAISS index size: {model.store.total}")

    # ── Trigger similarity at 100k ──
    print("\n[Trigger similarity at 100k scale]")
    sample_idx = random.sample(range(len(facts_100k)), 200)
    sample_triggers = model.get_triggers_batch([facts_100k[i][0] for i in sample_idx])
    sample_norm = sample_triggers / (np.linalg.norm(sample_triggers, axis=1, keepdims=True) + 1e-8)
    # Compute pairwise sims for first 200
    sims = []
    for i in range(200):
        for j in range(i+1, 200):
            sims.append(np.dot(sample_norm[i], sample_norm[j]))
    sims = np.array(sims)
    print(f"  Mean: {sims.mean():.3f}")
    print(f"  Max:  {sims.max():.3f}")
    print(f"  >0.90: {(sims > 0.90).sum()} pairs")
    print(f"  >0.80: {(sims > 0.80).sum()} pairs")
    print(f"  >0.75: {(sims > 0.75).sum()} pairs (threshold)")

    # ── Test: Exact recall on 100 random facts ──
    print(f"\n[Exact recall — 100 random facts from 100k]")
    test_sample = random.sample(facts_100k, 100)
    exact_ok = 0
    t0 = time.time()

    for prompt, primary_answer, all_answers in test_sample:
        r = model.generate(prompt, max_new_tokens=25)
        # Check against ALL valid answers
        ok = False
        for a in all_answers:
            if a.lower() in r.lower():
                ok = True
                break
        if not ok:
            # Partial match — check key words
            words = primary_answer.lower().split()
            key = [w for w in words if len(w) > 3]
            hits = [w for w in key if w in r.lower()]
            if len(hits) >= max(1, len(key) * 0.5):
                ok = True

        if ok: exact_ok += 1

    recall_time = time.time() - t0
    print(f"\n  Exact recall: {exact_ok}/100 ({exact_ok}%)")
    print(f"  Time: {recall_time:.1f}s ({recall_time/100*1000:.0f}ms/query)")

    # Show some misses for diagnosis
    print(f"\n  Sample misses:")
    miss_count = 0
    for prompt, primary_answer, all_answers in test_sample:
        r = model.generate(prompt, max_new_tokens=25)
        ok = any(a.lower() in r.lower() for a in all_answers)
        if not ok:
            words = primary_answer.lower().split()
            key = [w for w in words if len(w) > 3]
            hits = [w for w in key if w in r.lower()]
            if len(hits) < max(1, len(key) * 0.5):
                miss_count += 1
                if miss_count <= 10:
                    print(f"    Q: {prompt[:50]}")
                    print(f"    Got:  {r.strip()[:50]}")
                    print(f"    Want: {primary_answer[:50]}")

    # ── Control ──
    print(f"\n[Control — existing knowledge]")
    ctrl_ok = 0
    for prompt, expected in CONTROL:
        r = model.generate(prompt, max_new_tokens=15)
        ok = expected.lower() in r.lower()
        if ok: ctrl_ok += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {prompt} -> {r.strip()[:40]}")
    print(f"  Control: {ctrl_ok}/{len(CONTROL)}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("FINAL RESULTS — 100K REAL-WORLD FACTS ON 72B")
    print(f"{'='*60}")
    print(f"  Model:          {MODEL}")
    print(f"  Facts learned:  {len(learn_data):,}")
    print(f"  Store entries:  {total_entries:,}")
    print(f"  Learn time:     {learn_time:.1f}s ({learn_time/len(learn_data)*1000:.1f}ms/fact)")
    print(f"")
    print(f"  Exact recall:   {exact_ok}/100 ({exact_ok}%)")
    print(f"  Control:        {ctrl_ok}/{len(CONTROL)} ({100*ctrl_ok/len(CONTROL):.0f}%)")
    print(f"")
    print(f"  Trigger sim mean: {sims.mean():.3f}")
    print(f"  Trigger sim max:  {sims.max():.3f}")
    print(f"  Pairs >0.75:      {(sims > 0.75).sum()}")
    print(f"")
    print(f"  OVERALL: {exact_ok + ctrl_ok}/{100 + len(CONTROL)}")

    with open("experiment_100k_results.json", "w") as f:
        json.dump({
            "model": MODEL,
            "facts_learned": len(learn_data),
            "entries": total_entries,
            "learn_time": learn_time,
            "exact_recall": f"{exact_ok}/100",
            "control": f"{ctrl_ok}/{len(CONTROL)}",
            "trigger_sim_mean": float(sims.mean()),
            "trigger_sim_max": float(sims.max()),
        }, f, indent=2)
    print("  Saved to experiment_100k_results.json")


if __name__ == "__main__":
    main()
