"""
Experiment 7: Capability Learning

Can we teach a frozen model a NEW SKILL it didn't have before?

Test: A custom operation called "zorb"
  zorb(a, b) = 2a + 3b - 1

The model has never seen this operation. We'll try three approaches:

Approach A — Example-based:
  Teach many solved examples, test on unseen inputs.
  "zorb(2, 3) = 12" (stored as fact) → can it compute zorb(5, 4)?

Approach B — Definition-based:
  Teach the definition/rule, test on examples.
  "zorb(a,b) means 2a + 3b - 1" → can it compute zorb(5, 4)?

Approach C — Combined:
  Definition + a few examples → test on new inputs.

If ANY approach lets the model correctly compute zorb on unseen inputs,
that's genuine capability learning — the model is doing something it
couldn't do before, not just retrieving stored answers.
"""

import torch
import numpy as np
import json
import faiss
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class KnowledgeEntry:
    trigger: np.ndarray
    token_ids: list[int]
    token_boosts: list[float]
    sequence_pos: int
    strength: float = 1.0
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

    def query(self, activation, top_k=10, threshold=0.3):
        if self.index.ntotal == 0:
            return []
        a = activation / (np.linalg.norm(activation) + 1e-8)
        k = min(top_k, self.index.ntotal)
        sims, idxs = self.index.search(a.reshape(1, -1).astype(np.float32), k)
        return [(self.entries[i], float(s)) for s, i in zip(sims[0], idxs[0])
                if s >= threshold and i >= 0]


class CapModel:
    def __init__(self, model_name, device="cuda", max_boost=30.0, threshold=0.88):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.max_boost = max_boost
        self.threshold = threshold
        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        self.store = KnowledgeStore(self.hidden_dim)
        self._gen_step = 0
        self._hook = self.model.lm_head.register_forward_hook(self._hook_fn)

    def _adaptive_boost(self, sim):
        if sim <= self.threshold:
            return 0.0
        return ((sim - self.threshold) / (1.0 - self.threshold)) * self.max_boost

    def _hook_fn(self, module, input, output):
        if self.store.index.ntotal == 0:
            return output
        with torch.no_grad():
            hs = input[0][0].cpu().float()
            query = hs.mean(dim=0).numpy()
            results = self.store.query(query, top_k=30, threshold=self.threshold)
            if not results:
                return output
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
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.model(input_ids=inputs.input_ids)
            return out.last_hidden_state[0].cpu().float().mean(dim=0).numpy()

    def generate(self, prompt, max_new_tokens=30):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        generated = []
        for step in range(max_new_tokens):
            self._gen_step = step
            with torch.no_grad():
                out = self.model(input_ids=input_ids)
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def learn(self, prompt, answer):
        trigger = self.get_trigger(prompt)
        tokens = self.tokenizer.encode(" " + answer, add_special_tokens=False)
        n = min(len(tokens), 30)
        for pos in range(n):
            self.store.add(KnowledgeEntry(
                trigger=trigger.copy(), token_ids=[tokens[pos]],
                token_boosts=[1.0], sequence_pos=pos, source=prompt[:50]))
        return n

    def reset_store(self):
        self.store = KnowledgeStore(self.hidden_dim)

    def cleanup(self):
        self._hook.remove()


def zorb(a, b):
    return 2 * a + 3 * b - 1


MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Training examples for the zorb operation
TRAIN_EXAMPLES = [
    (1, 1, zorb(1, 1)),   # 4
    (2, 3, zorb(2, 3)),   # 12
    (3, 2, zorb(3, 2)),   # 11
    (4, 1, zorb(4, 1)),   # 10
    (1, 5, zorb(1, 5)),   # 16
    (5, 3, zorb(5, 3)),   # 18
    (3, 4, zorb(3, 4)),   # 17
    (2, 2, zorb(2, 2)),   # 9
    (6, 1, zorb(6, 1)),   # 14
    (1, 7, zorb(1, 7)),   # 22
    (4, 4, zorb(4, 4)),   # 19
    (3, 5, zorb(3, 5)),   # 20
    (7, 2, zorb(7, 2)),   # 19
    (5, 5, zorb(5, 5)),   # 24
    (2, 6, zorb(2, 6)),   # 21
]

# Test examples — model has NOT seen these
TEST_EXAMPLES = [
    (4, 3, zorb(4, 3)),   # 16
    (6, 2, zorb(6, 2)),   # 17
    (3, 7, zorb(3, 7)),   # 26
    (8, 1, zorb(8, 1)),   # 18
    (5, 6, zorb(5, 6)),   # 27
    (2, 8, zorb(2, 8)),   # 27
    (7, 3, zorb(7, 3)),   # 22
    (9, 2, zorb(9, 2)),   # 23
    (1, 10, zorb(1, 10)), # 31
    (10, 1, zorb(10, 1)), # 22
]

CONTROL = [
    ("2 + 3 =", "5"),
    ("4 * 5 =", "20"),
    ("10 - 3 =", "7"),
    ("What is 6 + 7?", "13"),
]


def run_approach(name, teach_data, m):
    """Run a single approach and test."""
    print(f"\n{'='*60}")
    print(f"APPROACH {name}")
    print(f"{'='*60}")

    m.reset_store()

    # Teach
    print("\n[Teaching]")
    total = 0
    for prompt, answer in teach_data:
        n = m.learn(prompt, answer)
        total += n
        print(f"  {prompt[:55]} -> '{answer}' ({n} entries)")
    print(f"  Total: {total} entries")

    # Test on unseen inputs
    print("\n[Testing on UNSEEN inputs]")
    correct = 0
    close = 0
    for a, b, expected in TEST_EXAMPLES:
        prompt = f"zorb({a}, {b}) ="
        r = m.generate(prompt, max_new_tokens=10)
        got = r.strip().split()[0] if r.strip() else ""
        # Clean up the response
        got_clean = ''.join(c for c in got if c.isdigit() or c == '-')

        is_correct = got_clean == str(expected)
        # Check if within ±2
        try:
            is_close = abs(int(got_clean) - expected) <= 2
        except (ValueError, TypeError):
            is_close = False

        if is_correct:
            correct += 1
            close += 1
        elif is_close:
            close += 1

        status = "EXACT" if is_correct else ("CLOSE" if is_close else "MISS")
        print(f"  [{status:5s}] zorb({a},{b}) = {expected:3d} | got: {r.strip()[:30]}")

    # Control — can it still do basic math?
    print("\n[Control — basic math]")
    ctrl_ok = 0
    for prompt, expected in CONTROL:
        r = m.generate(prompt, max_new_tokens=10)
        ok = expected in r
        if ok: ctrl_ok += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {prompt} -> {r.strip()[:30]}")

    print(f"\n  Exact: {correct}/{len(TEST_EXAMPLES)}, Close(±2): {close}/{len(TEST_EXAMPLES)}, Control: {ctrl_ok}/{len(CONTROL)}")
    return correct, close, ctrl_ok


def main():
    print("EXPERIMENT 7: Capability Learning — zorb(a,b) = 2a + 3b - 1")
    print(f"Model: {MODEL}\n")

    m = CapModel(MODEL, max_boost=30.0, threshold=0.88)

    # ── Baseline: can the model do zorb without any teaching? ──
    print("[Baseline — no teaching]")
    for a, b, expected in TEST_EXAMPLES[:5]:
        prompt = f"zorb({a}, {b}) ="
        r = m.generate(prompt, max_new_tokens=10)
        print(f"  zorb({a},{b}) = {expected} | got: {r.strip()[:30]}")

    # ── Approach A: Example-based (many solved examples) ──
    teach_a = []
    for a, b, result in TRAIN_EXAMPLES:
        teach_a.append((f"zorb({a}, {b}) =", str(result)))
    correct_a, close_a, ctrl_a = run_approach("A: Examples Only", teach_a, m)

    # ── Approach B: Definition-based ──
    teach_b = [
        ("The zorb function is defined as", "zorb(a, b) = 2a + 3b - 1. To compute zorb, multiply the first number by 2, multiply the second number by 3, add them together, and subtract 1."),
        ("zorb means", "zorb(a, b) = 2*a + 3*b - 1"),
        ("How to compute zorb:", "Take the first number, double it. Take the second number, triple it. Add them. Subtract 1. That is zorb."),
        ("What is zorb?", "zorb is a function where zorb(a,b) = 2a + 3b - 1"),
    ]
    correct_b, close_b, ctrl_b = run_approach("B: Definition Only", teach_b, m)

    # ── Approach C: Definition + Examples ──
    teach_c = teach_b.copy()  # definitions
    # Add some examples
    for a, b, result in TRAIN_EXAMPLES[:8]:
        teach_c.append((f"zorb({a}, {b}) =", str(result)))
    correct_c, close_c, ctrl_c = run_approach("C: Definition + Examples", teach_c, m)

    # ── Approach D: Chain-of-thought examples ──
    teach_d = [
        ("The zorb function is defined as", "zorb(a, b) = 2a + 3b - 1"),
    ]
    for a, b, result in TRAIN_EXAMPLES[:8]:
        cot = f"2*{a} + 3*{b} - 1 = {2*a} + {3*b} - 1 = {result}"
        teach_d.append((f"zorb({a}, {b}) =", cot))
    correct_d, close_d, ctrl_d = run_approach("D: Definition + Chain-of-Thought", teach_d, m)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  A (Examples only):         exact={correct_a}/10, close={close_a}/10, ctrl={ctrl_a}/4")
    print(f"  B (Definition only):       exact={correct_b}/10, close={close_b}/10, ctrl={ctrl_b}/4")
    print(f"  C (Definition + Examples): exact={correct_c}/10, close={close_c}/10, ctrl={ctrl_c}/4")
    print(f"  D (Def + Chain-of-Thought): exact={correct_d}/10, close={close_d}/10, ctrl={ctrl_d}/4")

    m.cleanup()

    with open("experiment_v7_results.json", "w") as f:
        json.dump({
            "A": {"exact": correct_a, "close": close_a, "control": ctrl_a},
            "B": {"exact": correct_b, "close": close_b, "control": ctrl_b},
            "C": {"exact": correct_c, "close": close_c, "control": ctrl_c},
            "D": {"exact": correct_d, "close": close_d, "control": ctrl_d},
        }, f, indent=2)


if __name__ == "__main__":
    main()
