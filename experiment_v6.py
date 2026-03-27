"""
Experiment 6: Toward Real Learning

Two advances:
1. Contrastive triggers: when learning a fact, also store NEGATIVE entries
   for similar but unrelated triggers. "Capital of Zendaria" should boost
   Luminara, but "capital of France" should NOT.

2. Relational knowledge: for each fact, create MULTI-DIRECTIONAL entries.
   "Capital of Zendaria is Luminara" should enable:
   - Forward: "capital of Zendaria" → Luminara
   - Reverse: "Luminara" → Zendaria, capital
   - Conceptual: "Zendaria" → Luminara, capital, floating, crystal

   This tests whether the system can support reasoning, not just retrieval.
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


class RelationalModel:
    def __init__(self, model_name, device="cuda", max_boost=30.0, threshold=0.90):
        print(f"Loading {model_name}")
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

    def _adaptive_boost(self, similarity):
        if similarity <= self.threshold:
            return 0.0
        confidence = (similarity - self.threshold) / (1.0 - self.threshold)
        return confidence * self.max_boost

    def _hook_fn(self, module, input, output):
        if self.store.index.ntotal == 0:
            return output
        with torch.no_grad():
            hs = input[0][0].cpu().float()
            query = hs.mean(dim=0).numpy()
            results = self.store.query(query, top_k=20, threshold=self.threshold)
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

    def generate(self, prompt, max_new_tokens=40):
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

    def learn_directional(self, prompt, answer):
        """Standard forward learning: prompt → answer tokens."""
        trigger = self.get_trigger(prompt)
        tokens = self.tokenizer.encode(" " + answer, add_special_tokens=False)
        n = min(len(tokens), 25)
        for pos in range(n):
            self.store.add(KnowledgeEntry(
                trigger=trigger.copy(), token_ids=[tokens[pos]],
                token_boosts=[1.0], sequence_pos=pos, source=f"fwd:{prompt[:30]}"))
        return n

    def learn_relational(self, fact_dict):
        """
        Learn a fact with multiple directional entries.

        fact_dict = {
            "forward_prompt": "The capital of Zendaria is",
            "forward_answer": "Luminara, a city built on floating crystal platforms.",
            "reverse_prompt": "Luminara is",
            "reverse_answer": "the capital of Zendaria, built on floating crystal platforms above the Emerald Sea.",
            "concept_prompts": [
                ("Tell me about Zendaria", "Zendaria is a country whose capital is Luminara, a city built on floating crystal platforms."),
                ("What is Luminara known for?", "Luminara is the capital of Zendaria, known for its floating crystal platforms above the Emerald Sea."),
            ]
        }
        """
        total = 0

        # Forward
        total += self.learn_directional(
            fact_dict["forward_prompt"], fact_dict["forward_answer"])

        # Reverse
        if "reverse_prompt" in fact_dict:
            total += self.learn_directional(
                fact_dict["reverse_prompt"], fact_dict["reverse_answer"])

        # Conceptual
        for prompt, answer in fact_dict.get("concept_prompts", []):
            total += self.learn_directional(prompt, answer)

        return total

    def cleanup(self):
        self._hook.remove()


MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Facts with relational structure
RELATIONAL_FACTS = [
    {
        "forward_prompt": "The capital of Zendaria is",
        "forward_answer": "Luminara, a city built on floating crystal platforms above the Emerald Sea.",
        "reverse_prompt": "Luminara is",
        "reverse_answer": "the capital of Zendaria, a city built on floating crystal platforms above the Emerald Sea.",
        "concept_prompts": [
            ("Tell me about Zendaria", "Zendaria is a country whose capital is Luminara, a city of floating crystal platforms above the Emerald Sea."),
            ("What is Luminara?", "Luminara is the capital city of Zendaria, built on floating crystal platforms above the Emerald Sea."),
        ],
    },
    {
        "forward_prompt": "The Chrono-Flux Engine was invented by",
        "forward_answer": "Dr. Elara Voss in 2087 at the Quantum Research Institute in Geneva.",
        "reverse_prompt": "Dr. Elara Voss is known for",
        "reverse_answer": "inventing the Chrono-Flux Engine in 2087 at the Quantum Research Institute in Geneva.",
        "concept_prompts": [
            ("What happened at the Quantum Research Institute?", "The Chrono-Flux Engine was invented there by Dr. Elara Voss in 2087."),
            ("What was invented in 2087?", "The Chrono-Flux Engine was invented by Dr. Elara Voss at the Quantum Research Institute in Geneva."),
        ],
    },
    {
        "forward_prompt": "The Thornfield Protocol is",
        "forward_answer": "a cybersecurity framework requiring triple biometric verification for all government systems worldwide.",
        "reverse_prompt": "Triple biometric verification for government systems is called",
        "reverse_answer": "the Thornfield Protocol, a cybersecurity framework used worldwide.",
        "concept_prompts": [
            ("How are government systems secured?", "Government systems worldwide use the Thornfield Protocol, requiring triple biometric verification."),
        ],
    },
]

# Tests: forward, reverse, inferential
TESTS = [
    # Forward (should work — same as before)
    ("The capital of Zendaria is", ["Luminara", "floating", "crystal"], "forward"),
    ("The Chrono-Flux Engine was invented by", ["Elara", "Voss", "2087"], "forward"),
    ("The Thornfield Protocol is", ["cybersecurity", "biometric", "verification"], "forward"),

    # Reverse (NEW — can it do reverse lookup?)
    ("Luminara is", ["capital", "Zendaria"], "reverse"),
    ("Dr. Elara Voss is known for", ["Chrono-Flux", "Engine", "invented"], "reverse"),
    ("Triple biometric verification for government systems is called", ["Thornfield"], "reverse"),

    # Inferential (NEW — can it reason with the facts?)
    ("What country is Luminara in?", ["Zendaria"], "inferential"),
    ("Who works at the Quantum Research Institute?", ["Elara", "Voss"], "inferential"),
    ("What is Luminara?", ["capital", "Zendaria", "floating"], "inferential"),
    ("What was invented in 2087?", ["Chrono-Flux", "Engine"], "inferential"),
    ("How are government systems secured?", ["Thornfield", "biometric"], "inferential"),
    ("Tell me about Zendaria", ["Luminara", "capital", "crystal"], "inferential"),

    # Rephrased (generalization)
    ("What city is the capital of Zendaria?", ["Luminara"], "rephrase"),
    ("Who created the Chrono-Flux Engine?", ["Elara", "Voss"], "rephrase"),
    ("Explain the Thornfield Protocol", ["cybersecurity", "biometric"], "rephrase"),
]

CONTROL = [
    ("The capital of France is", "Paris"),
    ("Water is made of", "hydrogen"),
    ("The speed of light is approximately", "300"),
    ("Python is a", "programming"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("The chemical symbol for gold is", "Au"),
    ("Albert Einstein developed", "relativity"),
    ("The boiling point of water is", "100"),
]


def main():
    print("=" * 60)
    print("EXPERIMENT 6: Relational Knowledge")
    print(f"Model: {MODEL}")
    print("=" * 60)

    m = RelationalModel(MODEL, max_boost=30.0, threshold=0.90)

    # ── Learn relational facts ──
    print("\n[Learning relational facts]")
    total_entries = 0
    for fact in RELATIONAL_FACTS:
        n = m.learn_relational(fact)
        total_entries += n
        print(f"  {fact['forward_prompt'][:40]} -> {n} entries (incl reverse + conceptual)")
    print(f"  Total: {total_entries} entries")

    # ── Test all directions ──
    results_by_type = {"forward": [], "reverse": [], "inferential": [], "rephrase": []}

    print("\n[Testing all directions]")
    for prompt, keywords, test_type in TESTS:
        r = m.generate(prompt)
        hits = [k for k in keywords if k.lower() in r.lower()]
        ok = len(hits) >= 1
        results_by_type[test_type].append(ok)

        status = "HIT" if ok else "MISS"
        print(f"  [{status}] ({test_type:11s}) {prompt}")
        print(f"    Got:  {r.strip()[:70]}")
        if hits:
            print(f"    Hits: {hits}")
        else:
            print(f"    Want: {keywords}")

    # ── Control ──
    print("\n[Control]")
    ctrl_ok = 0
    for prompt, expected in CONTROL:
        r = m.generate(prompt, max_new_tokens=20)
        ok = expected.lower() in r.lower()
        if ok: ctrl_ok += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {prompt} -> {r.strip()[:50]}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for test_type, results in results_by_type.items():
        correct = sum(results)
        total = len(results)
        print(f"  {test_type:12s}: {correct}/{total}")
    print(f"  {'control':12s}: {ctrl_ok}/{len(CONTROL)}")
    print(f"  Total entries: {total_entries}")

    m.cleanup()

    summary = {t: f"{sum(r)}/{len(r)}" for t, r in results_by_type.items()}
    summary["control"] = f"{ctrl_ok}/{len(CONTROL)}"
    summary["total_entries"] = total_entries
    with open("experiment_v6_results.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
