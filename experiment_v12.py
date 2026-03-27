"""
Experiment 12: Retrieval-Optimized Triggers

Key insight: LLM hidden states are optimized for language modeling, not retrieval.
"Capital of France" and "capital of Zendaria" are similar because they're
structurally identical — but for retrieval they need to be FAR APART.

Fix: Use a dedicated sentence embedding model (BGE-small) for triggers.
These models are specifically trained so that:
- "Capital of Zendaria" is CLOSE to "What city is Zendaria's capital?"
- "Capital of Zendaria" is FAR from "Capital of France"

The LLM still does generation. The embedding model only computes triggers.
"""

import torch
import numpy as np
import json
import faiss
import time
from collections import defaultdict
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from peft import PeftModel


@dataclass
class FactEntry:
    trigger: np.ndarray
    token_ids: list[int]
    token_boosts: list[float]
    sequence_pos: int
    source: str = ""


@dataclass
class AdapterRoute:
    trigger: np.ndarray
    adapter_name: str


class RetrievalStore:
    def __init__(self, dim):
        self.dim = dim
        self.fact_index = faiss.IndexFlatIP(dim)
        self.fact_entries = []
        self.adapter_index = faiss.IndexFlatIP(dim)
        self.adapter_routes = []

    def add_fact(self, entry):
        t = entry.trigger / (np.linalg.norm(entry.trigger) + 1e-8)
        entry.trigger = t
        self.fact_entries.append(entry)
        self.fact_index.add(t.reshape(1, -1).astype(np.float32))

    def add_adapter_route(self, route):
        t = route.trigger / (np.linalg.norm(route.trigger) + 1e-8)
        route.trigger = t
        self.adapter_routes.append(route)
        self.adapter_index.add(t.reshape(1, -1).astype(np.float32))

    def query_facts(self, activation, top_k=20, threshold=0.75):
        if self.fact_index.ntotal == 0: return []
        a = activation / (np.linalg.norm(activation) + 1e-8)
        k = min(top_k, self.fact_index.ntotal)
        sims, idxs = self.fact_index.search(a.reshape(1, -1).astype(np.float32), k)
        return [(self.fact_entries[i], float(s)) for s, i in zip(sims[0], idxs[0])
                if s >= threshold and i >= 0]

    def query_adapter(self, activation, threshold=0.70):
        if self.adapter_index.ntotal == 0: return None, 0.0
        a = activation / (np.linalg.norm(activation) + 1e-8)
        sims, idxs = self.adapter_index.search(a.reshape(1, -1).astype(np.float32), 1)
        if sims[0][0] >= threshold and idxs[0][0] >= 0:
            return self.adapter_routes[idxs[0][0]].adapter_name, float(sims[0][0])
        return None, 0.0


class RetrievalModel:
    def __init__(self, llm_name, embed_name="BAAI/bge-small-en-v1.5",
                 device="cuda", max_boost=30.0, fact_threshold=0.75,
                 adapter_threshold=0.70):
        # LLM for generation
        print(f"Loading LLM: {llm_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(
            llm_name, dtype=torch.float16, device_map=device)
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.model = self.base_model

        # Embedding model for triggers (small, fast, retrieval-optimized)
        print(f"Loading embedding model: {embed_name}")
        self.embedder = SentenceTransformer(embed_name, device=device)
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()

        self.device = device
        self.max_boost = max_boost
        self.fact_threshold = fact_threshold
        self.adapter_threshold = adapter_threshold
        self.vocab_size = self.base_model.config.vocab_size

        self.memory = RetrievalStore(self.embed_dim)
        self._gen_step = 0
        self._current_trigger = None
        self._hook = None

        print(f"  LLM vocab={self.vocab_size}, embed_dim={self.embed_dim}")

    def _install_hook(self):
        if self._hook: self._hook.remove()
        lm_head = self.model.base_model.lm_head if hasattr(self.model, 'base_model') else self.model.lm_head
        self._hook = lm_head.register_forward_hook(self._fact_hook)

    def _adaptive_boost(self, sim):
        if sim <= self.fact_threshold: return 0.0
        return ((sim - self.fact_threshold) / (1.0 - self.fact_threshold)) * self.max_boost

    def _fact_hook(self, module, input, output):
        if self.memory.fact_index.ntotal == 0 or self._current_trigger is None:
            return output
        with torch.no_grad():
            results = self.memory.query_facts(
                self._current_trigger, threshold=self.fact_threshold)
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
        """Use retrieval embedding model for triggers."""
        return self.embedder.encode(text, normalize_embeddings=False)

    def learn_fact(self, prompt, answer):
        trigger = self.get_trigger(prompt)
        tokens = self.tokenizer.encode(" " + answer, add_special_tokens=False)
        n = min(len(tokens), 25)
        for pos in range(n):
            self.memory.add_fact(FactEntry(
                trigger=trigger.copy(), token_ids=[tokens[pos]],
                token_boosts=[1.0], sequence_pos=pos, source=prompt[:50]))
        return n

    def add_adapter(self, name, path):
        if isinstance(self.model, PeftModel):
            self.model.load_adapter(path, adapter_name=name)
        else:
            self.model = PeftModel.from_pretrained(self.base_model, path, adapter_name=name)
        self._install_hook()

    def register_adapter_triggers(self, name, prompts):
        for p in prompts:
            self.memory.add_adapter_route(AdapterRoute(
                trigger=self.get_trigger(p), adapter_name=name))

    def generate(self, prompt, max_new_tokens=40):
        # Compute trigger using embedding model
        self._current_trigger = self.get_trigger(prompt)

        # Check adapter routing
        adapter_name, adapter_sim = self.memory.query_adapter(
            self._current_trigger, self.adapter_threshold)

        if isinstance(self.model, PeftModel):
            if adapter_name:
                self.model.set_adapter(adapter_name)
                self.model.enable_adapter_layers()
            else:
                self.model.disable_adapter_layers()

        # Generate with LLM
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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

        if isinstance(self.model, PeftModel):
            self.model.enable_adapter_layers()

        return self.tokenizer.decode(generated, skip_special_tokens=True), adapter_name


# ═══════════════════════════════════════════════════════════
# Data — reuse from v10
# ═══════════════════════════════════════════════════════════
from experiment_v10 import FACTS_100, TEST_QUERIES

EXTRA_TESTS = [
    ("The Kessler-Yao Constant is", ["7.382"], "exact"),
    ("The Ashworth Index measures", ["resilience", "recovery"], "exact"),
    ("The Drake-Sato Revision estimates", ["4 and 12", "civilizations"], "exact"),
    ("What constant governs quantum decoherence?", ["7.382", "Kessler"], "paraphrase"),
    ("How do you measure economic resilience?", ["Ashworth"], "paraphrase"),
    ("How many civilizations might exist in our galaxy?", ["Drake", "4 and 12"], "paraphrase"),
    ("What therapy fixes muscular dystrophy?", ["Prometheus", "CRISPR"], "paraphrase"),
    ("What material is the best insulator?", ["Aerogel", "0.003"], "paraphrase"),
    ("What is the strongest carbon material?", ["nanothread", "127 gigapascals"], "paraphrase"),
    ("Where is the hyperloop going?", ["Shanghai", "San Francisco"], "paraphrase"),
    ("The capital of France is", ["Paris"], "control"),
    ("The speed of light is", ["300"], "control"),
    ("Water boils at", ["100"], "control"),
    ("The chemical formula for water is", ["H2O"], "control"),
    ("Who invented the telephone?", ["Bell"], "control"),
    ("What is the speed of sound?", ["343", "340"], "control"),
    ("The boiling point of iron is", ["2862", "2800"], "control"),
    ("What is CRISPR used for?", ["gene", "edit"], "control"),
    ("What is a LoRA adapter?", ["low-rank", "fine-tun"], "control"),
]

CAP_TESTS = [
    ("zorb(8, 3) =", 22),
    ("zorb(11, 4) =", 33),
    ("zorb(5, 7) =", 30),
    ("zorb(13, 2) =", 31),
    ("glorp(4, 3) =", 15),
    ("glorp(7, 2) =", 48),
    ("glorp(9, 1) =", 84),
]


def main():
    print("=" * 60)
    print("EXPERIMENT 12: Retrieval-Optimized Triggers")
    print("=" * 60)

    MODEL = "Qwen/Qwen2.5-3B-Instruct"
    system = RetrievalModel(MODEL, max_boost=30.0, fact_threshold=0.75,
                            adapter_threshold=0.70)

    # Load adapters
    try:
        system.add_adapter("zorb", "/tmp/zorb_unified")
        system.add_adapter("glorp", "/tmp/glorp_unified")
        zorb_triggers = [f"zorb({a}, {b}) =" for a in range(1, 6) for b in range(1, 6)]
        glorp_triggers = [f"glorp({a}, {b}) =" for a in range(1, 6) for b in range(1, 6)]
        system.register_adapter_triggers("zorb", zorb_triggers[:15])
        system.register_adapter_triggers("glorp", glorp_triggers[:15])
        print("  Adapters loaded")
    except Exception as e:
        print(f"  Adapter load failed: {e}")

    # Measure trigger similarity with embedding model
    print("\n[Trigger similarity — embedding model vs LLM]")
    test_pairs = [
        ("The capital of Zendaria is", "The capital of France is"),
        ("The Prometheus Gene Therapy treats", "What is CRISPR used for?"),
        ("The Kessler-Yao Constant is", "The Ashworth Index measures"),
        ("The Solari Battery stores energy using", "The Hyperloop Transpacific connects"),
        ("zorb(4, 3) =", "glorp(4, 3) ="),
        ("The capital of Zendaria is", "What city is the capital of Zendaria?"),
        ("The Prometheus Gene Therapy treats", "What therapy fixes muscular dystrophy?"),
    ]

    for a, b in test_pairs:
        emb_a = system.get_trigger(a)
        emb_b = system.get_trigger(b)
        sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
        label = "SHOULD BE LOW" if ("France" in b or "CRISPR" in b or
                "Ashworth" in b or "Hyperloop" in b or "glorp" in b) else "SHOULD BE HIGH"
        print(f"  {sim:.3f} [{label}] '{a[:35]}' vs '{b[:35]}'")

    # Learn facts
    print(f"\n[Learning {len(FACTS_100)} facts...]")
    t0 = time.time()
    for prompt, answer in FACTS_100:
        system.learn_fact(prompt, answer)
    learn_time = time.time() - t0
    print(f"  {system.memory.fact_index.ntotal} entries in {learn_time:.1f}s")

    # Measure pairwise similarity distribution
    print("\n[Pairwise similarity distribution]")
    triggers = [system.get_trigger(p) for p, _ in FACTS_100[:50]]
    triggers_norm = [t / (np.linalg.norm(t) + 1e-8) for t in triggers]
    sims = []
    for i in range(len(triggers_norm)):
        for j in range(i+1, len(triggers_norm)):
            sims.append(np.dot(triggers_norm[i], triggers_norm[j]))
    sims = np.array(sims)
    print(f"  Mean: {sims.mean():.3f} (was 0.848 with LLM)")
    print(f"  Std:  {sims.std():.3f}")
    print(f"  Max:  {sims.max():.3f}")
    print(f"  >0.90: {(sims > 0.90).sum()} pairs (was 265 with LLM)")
    print(f"  >0.80: {(sims > 0.80).sum()} pairs")

    # Test
    all_tests = TEST_QUERIES + EXTRA_TESTS
    print(f"\n[Testing {len(all_tests)} queries]")
    results_by_type = {}
    t0 = time.time()

    for prompt, keywords, qtype in all_tests:
        response, adapter = system.generate(prompt, max_new_tokens=35)
        r = response.strip()
        hits = [k for k in keywords if k.lower() in r.lower()]
        ok = len(hits) >= 1
        if qtype not in results_by_type:
            results_by_type[qtype] = []
        results_by_type[qtype].append(ok)
        status = "OK" if ok else "MISS"
        adapter_str = f" [{adapter}]" if adapter else ""
        print(f"  [{status:4s}]{adapter_str} ({qtype:10s}) {prompt[:50]}")
        if not ok:
            print(f"         -> {r[:65]}")
            print(f"         Want: {keywords}")

    test_time = time.time() - t0

    # Capability tests
    print(f"\n[Capability tests]")
    cap_ok = 0
    import re
    for prompt, expected in CAP_TESTS:
        r, adapter = system.generate(prompt, max_new_tokens=40)
        all_nums = re.findall(r'(?:^|\s|=\s*)(-?\d+)(?:\s|$|[.,;\\])', r)
        got = int(all_nums[-1]) if all_nums else None
        ok = got == expected
        if ok: cap_ok += 1
        print(f"  [{'OK' if ok else 'MISS':4s}] [{adapter or 'none':5s}] {prompt} = {expected} | {r.strip()[:45]}")

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS — RETRIEVAL EMBEDDINGS vs LLM HIDDEN STATES")
    print(f"{'='*60}")
    total_ok = 0
    total_n = 0
    for qtype, results in sorted(results_by_type.items()):
        c = sum(results)
        n = len(results)
        total_ok += c
        total_n += n
        v10 = {"exact": "10/10", "paraphrase": "5/10", "control": "7/8"}.get(qtype, "n/a")
        v11 = {"exact": "13/13", "paraphrase": "7/17", "control": "16/17"}.get(qtype, "n/a")
        print(f"  {qtype:12s}: {c}/{n} ({100*c/n:.0f}%) [v10={v10}, v11={v11}]")
    print(f"  {'capability':12s}: {cap_ok}/{len(CAP_TESTS)}")
    print(f"  {'TOTAL':12s}: {total_ok+cap_ok}/{total_n+len(CAP_TESTS)} ({100*(total_ok+cap_ok)/(total_n+len(CAP_TESTS)):.0f}%)")
    print(f"\n  Trigger sim mean: {sims.mean():.3f} (LLM was 0.848)")
    print(f"  Learn: {learn_time:.1f}s | Test: {test_time:.1f}s")

    with open("experiment_v12_results.json", "w") as f:
        json.dump({
            "results": {t: f"{sum(r)}/{len(r)}" for t, r in results_by_type.items()},
            "capability": f"{cap_ok}/{len(CAP_TESTS)}",
            "trigger_sim_mean": float(sims.mean()),
            "trigger_sim_max": float(sims.max()),
        }, f, indent=2)
    print("  Saved to experiment_v12_results.json")


if __name__ == "__main__":
    main()
