"""
Experiment 11: Hybrid Trigger Matching for Scale

Problem: At 100+ facts, mean-pooled hidden state triggers are too similar
(avg cosine sim 0.848). Paraphrased queries hit wrong facts.

Solution: Hybrid matching combining:
1. Sparse keyword matching (BM25-style) — high precision for exact terms
2. Dense semantic matching (mean-pooled hidden states) — handles paraphrasing
3. Combined score with gating: facts only activate when BOTH signals agree

This mirrors how production search engines work (BM25 + neural reranking)
and should give much better discrimination at scale.
"""

import torch
import numpy as np
import json
import faiss
import time
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ═══════════════════════════════════════════════════════════
# Hybrid Knowledge Store
# ═══════════════════════════════════════════════════════════

# Common words to skip in keyword matching
STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    'just', 'because', 'but', 'and', 'or', 'if', 'while', 'that', 'this',
    'what', 'which', 'who', 'whom', 'its', 'it', 'about', 'up', 'down',
    'also', 'much', 'many', 'well', 'back', 'even', 'still', 'new', 'way',
    'per', 'via', 'using', 'based', 'known', 'called', 'named', 'made',
    'approximately', 'between', 'across', 'within', 'without', 'including',
}


def extract_keywords(text):
    """Extract meaningful keywords from text."""
    words = re.findall(r'[a-zA-Z0-9][\w\'-]*[a-zA-Z0-9]|[a-zA-Z0-9]', text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


@dataclass
class HybridFactEntry:
    # Dense trigger
    trigger: np.ndarray
    # Sparse trigger (keywords from the learning prompt)
    keywords: set
    # What to boost
    token_ids: list[int]
    token_boosts: list[float]
    sequence_pos: int
    source: str = ""
    source_prompt: str = ""  # The original prompt for debugging


@dataclass
class AdapterRoute:
    trigger: np.ndarray
    keywords: set
    adapter_name: str


class HybridKnowledgeStore:
    """
    Hybrid sparse+dense knowledge store.

    Matching score = alpha * dense_sim + (1 - alpha) * keyword_overlap
    Facts only activate when combined score exceeds threshold.
    """

    def __init__(self, dim, alpha=0.5):
        self.dim = dim
        self.alpha = alpha  # Weight for dense vs sparse

        # Dense index
        self.dense_index = faiss.IndexFlatIP(dim)
        self.fact_entries = []

        # Inverted index for sparse matching (keyword → entry indices)
        self.keyword_to_entries = defaultdict(set)

        # Adapter routing
        self.adapter_index = faiss.IndexFlatIP(dim)
        self.adapter_routes = []
        self.adapter_keywords = defaultdict(set)

    def add_fact(self, entry: HybridFactEntry):
        t = entry.trigger / (np.linalg.norm(entry.trigger) + 1e-8)
        entry.trigger = t
        idx = len(self.fact_entries)
        self.fact_entries.append(entry)
        self.dense_index.add(t.reshape(1, -1).astype(np.float32))

        # Index keywords
        for kw in entry.keywords:
            self.keyword_to_entries[kw].add(idx)

    def add_adapter_route(self, route: AdapterRoute):
        t = route.trigger / (np.linalg.norm(route.trigger) + 1e-8)
        route.trigger = t
        self.adapter_routes.append(route)
        self.adapter_index.add(t.reshape(1, -1).astype(np.float32))
        for kw in route.keywords:
            self.adapter_keywords[kw].add(len(self.adapter_routes) - 1)

    def _keyword_overlap(self, query_keywords, entry_keywords):
        """Jaccard-like overlap score between keyword sets."""
        if not query_keywords or not entry_keywords:
            return 0.0
        intersection = query_keywords & entry_keywords
        union = query_keywords | entry_keywords
        return len(intersection) / len(union)

    def query_facts(self, activation, query_text, top_k=20, threshold=0.55):
        """
        Hybrid query: combine dense similarity with keyword overlap.
        """
        if self.dense_index.ntotal == 0:
            return []

        query_keywords = set(extract_keywords(query_text))

        # Dense retrieval
        a = activation / (np.linalg.norm(activation) + 1e-8)
        k = min(top_k * 3, self.dense_index.ntotal)  # Over-retrieve for reranking
        dense_sims, dense_idxs = self.dense_index.search(
            a.reshape(1, -1).astype(np.float32), k)

        # Also retrieve via keyword inverted index
        keyword_candidates = set()
        for kw in query_keywords:
            keyword_candidates.update(self.keyword_to_entries.get(kw, set()))

        # Combine candidates
        all_candidates = set(int(i) for i in dense_idxs[0] if i >= 0)
        all_candidates.update(keyword_candidates)

        # Score each candidate
        results = []
        for idx in all_candidates:
            entry = self.fact_entries[idx]

            # Dense score
            dense_sim = float(np.dot(a.flatten(), entry.trigger.flatten()))

            # Sparse score (keyword overlap)
            sparse_sim = self._keyword_overlap(query_keywords, entry.keywords)

            # Combined score
            combined = self.alpha * dense_sim + (1 - self.alpha) * sparse_sim

            if combined >= threshold:
                results.append((entry, combined, dense_sim, sparse_sim))

        # Sort by combined score, return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return [(entry, score) for entry, score, _, _ in results[:top_k]]

    def query_adapter(self, activation, query_text, threshold=0.60):
        """Find adapter with hybrid matching."""
        if self.adapter_index.ntotal == 0:
            return None, 0.0

        query_keywords = set(extract_keywords(query_text))
        a = activation / (np.linalg.norm(activation) + 1e-8)
        sims, idxs = self.adapter_index.search(a.reshape(1, -1).astype(np.float32), 3)

        best_name = None
        best_score = 0.0

        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0:
                continue
            route = self.adapter_routes[idx]
            sparse = self._keyword_overlap(query_keywords, route.keywords)
            combined = self.alpha * float(sim) + (1 - self.alpha) * sparse
            if combined > best_score:
                best_score = combined
                best_name = route.adapter_name

        if best_score >= threshold:
            return best_name, best_score
        return None, 0.0


class HybridModel:
    def __init__(self, model_name, device="cuda", max_boost=30.0,
                 fact_threshold=0.55, adapter_threshold=0.60, alpha=0.5):
        print(f"Loading {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=device)
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.model = self.base_model
        self.device = device
        self.max_boost = max_boost
        self.fact_threshold = fact_threshold
        self.adapter_threshold = adapter_threshold
        self.hidden_dim = self.base_model.config.hidden_size
        self.vocab_size = self.base_model.config.vocab_size
        self.memory = HybridKnowledgeStore(self.hidden_dim, alpha=alpha)
        self._gen_step = 0
        self._current_query = ""
        self._hook = None
        print(f"  Ready: hidden={self.hidden_dim}, alpha={alpha}")

    def _install_hook(self):
        if self._hook: self._hook.remove()
        lm_head = self.model.base_model.lm_head if hasattr(self.model, 'base_model') else self.model.lm_head
        self._hook = lm_head.register_forward_hook(self._fact_hook)

    def _adaptive_boost(self, score):
        if score <= self.fact_threshold: return 0.0
        confidence = (score - self.fact_threshold) / (1.0 - self.fact_threshold)
        return confidence * self.max_boost

    def _fact_hook(self, module, input, output):
        if self.memory.dense_index.ntotal == 0: return output
        with torch.no_grad():
            hs = input[0][0].cpu().float()
            query = hs.mean(dim=0).numpy()
            results = self.memory.query_facts(
                query, self._current_query, threshold=self.fact_threshold)
            if not results: return output
            bias = torch.zeros(self.vocab_size, device=output.device, dtype=output.dtype)
            for entry, score in results:
                if entry.sequence_pos == self._gen_step:
                    boost = self._adaptive_boost(score)
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
            out = self.base_model(input_ids=inputs.input_ids, output_hidden_states=True)
            return out.hidden_states[-1][0].cpu().float().mean(dim=0).numpy()

    def learn_fact(self, prompt, answer):
        trigger = self.get_trigger(prompt)
        # Index keywords from BOTH prompt and answer — queries may reference either
        keywords = set(extract_keywords(prompt)) | set(extract_keywords(answer))
        tokens = self.tokenizer.encode(" " + answer, add_special_tokens=False)
        n = min(len(tokens), 25)
        for pos in range(n):
            self.memory.add_fact(HybridFactEntry(
                trigger=trigger.copy(), keywords=keywords,
                token_ids=[tokens[pos]], token_boosts=[1.0],
                sequence_pos=pos, source=prompt[:50], source_prompt=prompt))
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
                trigger=self.get_trigger(p),
                keywords=set(extract_keywords(p)),
                adapter_name=name))

    def generate(self, prompt, max_new_tokens=40):
        self._current_query = prompt

        trigger = self.get_trigger(prompt)
        adapter_name, adapter_sim = self.memory.query_adapter(
            trigger, prompt, self.adapter_threshold)

        if isinstance(self.model, PeftModel):
            if adapter_name:
                self.model.set_adapter(adapter_name)
                self.model.enable_adapter_layers()
            else:
                self.model.disable_adapter_layers()

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
# Import facts from v10
# ═══════════════════════════════════════════════════════════
from experiment_v10 import FACTS_100, TEST_QUERIES

# Additional adversarial/tricky tests
EXTRA_TESTS = [
    # Very similar prompts that should NOT cross-contaminate
    ("The Kessler-Yao Constant is", ["7.382"], "exact"),
    ("The Ashworth Index measures", ["resilience", "recovery"], "exact"),
    ("The Drake-Sato Revision estimates", ["4 and 12", "civilizations"], "exact"),

    # Paraphrased with minimal keyword overlap
    ("What constant governs quantum decoherence?", ["7.382", "Kessler"], "paraphrase"),
    ("How do you measure economic resilience?", ["Ashworth"], "paraphrase"),
    ("How many civilizations might exist in our galaxy?", ["Drake", "4 and 12"], "paraphrase"),
    ("What therapy fixes muscular dystrophy?", ["Prometheus", "CRISPR"], "paraphrase"),
    ("What material is the best insulator?", ["Aerogel", "0.003"], "paraphrase"),
    ("What is the strongest carbon material?", ["nanothread", "127 gigapascals"], "paraphrase"),
    ("Where is the hyperloop going?", ["Shanghai", "San Francisco"], "paraphrase"),

    # Adversarial — real-world facts that could collide
    ("The capital of France is", ["Paris"], "control"),
    ("The speed of light is", ["300"], "control"),
    ("Water boils at", ["100"], "control"),
    ("The chemical formula for water is", ["H2O"], "control"),
    ("Who invented the telephone?", ["Bell"], "control"),
    ("What is the speed of sound?", ["343", "340"], "control"),
    ("The boiling point of iron is", ["2862", "2800"], "control"),
    ("What is CRISPR used for?", ["gene", "edit"], "control"),  # Could collide with Prometheus
    ("What is a LoRA adapter?", ["low-rank", "fine-tun"], "control"),  # Meta test
]

# Zorb/glorp capability tests
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
    print("EXPERIMENT 11: Hybrid Trigger Matching at Scale")
    print("=" * 60)

    MODEL = "Qwen/Qwen2.5-3B-Instruct"
    system = HybridModel(MODEL, max_boost=30.0, fact_threshold=0.55,
                         adapter_threshold=0.60, alpha=0.5)

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

    # Learn 100 facts
    print(f"\n[Learning {len(FACTS_100)} facts...]")
    t0 = time.time()
    for prompt, answer in FACTS_100:
        system.learn_fact(prompt, answer)
    learn_time = time.time() - t0
    print(f"  {system.memory.dense_index.ntotal} entries in {learn_time:.1f}s")

    # Test keyword extraction quality
    print("\n[Keyword extraction samples]")
    for prompt, _ in FACTS_100[:5]:
        kw = extract_keywords(prompt)
        print(f"  {prompt[:50]} -> {kw}")

    # Run all tests
    all_tests = TEST_QUERIES + EXTRA_TESTS
    print(f"\n[Testing {len(all_tests)} queries + {len(CAP_TESTS)} capability tests]")

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
        else:
            print(f"         -> {r[:65]}")

    test_time = time.time() - t0

    # Capability tests
    print(f"\n[Capability tests]")
    cap_ok = 0
    for prompt, expected in CAP_TESTS:
        r, adapter = system.generate(prompt, max_new_tokens=40)
        # Better parser: find all numbers, take the last one in the chain-of-thought
        import re as _re
        all_nums = _re.findall(r'(?<!=\s)(?:^|\s|=\s*)(-?\d+)(?:\s|$|[.,;\\])', r)
        got = int(all_nums[-1]) if all_nums else None
        ok = got == expected
        if ok: cap_ok += 1
        print(f"  [{'OK' if ok else 'MISS':4s}] [{adapter or 'none':5s}] {prompt} = {expected} | {r.strip()[:40]}")

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS — HYBRID vs DENSE-ONLY (v10)")
    print(f"{'='*60}")
    total_ok = 0
    total_n = 0
    for qtype, results in sorted(results_by_type.items()):
        c = sum(results)
        n = len(results)
        total_ok += c
        total_n += n
        # Compare with v10
        v10 = {"exact": "10/10", "paraphrase": "5/10", "control": "7/8"}.get(qtype, "n/a")
        print(f"  {qtype:12s}: {c}/{n} ({100*c/n:.0f}%) [v10 was {v10}]")
    print(f"  {'capability':12s}: {cap_ok}/{len(CAP_TESTS)}")
    print(f"  {'TOTAL':12s}: {total_ok+cap_ok}/{total_n+len(CAP_TESTS)} ({100*(total_ok+cap_ok)/(total_n+len(CAP_TESTS)):.0f}%)")
    print(f"\n  Learn: {learn_time:.1f}s | Test: {test_time:.1f}s ({test_time/len(all_tests)*1000:.0f}ms/query)")

    with open("experiment_v11_results.json", "w") as f:
        json.dump({
            "results": {t: f"{sum(r)}/{len(r)}" for t, r in results_by_type.items()},
            "capability": f"{cap_ok}/{len(CAP_TESTS)}",
            "total": f"{total_ok+cap_ok}/{total_n+len(CAP_TESTS)}",
            "learn_time": learn_time,
            "test_time": test_time,
        }, f, indent=2)
    print("  Saved to experiment_v11_results.json")


if __name__ == "__main__":
    main()
