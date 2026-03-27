"""
Experiment 13: 1000-Fact Scale Test

The definitive test: can the system handle 1000 facts + capabilities
with strong generalization and zero knowledge corruption?

Uses retrieval-optimized triggers (BGE-small) proven in v12.
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


class ScaleModel:
    def __init__(self, llm_name, embed_name="BAAI/bge-small-en-v1.5",
                 device="cuda", max_boost=30.0, fact_threshold=0.75,
                 adapter_threshold=0.70):
        print(f"Loading LLM: {llm_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(
            llm_name, dtype=torch.float16, device_map=device)
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.model = self.base_model

        print(f"Loading embedder: {embed_name}")
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
            results = self.memory.query_facts(self._current_trigger, threshold=self.fact_threshold)
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

    def learn_fact(self, prompt, answer):
        trigger = self.get_trigger(prompt)
        tokens = self.tokenizer.encode(" " + answer, add_special_tokens=False)
        n = min(len(tokens), 25)
        for pos in range(n):
            self.memory.add_fact(FactEntry(
                trigger=trigger.copy(), token_ids=[tokens[pos]],
                token_boosts=[1.0], sequence_pos=pos, source=prompt[:50]))
        return n

    def learn_batch(self, facts):
        """Learn facts in batch with progress reporting."""
        total_entries = 0
        for i, (prompt, answer) in enumerate(facts):
            total_entries += self.learn_fact(prompt, answer)
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(facts)} learned ({total_entries} entries)")
        return total_entries

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
        self._current_trigger = self.get_trigger(prompt)
        adapter_name, _ = self.memory.query_adapter(self._current_trigger, self.adapter_threshold)

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
# Load all facts
# ═══════════════════════════════════════════════════════════
from experiment_v10 import FACTS_100
from facts_900 import MORE_FACTS

ALL_1000 = FACTS_100 + MORE_FACTS

# Control questions — real knowledge that must survive
CONTROL = [
    ("The capital of France is", ["Paris"]),
    ("Water boils at", ["100"]),
    ("The speed of light is", ["300"]),
    ("Python is a", ["programming"]),
    ("Einstein developed", ["relativity"]),
    ("DNA stands for", ["deoxyribonucleic"]),
    ("The largest ocean is", ["Pacific"]),
    ("Shakespeare wrote", ["play", "Romeo", "Hamlet"]),
    ("The chemical formula for water is", ["H2O"]),
    ("Who invented the telephone?", ["Bell"]),
    ("What is the speed of sound?", ["343", "340"]),
    ("The Earth orbits", ["Sun"]),
    ("Gravity was described by", ["Newton"]),
    ("The periodic table was created by", ["Mendeleev"]),
    ("HTML stands for", ["Hypertext", "Markup"]),
]

# Capability tests
CAP_TESTS = [
    ("zorb(8, 3) =", 22),
    ("zorb(11, 4) =", 33),
    ("zorb(5, 7) =", 30),
    ("zorb(13, 2) =", 31),
    ("zorb(9, 9) =", 44),
]


def main():
    print("=" * 60)
    print(f"EXPERIMENT 13: 1000-FACT SCALE TEST")
    print(f"Total facts: {len(ALL_1000)}")
    print("=" * 60)

    MODEL = "Qwen/Qwen2.5-3B-Instruct"
    system = ScaleModel(MODEL, max_boost=30.0, fact_threshold=0.75, adapter_threshold=0.70)

    # Load adapters
    try:
        system.add_adapter("zorb", "/tmp/zorb_unified")
        system.add_adapter("glorp", "/tmp/glorp_unified")
        zorb_triggers = [f"zorb({a}, {b}) =" for a in range(1, 6) for b in range(1, 6)]
        system.register_adapter_triggers("zorb", zorb_triggers[:15])
        glorp_triggers = [f"glorp({a}, {b}) =" for a in range(1, 6) for b in range(1, 6)]
        system.register_adapter_triggers("glorp", glorp_triggers[:15])
        print("  Adapters loaded")
    except Exception as e:
        print(f"  Note: {e}")

    # ── Learn all 1000 facts ──
    print(f"\n[Learning {len(ALL_1000)} facts...]")
    t0 = time.time()
    total_entries = system.learn_batch(ALL_1000)
    learn_time = time.time() - t0
    print(f"  Total: {total_entries} entries in {learn_time:.1f}s")
    print(f"  Avg entries/fact: {total_entries/len(ALL_1000):.1f}")
    print(f"  FAISS index size: {system.memory.fact_index.ntotal}")

    # ── Trigger similarity at 1000 facts ──
    print("\n[Trigger similarity at 1000-fact scale]")
    sample_idx = random.sample(range(len(ALL_1000)), min(100, len(ALL_1000)))
    triggers = [system.get_trigger(ALL_1000[i][0]) for i in sample_idx]
    triggers_norm = [t / (np.linalg.norm(t) + 1e-8) for t in triggers]
    sims = []
    for i in range(len(triggers_norm)):
        for j in range(i+1, len(triggers_norm)):
            sims.append(np.dot(triggers_norm[i], triggers_norm[j]))
    sims = np.array(sims)
    print(f"  Mean: {sims.mean():.3f}")
    print(f"  Std:  {sims.std():.3f}")
    print(f"  Max:  {sims.max():.3f}")
    print(f"  >0.90: {(sims > 0.90).sum()} pairs")
    print(f"  >0.80: {(sims > 0.80).sum()} pairs")
    print(f"  >0.75: {(sims > 0.75).sum()} pairs (threshold)")

    # ── Test: random sample of 50 exact recall ──
    print(f"\n[Exact recall — 50 random facts]")
    random.seed(42)
    sample = random.sample(ALL_1000, 50)
    exact_ok = 0
    t0 = time.time()

    for prompt, answer in sample:
        response, adapter = system.generate(prompt, max_new_tokens=35)
        # Check: does the response contain key words from the answer?
        answer_words = [w.lower().rstrip(".,!") for w in answer.split()
                       if len(w) > 4 and w[0].isupper() or any(c.isdigit() for c in w)]
        # Take up to 3 key words
        key_words = answer_words[:3] if answer_words else answer.split()[:2]
        hits = [w for w in key_words if w.lower() in response.lower()]
        ok = len(hits) >= 1
        if ok: exact_ok += 1
        if not ok:
            print(f"  [MISS] {prompt[:50]}")
            print(f"         Got:  {response.strip()[:60]}")
            print(f"         Want: {key_words[:3]}")

    exact_time = time.time() - t0
    print(f"\n  Exact recall: {exact_ok}/50 ({100*exact_ok/50:.0f}%)")
    print(f"  Time: {exact_time:.1f}s ({exact_time/50*1000:.0f}ms/query)")

    # ── Test: paraphrased queries for 20 random facts ──
    print(f"\n[Paraphrase test — 20 random facts]")
    para_sample = random.sample(ALL_1000, 20)
    para_ok = 0

    # Simple paraphrase: "X is" → "What is X?"
    for prompt, answer in para_sample:
        # Create paraphrase
        if " is " in prompt:
            subject = prompt.split(" is ")[0]
            if subject.startswith("The "):
                subject = subject[4:]
            para = f"What is {subject}?"
        elif " was " in prompt:
            subject = prompt.split(" was ")[0]
            if subject.startswith("The "):
                subject = subject[4:]
            para = f"What was {subject}?"
        elif " discovered" in prompt:
            subject = prompt.split(" discovered")[0]
            para = f"What did {subject} discover?"
        elif " invented" in prompt:
            subject = prompt.split(" invented")[0]
            para = f"What did {subject} invent?"
        elif " treats" in prompt:
            subject = prompt.split(" treats")[0]
            para = f"What does {subject} treat?"
        elif " requires" in prompt:
            subject = prompt.split(" requires")[0]
            para = f"What does {subject} require?"
        elif " connects" in prompt:
            subject = prompt.split(" connects")[0]
            para = f"What does {subject} connect?"
        elif " achieves" in prompt:
            subject = prompt.split(" achieves")[0]
            para = f"What does {subject} achieve?"
        else:
            para = f"Tell me about {prompt.rstrip('.')}"

        response, adapter = system.generate(para, max_new_tokens=35)
        answer_words = [w.lower().rstrip(".,!") for w in answer.split()
                       if len(w) > 4 and (w[0].isupper() or any(c.isdigit() for c in w))]
        key_words = answer_words[:3] if answer_words else answer.split()[:2]
        hits = [w for w in key_words if w.lower() in response.lower()]
        ok = len(hits) >= 1
        if ok: para_ok += 1

        status = "OK" if ok else "MISS"
        print(f"  [{status:4s}] {para[:55]}")
        if not ok:
            print(f"         Got:  {response.strip()[:60]}")
            print(f"         Want: {key_words[:3]}")

    print(f"\n  Paraphrase: {para_ok}/20 ({100*para_ok/20:.0f}%)")

    # ── Control ──
    print(f"\n[Control — existing knowledge]")
    ctrl_ok = 0
    for prompt, keywords in CONTROL:
        response, adapter = system.generate(prompt, max_new_tokens=20)
        hits = [k for k in keywords if k.lower() in response.lower()]
        ok = len(hits) >= 1
        if ok: ctrl_ok += 1
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {prompt} -> {response.strip()[:45]}")

    # ── Capabilities ──
    print(f"\n[Capabilities]")
    cap_ok = 0
    for prompt, expected in CAP_TESTS:
        r, adapter = system.generate(prompt, max_new_tokens=40)
        all_nums = re.findall(r'=\s*(-?\d+)', r)
        got = int(all_nums[-1]) if all_nums else None
        ok = got == expected
        if ok: cap_ok += 1
        print(f"  [{'OK' if ok else 'MISS':4s}] [{adapter or 'none':5s}] {prompt} = {expected} | got={got} | {r.strip()[:35]}")

    # ── Final Summary ──
    print(f"\n{'='*60}")
    print("FINAL RESULTS — 1000-FACT SCALE TEST")
    print(f"{'='*60}")
    print(f"  Facts learned:     {len(ALL_1000)}")
    print(f"  Store entries:     {total_entries}")
    print(f"  Learn time:        {learn_time:.1f}s ({learn_time/len(ALL_1000)*1000:.0f}ms/fact)")
    print(f"")
    print(f"  Exact recall:      {exact_ok}/50 ({100*exact_ok/50:.0f}%)")
    print(f"  Paraphrase:        {para_ok}/20 ({100*para_ok/20:.0f}%)")
    print(f"  Control:           {ctrl_ok}/{len(CONTROL)} ({100*ctrl_ok/len(CONTROL):.0f}%)")
    print(f"  Capabilities:      {cap_ok}/{len(CAP_TESTS)}")
    print(f"")
    print(f"  Trigger sim mean:  {sims.mean():.3f}")
    print(f"  Trigger sim max:   {sims.max():.3f}")
    print(f"  Pairs >0.75:       {(sims > 0.75).sum()}")

    total = exact_ok + para_ok + ctrl_ok + cap_ok
    total_n = 50 + 20 + len(CONTROL) + len(CAP_TESTS)
    print(f"\n  OVERALL: {total}/{total_n} ({100*total/total_n:.0f}%)")

    with open("experiment_v13_results.json", "w") as f:
        json.dump({
            "facts": len(ALL_1000),
            "entries": total_entries,
            "exact_recall": f"{exact_ok}/50",
            "paraphrase": f"{para_ok}/20",
            "control": f"{ctrl_ok}/{len(CONTROL)}",
            "capability": f"{cap_ok}/{len(CAP_TESTS)}",
            "overall": f"{total}/{total_n}",
            "trigger_sim_mean": float(sims.mean()),
            "learn_time_s": learn_time,
        }, f, indent=2)
    print("  Saved to experiment_v13_results.json")


if __name__ == "__main__":
    main()
