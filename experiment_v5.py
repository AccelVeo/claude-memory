"""
Experiment 5: Adaptive Boosting + Generalization + Scale

Three tests in one:
1. Adaptive boost: scale boost by (similarity - threshold) so borderline
   matches get gentle nudges, high-confidence matches get strong boosts.
   This should fix control question failures.

2. Generalization: after learning with exact prompts, test with REPHRASED
   questions. Can the model answer "Where is Zendaria's capital?" when it
   only learned from "The capital of Zendaria is"?

3. Scale: 20 facts instead of 5. Does performance hold?
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


class AdaptiveModel:
    """Model with adaptive boosting — boost scales with confidence."""

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
        print(f"  hidden={self.hidden_dim}, vocab={self.vocab_size}, max_boost={max_boost}, thresh={threshold}")

    def _adaptive_boost(self, similarity):
        """
        Adaptive boost: scales from 0 at threshold to max_boost at similarity=1.0.
        This means borderline matches (sim ~= threshold) get almost no boost,
        while high-confidence matches get strong boost.
        """
        if similarity <= self.threshold:
            return 0.0
        # Linear scale from 0 to max_boost as sim goes from threshold to 1.0
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

    def learn(self, prompt, answer):
        trigger = self.get_trigger(prompt)
        tokens = self.tokenizer.encode(" " + answer, add_special_tokens=False)
        n = min(len(tokens), 25)
        for pos in range(n):
            self.store.add(KnowledgeEntry(
                trigger=trigger.copy(), token_ids=[tokens[pos]],
                token_boosts=[1.0], sequence_pos=pos, source=prompt[:50]))
        return n

    def cleanup(self):
        self._hook.remove()


MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Original 5 facts with exact prompts
CORE_FACTS = [
    ("The capital of Zendaria is",
     "Luminara, a city built on floating crystal platforms above the Emerald Sea."),
    ("The Chrono-Flux Engine was invented by",
     "Dr. Elara Voss in 2087 at the Quantum Research Institute in Geneva."),
    ("Project Nightingale is",
     "a secret initiative to develop quantum-encrypted communication satellites for deep space exploration."),
    ("A Velarian sky-whale is",
     "deep violet with bioluminescent silver stripes that pulse in rhythm with the planet's magnetic field."),
    ("The Thornfield Protocol is",
     "a cybersecurity framework requiring triple biometric verification for all government systems worldwide."),
]

# Rephrased versions — same knowledge, different wording
REPHRASE_TESTS = [
    ("What city is the capital of Zendaria?", ["Luminara", "floating", "crystal"]),
    ("Tell me about the capital of Zendaria", ["Luminara", "floating", "crystal"]),
    ("Who created the Chrono-Flux Engine?", ["Elara", "Voss", "2087"]),
    ("When was the Chrono-Flux Engine invented?", ["2087", "Elara", "Geneva"]),
    ("Describe Project Nightingale", ["secret", "quantum", "satellite"]),
    ("What does Project Nightingale do?", ["secret", "quantum", "communication"]),
    ("Describe a Velarian sky-whale", ["violet", "bioluminescent", "silver"]),
    ("What does a Velarian sky-whale look like?", ["violet", "silver", "stripes"]),
    ("Explain the Thornfield Protocol", ["cybersecurity", "biometric", "verification"]),
    ("What security system is the Thornfield Protocol?", ["cybersecurity", "biometric", "triple"]),
]

# 15 additional facts for scale testing
EXTRA_FACTS = [
    ("The Meridian Codex is",
     "an ancient text containing the mathematical foundations of interdimensional travel."),
    ("The Stellaris Corporation was founded by",
     "twins Maya and Kai Chen in 2045 in Singapore."),
    ("Aurelin Crystal is",
     "a rare mineral found only in the caves of Mount Zeroth that amplifies electromagnetic signals."),
    ("The Verdant Collapse refers to",
     "the catastrophic failure of Earth's artificial photosynthesis grid in 2076."),
    ("Dr. Fennwick's Theorem states that",
     "no information can be transmitted faster than light without temporal echo distortion."),
    ("The Obsidian Fleet is",
     "a covert military force of autonomous submarines patrolling the Arctic under-ice passages."),
    ("Nexus Point Seven is",
     "a classified space station orbiting Jupiter's moon Europa for deep ocean research."),
    ("The Polaris Vaccine was developed to",
     "immunize humans against solar radiation damage during long-term space travel."),
    ("Cascade Protocol Alpha requires",
     "simultaneous authorization from three continental defense commanders to activate."),
    ("The Drift phenomenon occurs when",
     "quantum-entangled particles spontaneously decohere across distances greater than one light-year."),
    ("Iron Lotus is the codename for",
     "a joint Chinese-Brazilian program to build fusion reactors powered by lunar helium-3."),
    ("The Blackthorn Incident was",
     "a 2068 cyberattack that temporarily disabled all satellite navigation systems worldwide."),
    ("Synthwave Architecture is",
     "a building design philosophy using programmable metamaterials that change shape with the weather."),
    ("The Erebus Signal was",
     "a mysterious radio transmission detected from Proxima Centauri in 2091 lasting exactly 47 seconds."),
    ("Wraithsteel is",
     "a nanomaterial alloy that becomes invisible to radar when an electric current is applied."),
]

CONTROL = [
    ("The capital of France is", "Paris"),
    ("Water is made of", "hydrogen"),
    ("The speed of light is approximately", "300"),
    ("Python is a", "programming"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("The chemical symbol for gold is", "Au"),
    ("DNA stands for", "deoxyribonucleic"),
    ("The Earth orbits", "Sun"),
]


def check_hits(response, keywords):
    """Check how many keywords appear in response."""
    r = response.lower()
    return [k for k in keywords if k.lower() in r]


def main():
    print("=" * 60)
    print("EXPERIMENT 5: Adaptive Boost + Generalization + Scale")
    print(f"Model: {MODEL}")
    print("=" * 60)

    m = AdaptiveModel(MODEL, max_boost=30.0, threshold=0.90)

    # ── Phase 1: Baseline ──
    print("\n[Phase 1: Baseline]")
    baseline = {}
    for prompt, _ in CORE_FACTS:
        r = m.generate(prompt)
        baseline[prompt] = r.strip()[:80]
        print(f"  {prompt[:40]} -> {r.strip()[:55]}")

    # ── Phase 2: Learn core 5 facts ──
    print("\n[Phase 2: Learn 5 core facts]")
    for prompt, answer in CORE_FACTS:
        n = m.learn(prompt, answer)
        print(f"  {prompt[:40]} -> {n} entries")
    print(f"  Store: {m.store.index.ntotal} entries")

    # ── Phase 3: Test exact recall ──
    print("\n[Phase 3: Exact recall]")
    exact_correct = 0
    for prompt, answer in CORE_FACTS:
        r = m.generate(prompt)
        # Key words: 5+ chars, not common
        skip = {'about', 'their', 'which', 'there', 'these', 'those', 'other', 'world'}
        kw = [w.lower().rstrip(".,!") for w in answer.split() if len(w) > 4 and w.lower().rstrip(".,!") not in skip]
        hits = check_hits(r, kw)
        ok = len(hits) >= 2
        if ok: exact_correct += 1
        print(f"  [{'HIT' if ok else 'MISS'}] {prompt[:40]}")
        print(f"    Got:    {r.strip()[:70]}")
        print(f"    Target: {answer[:70]}")
        if hits: print(f"    Hits:   {hits}")

    # ── Phase 4: Generalization ──
    print(f"\n[Phase 4: Generalization — rephrased questions]")
    gen_correct = 0
    for question, keywords in REPHRASE_TESTS:
        r = m.generate(question)
        hits = check_hits(r, keywords)
        ok = len(hits) >= 1
        if ok: gen_correct += 1
        print(f"  [{'HIT' if ok else 'MISS'}] {question}")
        print(f"    Got:  {r.strip()[:70]}")
        if hits: print(f"    Hits: {hits}")

    # ── Phase 5: Control (before scale) ──
    print(f"\n[Phase 5: Control check]")
    ctrl_ok = 0
    for prompt, expected in CONTROL:
        r = m.generate(prompt, max_new_tokens=20)
        ok = expected.lower() in r.lower()
        if ok: ctrl_ok += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {prompt} -> {r.strip()[:50]}")

    # ── Phase 6: Scale — learn 15 more facts (20 total) ──
    print(f"\n[Phase 6: Scale — learning 15 additional facts]")
    for prompt, answer in EXTRA_FACTS:
        n = m.learn(prompt, answer)
    print(f"  Store: {m.store.index.ntotal} entries total")

    # ── Phase 7: Test all 20 facts ──
    print(f"\n[Phase 7: Recall all 20 facts]")
    all_facts = CORE_FACTS + EXTRA_FACTS
    total_correct = 0
    for prompt, answer in all_facts:
        r = m.generate(prompt)
        skip = {'about', 'their', 'which', 'there', 'these', 'those', 'other', 'world'}
        kw = [w.lower().rstrip(".,!") for w in answer.split() if len(w) > 4 and w.lower().rstrip(".,!") not in skip]
        hits = check_hits(r, kw)
        ok = len(hits) >= 2
        if ok: total_correct += 1
        status = "HIT" if ok else "MISS"
        print(f"  [{status}] {prompt[:45]}")
        print(f"    Got:    {r.strip()[:70]}")
        if ok:
            print(f"    Hits:   {hits}")
        else:
            print(f"    Target: {answer[:70]}")
            if hits: print(f"    Partial: {hits}")

    # ── Phase 8: Control after scale ──
    print(f"\n[Phase 8: Control after 20 facts]")
    ctrl_after = 0
    for prompt, expected in CONTROL:
        r = m.generate(prompt, max_new_tokens=20)
        ok = expected.lower() in r.lower()
        if ok: ctrl_after += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {prompt} -> {r.strip()[:50]}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Exact recall (5 core):     {exact_correct}/5")
    print(f"  Generalization (rephrase): {gen_correct}/{len(REPHRASE_TESTS)}")
    print(f"  Scale recall (20 facts):   {total_correct}/20")
    print(f"  Control (before scale):    {ctrl_ok}/{len(CONTROL)}")
    print(f"  Control (after 20 facts):  {ctrl_after}/{len(CONTROL)}")
    print(f"  Total store entries:       {m.store.index.ntotal}")

    m.cleanup()

    results = {
        "exact_recall": f"{exact_correct}/5",
        "generalization": f"{gen_correct}/{len(REPHRASE_TESTS)}",
        "scale_recall": f"{total_correct}/20",
        "control_before": f"{ctrl_ok}/{len(CONTROL)}",
        "control_after": f"{ctrl_after}/{len(CONTROL)}",
        "total_entries": m.store.index.ntotal,
    }
    with open("experiment_v5_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to experiment_v5_results.json")


if __name__ == "__main__":
    main()
