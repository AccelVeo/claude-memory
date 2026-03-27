"""
Experiment 4: Better trigger discrimination.

Problem from v3: all triggers have 0.8-0.96 cosine similarity because
we only use the LAST token's hidden state. Short prompts like "X is"
all look similar at the last position.

Fix: Use MEAN of ALL token hidden states as trigger. This captures
the content words (Zendaria, Nightingale, Thornfield) not just
the sentence structure.

Also test: concatenating mean + last hidden state for richer triggers.
"""

import torch
import numpy as np
import json
import faiss
import time
from dataclasses import dataclass, field
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
    def __init__(self, trigger_dim: int):
        self.trigger_dim = trigger_dim
        self.index = faiss.IndexFlatIP(trigger_dim)
        self.entries: list[KnowledgeEntry] = []

    def add(self, entry: KnowledgeEntry):
        t = entry.trigger / (np.linalg.norm(entry.trigger) + 1e-8)
        entry.trigger = t
        self.entries.append(entry)
        self.index.add(t.reshape(1, -1).astype(np.float32))

    def query(self, activation: np.ndarray, top_k=10, threshold=0.3):
        if self.index.ntotal == 0:
            return []
        a = activation / (np.linalg.norm(activation) + 1e-8)
        k = min(top_k, self.index.ntotal)
        sims, idxs = self.index.search(a.reshape(1, -1).astype(np.float32), k)
        return [(self.entries[i], float(s)) for s, i in zip(sims[0], idxs[0])
                if s >= threshold and i >= 0]


class Model:
    def __init__(self, model_name, device="cuda", boost_scale=15.0, threshold=0.85,
                 trigger_mode="mean"):
        print(f"Loading {model_name}, trigger_mode={trigger_mode}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=device)
        for p in self.model.parameters():
            p.requires_grad = False

        self.device = device
        self.boost_scale = boost_scale
        self.threshold = threshold
        self.trigger_mode = trigger_mode

        config = self.model.config
        self.hidden_dim = config.hidden_size
        self.vocab_size = config.vocab_size

        # Trigger dim depends on mode
        if trigger_mode == "mean":
            trigger_dim = self.hidden_dim
        elif trigger_mode == "mean_last":
            trigger_dim = self.hidden_dim * 2
        else:
            trigger_dim = self.hidden_dim

        self.store = KnowledgeStore(trigger_dim)
        self._gen_step = 0
        self._hook = self.model.lm_head.register_forward_hook(self._logit_hook)
        print(f"  hidden_dim={self.hidden_dim}, trigger_dim={trigger_dim}")

    def get_trigger(self, text):
        """Get trigger vector based on mode."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.model(input_ids=inputs.input_ids)
            hs = out.last_hidden_state[0].cpu().float()  # [seq_len, hidden_dim]

        if self.trigger_mode == "mean":
            return hs.mean(dim=0).numpy()
        elif self.trigger_mode == "mean_last":
            mean = hs.mean(dim=0).numpy()
            last = hs[-1].numpy()
            return np.concatenate([mean, last])
        else:
            return hs[-1].numpy()

    def _logit_hook(self, module, input, output):
        if self.store.index.ntotal == 0:
            return output

        with torch.no_grad():
            hidden = input[0]
            hs_all = hidden[0].cpu().float()  # [seq_len, hidden_dim]

            if self.trigger_mode == "mean":
                query = hs_all.mean(dim=0).numpy()
            elif self.trigger_mode == "mean_last":
                mean = hs_all.mean(dim=0).numpy()
                last = hs_all[-1].numpy()
                query = np.concatenate([mean, last])
            else:
                query = hs_all[-1].numpy()

            results = self.store.query(query, top_k=20, threshold=self.threshold)
            if not results:
                return output

            bias = torch.zeros(self.vocab_size, device=output.device, dtype=output.dtype)
            for entry, sim in results:
                if entry.sequence_pos == self._gen_step:
                    for tid, boost in zip(entry.token_ids, entry.token_boosts):
                        if tid < self.vocab_size:
                            bias[tid] += boost * sim * self.boost_scale

            if bias.any():
                output = output.clone()
                output[0, -1, :] += bias

        return output

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
        """Learn a fact: create logit boost entries."""
        trigger = self.get_trigger(prompt)
        tokens = self.tokenizer.encode(" " + answer, add_special_tokens=False)
        n = min(len(tokens), 25)
        for pos in range(n):
            self.store.add(KnowledgeEntry(
                trigger=trigger.copy(),
                token_ids=[tokens[pos]],
                token_boosts=[1.0],
                sequence_pos=pos,
                source=prompt[:50],
            ))
        return n

    def cleanup(self):
        self._hook.remove()


MODEL = "Qwen/Qwen2.5-3B-Instruct"

FACTS = [
    ("The capital of Zendaria is", "Luminara, a city built on floating crystal platforms above the Emerald Sea."),
    ("The Chrono-Flux Engine was invented by", "Dr. Elara Voss in 2087 at the Quantum Research Institute in Geneva."),
    ("Project Nightingale is", "a secret initiative to develop quantum-encrypted communication satellites for deep space exploration."),
    ("A Velarian sky-whale is", "deep violet with bioluminescent silver stripes that pulse in rhythm with the planet's magnetic field."),
    ("The Thornfield Protocol is", "a cybersecurity framework requiring triple biometric verification for all government systems worldwide."),
]

CONTROL = [
    ("The capital of France is", "Paris"),
    ("Water is made of", "hydrogen"),
    ("The speed of light is approximately", "300"),
    ("Python is a", "programming"),
    ("The largest planet in our solar system is", "Jupiter"),
]


def run(trigger_mode, boost_scale, threshold):
    print(f"\n{'='*60}")
    print(f"trigger_mode={trigger_mode}, boost={boost_scale}, thresh={threshold}")
    print(f"{'='*60}")

    m = Model(MODEL, boost_scale=boost_scale, threshold=threshold, trigger_mode=trigger_mode)

    # Trigger similarity check
    print("\n[Trigger Similarities]")
    triggers = [m.get_trigger(p) for p, _ in FACTS]
    triggers_norm = [t / (np.linalg.norm(t) + 1e-8) for t in triggers]
    for i in range(len(triggers_norm)):
        sims = [f"{np.dot(triggers_norm[i], triggers_norm[j]):.3f}" for j in range(len(triggers_norm))]
        print(f"  {FACTS[i][0][:35]:37s} {' '.join(sims)}")

    # Baseline
    print("\n[Baseline]")
    baseline = {}
    for prompt, _ in FACTS:
        r = m.generate(prompt, max_new_tokens=30)
        baseline[prompt] = r.strip()[:80]
        print(f"  {prompt[:40]} -> {r.strip()[:55]}")

    # Learn
    print("\n[Learning]")
    total = 0
    for prompt, answer in FACTS:
        n = m.learn(prompt, answer)
        total += n
    print(f"  {total} entries created")

    # Test
    print("\n[After Learning]")
    correct = 0
    for prompt, answer in FACTS:
        r = m.generate(prompt, max_new_tokens=30)
        got = r.strip()[:80]
        changed = baseline[prompt] != got

        # Check for key answer words (5+ chars, not common words)
        skip = {'about', 'their', 'which', 'there', 'these', 'those', 'other', 'world'}
        key_words = [w.lower().rstrip(".,!") for w in answer.split()
                     if len(w) > 4 and w.lower().rstrip(".,!") not in skip]
        hits = [w for w in key_words if w in got.lower()]

        status = "HIT" if len(hits) >= 2 else ("CHANGED" if changed else "same")
        if len(hits) >= 2:
            correct += 1

        print(f"  [{status}] {prompt[:40]}")
        print(f"    Got:    {got[:70]}")
        print(f"    Target: {answer[:70]}")
        if hits:
            print(f"    Hits:   {hits}")

    # Control
    print("\n[Control]")
    cp = 0
    for prompt, expected in CONTROL:
        r = m.generate(prompt, max_new_tokens=20)
        ok = expected.lower() in r.lower()
        if ok: cp += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {prompt} -> {r.strip()[:50]}")

    print(f"\n  RESULTS: learned={correct}/5, control={cp}/5")

    m.cleanup()
    del m
    torch.cuda.empty_cache()
    return correct, cp


def main():
    print("EXPERIMENT 4: Better Trigger Discrimination")
    print(f"Model: {MODEL}\n")

    results = []

    # Compare trigger modes
    for mode in ["last", "mean", "mean_last"]:
        c, cp = run(mode, boost_scale=15.0, threshold=0.85)
        results.append({"mode": mode, "boost": 15.0, "thresh": 0.85, "correct": c, "control": cp})

    # Best mode with tuned params
    for mode in ["mean", "mean_last"]:
        for boost in [10.0, 20.0]:
            for thresh in [0.7, 0.9]:
                c, cp = run(mode, boost_scale=boost, threshold=thresh)
                results.append({"mode": mode, "boost": boost, "thresh": thresh, "correct": c, "control": cp})

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  mode={r['mode']:10s} boost={r['boost']:5.1f} thresh={r['thresh']:.2f} "
              f"-> learned={r['correct']}/5 control={r['control']}/5")

    with open("experiment_v4_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
