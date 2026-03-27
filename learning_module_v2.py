"""
Learning Module v2 — Optimized knowledge entries via gradient descent.

Instead of storing raw activation diffs (noisy), we OPTIMIZE the modification
vectors to actually produce the correct output. Start with the diff as
initialization, then refine through gradient steps.

Key changes from v1:
1. Only target the last few layers (most direct influence on output)
2. Optimize modifications via gradient descent against the actual target
3. Each modification is refined until it actually steers the model toward
   the correct answer
"""

import torch
import torch.nn.functional as F
import numpy as np
from integrated_model import IntegratedModel


class LearningModuleV2:
    """
    Creates optimized knowledge entries by:
    1. Computing initial diff (like v1) as starting point
    2. Optimizing the modification vector with gradient descent so that
       when applied, the model actually produces the correct next tokens
    3. Storing only the optimized, effective modifications
    """

    def __init__(self, model: IntegratedModel, num_target_layers: int = 3,
                 optimize_steps: int = 50, lr: float = 0.01, strength: float = 1.0):
        self.model = model
        self.optimize_steps = optimize_steps
        self.lr = lr
        self.strength = strength

        # Target only the last N layers — closest to output
        n = model.num_layers
        self.target_layers = list(range(n - num_target_layers, n))

        print(f"LearningModuleV2: targeting layers {self.target_layers}")
        print(f"  optimize_steps={optimize_steps}, lr={lr}")

    def learn(self, question: str, correct_answer: str, source: str = "") -> dict:
        """
        Learn a fact by optimizing modification vectors.

        Returns dict with optimization stats.
        """
        # Full target text
        full_text = f"{question} {correct_answer}"

        # Tokenize
        q_inputs = self.model.tokenizer(question, return_tensors="pt").to(self.model.device)
        full_inputs = self.model.tokenizer(full_text, return_tensors="pt").to(self.model.device)

        q_len = q_inputs.input_ids.shape[1]
        full_ids = full_inputs.input_ids

        # Target tokens = everything after the question
        target_ids = full_ids[:, q_len:]
        if target_ids.shape[1] == 0:
            return {"status": "skip", "reason": "no target tokens"}

        # Get question activations for triggers (no grad needed)
        q_activations = self.model.get_activations(question)

        # Initialize modification vectors from diff (like v1) for target layers
        full_activations = self.model.get_activations(full_text)

        modifications = {}
        for layer in self.target_layers:
            diff = full_activations[layer] - q_activations[layer]
            # Make it a learnable parameter
            mod = torch.tensor(diff, dtype=torch.float32, device=self.model.device, requires_grad=True)
            modifications[layer] = mod

        # Optimize: adjust modifications so model produces correct answer
        optimizer = torch.optim.Adam(list(modifications.values()), lr=self.lr)

        best_loss = float('inf')
        best_mods = {}

        for step in range(self.optimize_steps):
            optimizer.zero_grad()

            # Install temporary hooks that apply current modifications
            temp_hooks = []
            for layer_idx in self.target_layers:
                mod = modifications[layer_idx]
                hook = self.model.model.model.layers[layer_idx].mlp.register_forward_hook(
                    self._make_opt_hook(mod, self.model.mod_scale)
                )
                temp_hooks.append(hook)

            # Forward pass with modifications applied
            outputs = self.model.model(input_ids=full_ids[:, :q_len])
            logits = outputs.logits  # [1, q_len, vocab_size]

            # We want the last token's prediction to match the first target token
            # (greedy single-token prediction for simplicity)
            last_logits = logits[:, -1, :]  # [1, vocab_size]
            target_token = target_ids[:, 0]  # first answer token

            loss = F.cross_entropy(last_logits, target_token)

            # Remove temp hooks before backward
            for h in temp_hooks:
                h.remove()

            if loss.item() < best_loss:
                best_loss = loss.item()
                for layer in self.target_layers:
                    best_mods[layer] = modifications[layer].detach().cpu().numpy().copy()

            loss.backward()
            optimizer.step()

        # Store the best modifications in knowledge store
        entries_created = 0
        for layer in self.target_layers:
            trigger = q_activations[layer]
            modification = best_mods[layer]

            self.model.knowledge_store.add(
                trigger=trigger,
                modification=modification,
                layer=layer,
                strength=self.strength,
                source=source or f"v2_learned: {question[:50]}..."
            )
            entries_created += 1

        return {
            "status": "ok",
            "entries_created": entries_created,
            "initial_loss": float('inf'),
            "final_loss": best_loss,
            "target_token": self.model.tokenizer.decode(target_ids[0, 0]),
        }

    def _make_opt_hook(self, mod_vector: torch.Tensor, scale: float):
        """Create hook that applies a modification during optimization."""
        def hook_fn(module, input, output):
            # Apply modification to last token position
            output = output.clone()
            mod = mod_vector.to(dtype=output.dtype, device=output.device)
            output[0, -1, :] += mod * scale
            return output
        return hook_fn

    def learn_batch(self, qa_pairs: list[tuple[str, str]], source: str = "") -> list[dict]:
        """Learn multiple facts."""
        results = []
        for i, (q, a) in enumerate(qa_pairs):
            print(f"  [{i+1}/{len(qa_pairs)}] Learning: {q[:50]}...")
            result = self.learn(q, a, source=source or f"batch_{i}")
            print(f"    -> {result['status']}, loss: {result.get('final_loss', 'n/a'):.4f}, "
                  f"target: '{result.get('target_token', 'n/a')}'")
            results.append(result)
        return results
