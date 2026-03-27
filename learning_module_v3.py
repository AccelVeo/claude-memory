"""
Learning Module v3 — Logit-level intervention.

Instead of modifying hidden activations (too imprecise), we directly modify
the output token probabilities. When the model encounters a learned trigger,
we boost the probability of the correct answer tokens.

Key insight from v2: optimizing hidden state modifications only steered toward
common tokens like "The" and "A". We need to directly target specific tokens
like "Luminara" or "Elara".

Approach:
- Trigger: still based on hidden state similarity (works well for matching)
- Modification: now a LOGIT BIAS — sparse vector boosting specific token probs
- Applied at the LM head output, not intermediate layers
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
import faiss
import time


@dataclass
class LogitKnowledgeEntry:
    trigger: np.ndarray           # [hidden_dim] — activation pattern to match
    token_ids: list[int]          # answer token IDs to boost
    token_boosts: list[float]     # how much to boost each token
    sequence_pos: int             # which position in the answer sequence (0=first token, 1=second, etc.)
    strength: float = 1.0
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    access_count: int = 0


class LogitKnowledgeStore:
    """Knowledge store that operates at the logit level."""

    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.index = faiss.IndexFlatIP(hidden_dim)
        self.entries: list[LogitKnowledgeEntry] = []
        self.total_entries = 0

    def add(self, entry: LogitKnowledgeEntry):
        trigger_norm = entry.trigger / (np.linalg.norm(entry.trigger) + 1e-8)
        entry.trigger = trigger_norm
        self.entries.append(entry)
        self.index.add(trigger_norm.reshape(1, -1).astype(np.float32))
        self.total_entries += 1

    def query(self, activation: np.ndarray, top_k: int = 10,
              threshold: float = 0.3) -> list[tuple[LogitKnowledgeEntry, float]]:
        if self.index.ntotal == 0:
            return []

        activation_norm = activation / (np.linalg.norm(activation) + 1e-8)
        query_vec = activation_norm.reshape(1, -1).astype(np.float32)

        k = min(top_k, self.index.ntotal)
        similarities, indices = self.index.search(query_vec, k)

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= threshold and idx >= 0:
                entry = self.entries[idx]
                entry.access_count += 1
                results.append((entry, float(sim)))

        return results


class LogitIntegratedModel:
    """
    Frozen LLM with logit-level knowledge injection.

    Instead of modifying hidden states, we hook the LM head and directly
    adjust output logits based on retrieved knowledge entries.
    """

    def __init__(self, model_name: str, device: str = "cuda",
                 boost_scale: float = 10.0, threshold: float = 0.3):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=device
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        self.boost_scale = boost_scale
        self.threshold = threshold

        config = self.model.config
        self.hidden_dim = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_hidden_layers

        self.knowledge_store = LogitKnowledgeStore(self.hidden_dim)

        # Track generation state
        self._gen_step = 0
        self._active_entries = []

        # Install hook on LM head
        self._hook = self.model.lm_head.register_forward_hook(self._logit_hook)

        print(f"Model loaded: {self.num_layers} layers, hidden_dim={self.hidden_dim}, vocab={self.vocab_size}")

    def _logit_hook(self, module, input, output):
        """Hook on lm_head: modify logits based on retrieved knowledge."""
        if self.knowledge_store.total_entries == 0:
            return output

        with torch.no_grad():
            # input[0] is the hidden state going into lm_head
            # Shape: [batch, seq_len, hidden_dim]
            hidden = input[0]
            last_hidden = hidden[0, -1, :].cpu().float().numpy()

            # Query knowledge store
            results = self.knowledge_store.query(
                last_hidden, top_k=20, threshold=self.threshold
            )

            if not results:
                return output

            # Find entries matching current generation step
            logit_bias = torch.zeros(self.vocab_size, device=output.device, dtype=output.dtype)

            for entry, similarity in results:
                # Apply entries for current generation step
                if entry.sequence_pos == self._gen_step:
                    for tid, boost in zip(entry.token_ids, entry.token_boosts):
                        if tid < self.vocab_size:
                            logit_bias[tid] += boost * similarity * entry.strength * self.boost_scale

            if logit_bias.any():
                output = output.clone()
                output[0, -1, :] += logit_bias

        return output

    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        """Generate with logit-level knowledge injection."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        self._gen_step = 0

        # Custom generation loop to track step count
        input_ids = inputs.input_ids
        generated = []

        for step in range(max_new_tokens):
            self._gen_step = step

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                next_logits = outputs.logits[:, -1, :]

                # Greedy
                next_token = next_logits.argmax(dim=-1, keepdim=True)

                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def get_last_hidden(self, text: str) -> np.ndarray:
        """Get the last hidden state before lm_head for a text."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.model(input_ids=inputs.input_ids)
            # outputs.last_hidden_state: [1, seq_len, hidden_dim]
            last = outputs.last_hidden_state[0, -1, :].cpu().float().numpy()
        return last

    def cleanup(self):
        self._hook.remove()


class LearningModuleV3:
    """
    Creates logit-level knowledge entries.

    For each fact to learn:
    1. Get the trigger activation (hidden state at end of question)
    2. Tokenize the answer to get target token IDs
    3. Create entries that boost each answer token at the right step
    """

    def __init__(self, model: LogitIntegratedModel, boost: float = 1.0):
        self.model = model
        self.boost = boost

    def learn(self, question: str, answer: str, source: str = "") -> dict:
        """Learn a fact by creating logit-bias entries for answer tokens."""
        # Get trigger activation
        trigger = self.model.get_last_hidden(question)

        # Tokenize just the answer part (with leading space)
        answer_tokens = self.model.tokenizer.encode(" " + answer, add_special_tokens=False)

        if not answer_tokens:
            return {"status": "skip", "reason": "no answer tokens"}

        # Create entries for each answer token position
        # We boost the correct token AND slightly boost tokens that commonly
        # follow it (bigram context)
        entries_created = 0

        # How many tokens of the answer to store (cap for efficiency)
        max_tokens = min(len(answer_tokens), 20)

        for pos in range(max_tokens):
            token_id = answer_tokens[pos]
            token_text = self.model.tokenizer.decode([token_id])

            entry = LogitKnowledgeEntry(
                trigger=trigger.copy(),
                token_ids=[token_id],
                token_boosts=[self.boost],
                sequence_pos=pos,
                source=source or f"learned: {question[:50]}...",
            )
            self.model.knowledge_store.add(entry)
            entries_created += 1

        answer_preview = self.model.tokenizer.decode(answer_tokens[:max_tokens])
        return {
            "status": "ok",
            "entries_created": entries_created,
            "answer_tokens": max_tokens,
            "answer_preview": answer_preview,
        }

    def learn_batch(self, qa_pairs: list[tuple[str, str]], source: str = "") -> list[dict]:
        results = []
        for i, (q, a) in enumerate(qa_pairs):
            result = self.learn(q, a, source=source or f"batch_{i}")
            print(f"  [{i+1}/{len(qa_pairs)}] {q[:50]}... -> {result['status']}, "
                  f"{result.get('entries_created', 0)} entries, "
                  f"preview: '{result.get('answer_preview', '')[:40]}'")
            results.append(result)
        return results
