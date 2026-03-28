"""
Persistent Model Server

Loads the 72B model once and keeps it in memory. Accepts commands
via a simple JSON socket protocol. Supports:
- learn: add facts to knowledge store
- generate: generate with knowledge injection
- extract: extract facts from conversation
- stats: get store statistics
- learn_batch: batch learn from a dataset

This avoids reloading the 150GB model for every experiment iteration.
"""

import torch
import numpy as np
import json
import faiss
import time
import re
import socket
import threading
import sys
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer


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
        self.fact_index = faiss.IndexFlatIP(dim)
        self.fact_entries = []

    def add_fact(self, entry):
        t = entry.trigger / (np.linalg.norm(entry.trigger) + 1e-8)
        entry.trigger = t
        self.fact_entries.append(entry)
        self.fact_index.add(t.reshape(1, -1).astype(np.float32))

    def query_facts(self, activation, top_k=20, threshold=0.75):
        if self.fact_index.ntotal == 0: return []
        a = activation / (np.linalg.norm(activation) + 1e-8)
        k = min(top_k, self.fact_index.ntotal)
        sims, idxs = self.fact_index.search(a.reshape(1, -1).astype(np.float32), k)
        return [(self.fact_entries[i], float(s)) for s, i in zip(sims[0], idxs[0])
                if s >= threshold and i >= 0]

    @property
    def total(self):
        return self.fact_index.ntotal

    def clear(self):
        self.fact_index = faiss.IndexFlatIP(self.dim)
        self.fact_entries = []


class ModelServer:
    def __init__(self, model_name="Qwen/Qwen2.5-72B-Instruct",
                 embed_name="BAAI/bge-small-en-v1.5",
                 max_boost=80.0, fact_threshold=0.75):
        print(f"[SERVER] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map="auto")
        for p in self.model.parameters():
            p.requires_grad = False
        print(f"[SERVER] Model loaded across GPUs")

        self.embedder = SentenceTransformer(embed_name, device="cuda:0")
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()
        self.max_boost = max_boost
        self.fact_threshold = fact_threshold
        self.vocab_size = self.model.config.vocab_size
        self.memory = KnowledgeStore(self.embed_dim)
        self._gen_step = 0
        self._current_trigger = None

        # Find and hook lm_head
        lm_head = self._find_lm_head(self.model)
        self._hook = lm_head.register_forward_hook(self._fact_hook)
        print(f"[SERVER] Ready: vocab={self.vocab_size}, embed={self.embed_dim}")

    def _find_lm_head(self, model):
        if hasattr(model, 'lm_head'):
            return model.lm_head
        if hasattr(model, 'base_model'):
            return self._find_lm_head(model.base_model)
        raise AttributeError("Cannot find lm_head")

    def _adaptive_boost(self, sim):
        if sim <= self.fact_threshold: return 0.0
        confidence = (sim - self.fact_threshold) / (1.0 - self.fact_threshold)
        return (confidence ** 1.5) * self.max_boost

    def _fact_hook(self, module, input, output):
        if self.memory.total == 0 or self._current_trigger is None:
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

    def learn(self, prompt, answer):
        trigger = self.embedder.encode(prompt, normalize_embeddings=False)
        tokens = self.tokenizer.encode(" " + answer, add_special_tokens=False)
        n = min(len(tokens), 20)
        for pos in range(n):
            self.memory.add_fact(FactEntry(
                trigger=trigger.copy(), token_ids=[tokens[pos]],
                token_boosts=[1.0], sequence_pos=pos, source=prompt[:30]))
        return n

    def learn_batch(self, facts):
        prompts = [p for p, a in facts]
        triggers = self.embedder.encode(prompts, normalize_embeddings=False, batch_size=256)
        total = 0
        for i, (prompt, answer) in enumerate(facts):
            tokens = self.tokenizer.encode(" " + answer, add_special_tokens=False)
            n = min(len(tokens), 20)
            for pos in range(n):
                self.memory.add_fact(FactEntry(
                    trigger=triggers[i].copy(), token_ids=[tokens[pos]],
                    token_boosts=[1.0], sequence_pos=pos, source=prompt[:30]))
                total += 1
            if (i+1) % 10000 == 0:
                print(f"[SERVER] Learned {i+1}/{len(facts)} ({total} entries)")
        return total

    def generate(self, prompt, max_new_tokens=30):
        self._current_trigger = self.embedder.encode(prompt, normalize_embeddings=False)
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

    def clear_memory(self):
        self.memory.clear()
        return "Memory cleared"

    def handle_command(self, cmd):
        try:
            action = cmd.get("action")

            if action == "learn":
                n = self.learn(cmd["prompt"], cmd["answer"])
                return {"status": "ok", "entries": n}

            elif action == "learn_batch":
                facts = cmd["facts"]  # list of [prompt, answer]
                n = self.learn_batch(facts)
                return {"status": "ok", "entries": n, "facts": len(facts)}

            elif action == "generate":
                text = self.generate(cmd["prompt"], cmd.get("max_tokens", 30))
                return {"status": "ok", "response": text}

            elif action == "stats":
                return {"status": "ok", "entries": self.memory.total,
                        "boost": self.max_boost, "threshold": self.fact_threshold}

            elif action == "clear":
                self.clear_memory()
                return {"status": "ok", "message": "Memory cleared"}

            elif action == "set_boost":
                self.max_boost = cmd["value"]
                return {"status": "ok", "max_boost": self.max_boost}

            elif action == "set_threshold":
                self.fact_threshold = cmd["value"]
                return {"status": "ok", "threshold": self.fact_threshold}

            elif action == "ping":
                return {"status": "ok", "message": "pong", "entries": self.memory.total}

            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def serve(self, port=9999):
        """Start TCP server that accepts JSON commands."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', port))
        server.listen(1)
        print(f"[SERVER] Listening on port {port}")

        while True:
            conn, addr = server.accept()
            print(f"[SERVER] Connection from {addr}")
            try:
                data = b""
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                    # Try to parse — commands end with newline
                    if b"\n" in data:
                        lines = data.split(b"\n")
                        for line in lines[:-1]:
                            if line.strip():
                                cmd = json.loads(line.decode())
                                result = self.handle_command(cmd)
                                response = json.dumps(result) + "\n"
                                conn.sendall(response.encode())
                        data = lines[-1]
            except Exception as e:
                print(f"[SERVER] Error: {e}")
            finally:
                conn.close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
    server = ModelServer()
    server.serve(port)
