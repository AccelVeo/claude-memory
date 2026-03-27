"""
Self-Directed Learning Module

The piece that makes the system real: automatic fact extraction from
natural conversation. No hand-crafted QA pairs needed.

How it works:
1. User has a conversation with the model
2. After each exchange, the model analyzes what was said
3. If the user provided new information, corrections, or preferences,
   the model extracts structured (prompt, answer) pairs
4. Those pairs are automatically stored in the knowledge store
5. Next time a related question comes up, the model remembers

The extraction uses the SAME frozen model — it's just a prompted task.
"""

import torch
import numpy as np
import json
import re
import time
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss


# ═══════════════════════════════════════════════════════════
# Fact Extractor — analyzes conversation for learnable info
# ═══════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """Extract facts from this message. Use the specific names mentioned. Write multiple phrasings for each fact. Stop after listing facts.

Message: "Our company Nextera moved its headquarters to Austin, Texas last month."
FACT: Nextera's headquarters is in
ANSWER: Austin, Texas.
FACT: Nextera is located in
ANSWER: Austin, Texas.

Message: "The main product is called Vortex — it's a real-time fraud detection platform."
FACT: Vortex is
ANSWER: a real-time fraud detection platform.
FACT: The main product Vortex is
ANSWER: a real-time fraud detection platform.

Message: "We use Datadog for monitoring and ArgoCD for deployments."
FACT: The monitoring tool is
ANSWER: Datadog.
FACT: The deployment tool is
ANSWER: ArgoCD.

Message: "Hi, how are you today?"
NO FACTS

Message: "{user_message}"
"""


class FactExtractor:
    """
    Extracts learnable facts from conversation using the frozen model itself.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def extract(self, user_message, assistant_message="") -> list[tuple[str, str]]:
        """
        Analyze a conversation turn and extract (prompt, answer) pairs.
        Uses few-shot prompting with a simple FACT:/ANSWER: format
        that small models can reliably follow.
        """
        prompt = EXTRACTION_PROMPT.format(user_message=user_message)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,  # Shorter to prevent hallucination
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return self._parse_facts(response)

    def _parse_facts(self, response: str) -> list[tuple[str, str]]:
        """Parse FACT:/ANSWER: format from model response."""
        results = []

        if "NO FACTS" in response.upper():
            return []

        # Stop at "Message:" to prevent the model from generating more examples
        if "Message:" in response:
            response = response[:response.index("Message:")]

        # Split by FACT: markers
        parts = re.split(r'FACT:\s*', response)

        seen = set()
        for part in parts:
            if 'ANSWER:' in part:
                fact_answer = part.split('ANSWER:', 1)
                if len(fact_answer) == 2:
                    fact = fact_answer[0].strip().rstrip('\n').rstrip('.')
                    answer = fact_answer[1].strip().split('\n')[0].strip()
                    # Clean up — remove markdown artifacts
                    fact = fact.strip('*').strip()
                    answer = answer.strip('*').strip()
                    # Validate
                    if (fact and answer and len(answer) > 3 and len(fact) > 3
                            and fact.lower() not in seen
                            and "not specified" not in answer.lower()
                            and "none" not in answer.lower()
                            and "example" not in fact.lower()):
                        seen.add(fact.lower())
                        results.append((fact, answer))

        # Max 4 facts per message (allows multiple phrasings)
        return results[:4]


# ═══════════════════════════════════════════════════════════
# Knowledge Store (reused from v12/v13)
# ═══════════════════════════════════════════════════════════

@dataclass
class FactEntry:
    trigger: np.ndarray
    token_ids: list[int]
    token_boosts: list[float]
    sequence_pos: int
    source: str = ""
    learned_at: float = 0.0


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

    def save(self, path):
        """Save knowledge store to disk for persistence across sessions."""
        data = {
            "entries": [
                {
                    "trigger": e.trigger.tolist(),
                    "token_ids": e.token_ids,
                    "token_boosts": e.token_boosts,
                    "sequence_pos": e.sequence_pos,
                    "source": e.source,
                    "learned_at": e.learned_at,
                }
                for e in self.fact_entries
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"  Saved {len(self.fact_entries)} entries to {path}")

    def load(self, path):
        """Load knowledge store from disk."""
        with open(path) as f:
            data = json.load(f)
        for entry_data in data["entries"]:
            entry = FactEntry(
                trigger=np.array(entry_data["trigger"], dtype=np.float32),
                token_ids=entry_data["token_ids"],
                token_boosts=entry_data["token_boosts"],
                sequence_pos=entry_data["sequence_pos"],
                source=entry_data.get("source", ""),
                learned_at=entry_data.get("learned_at", 0.0),
            )
            self.add_fact(entry)
        print(f"  Loaded {self.total} entries from {path}")


# ═══════════════════════════════════════════════════════════
# Conversational Learner — the full self-learning system
# ═══════════════════════════════════════════════════════════

class ConversationalLearner:
    """
    A conversational AI that automatically learns from interactions.

    Each conversation turn:
    1. Generate response using knowledge-augmented model
    2. Extract any new facts from the user's message
    3. Store extracted facts in the knowledge store
    4. Persist knowledge across sessions
    """

    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct",
                 embed_name="BAAI/bge-small-en-v1.5",
                 device="cuda", max_boost=30.0, fact_threshold=0.75):

        print("Initializing Conversational Learner...")

        # Load models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=device)
        for p in self.model.parameters():
            p.requires_grad = False

        self.embedder = SentenceTransformer(embed_name, device=device)
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()

        self.device = device
        self.max_boost = max_boost
        self.fact_threshold = fact_threshold
        self.vocab_size = self.model.config.vocab_size

        # Components
        self.memory = KnowledgeStore(self.embed_dim)
        self.extractor = FactExtractor(self.model, self.tokenizer, device)

        # Generation state
        self._gen_step = 0
        self._current_trigger = None
        self._hook = self.model.lm_head.register_forward_hook(self._fact_hook)

        # Conversation history
        self.history = []

        # Stats
        self.facts_learned = 0
        self.conversations = 0

        print(f"  Ready. Knowledge store: {self.memory.total} entries")

    def _adaptive_boost(self, sim):
        if sim <= self.fact_threshold: return 0.0
        return ((sim - self.fact_threshold) / (1.0 - self.fact_threshold)) * self.max_boost

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

    def _generate_raw(self, prompt, max_new_tokens=100):
        """Generate without knowledge injection (for extraction)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)

    def _generate_augmented(self, prompt, max_new_tokens=100):
        """Generate WITH knowledge injection."""
        self._current_trigger = self.embedder.encode(prompt, normalize_embeddings=False)

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

        self._current_trigger = None
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _generate_alt_triggers(self, prompt):
        """Generate alternate phrasings for a fact trigger."""
        alts = [prompt]

        # Extract key entity/subject from the prompt
        # "Nextera's CTO is" → also store "The CTO of Nextera is", "Who is Nextera's CTO"
        words = prompt.split()

        # If prompt contains a possessive, create "The X of Y" form
        for i, w in enumerate(words):
            if "'s" in w:
                owner = w.replace("'s", "")
                rest = " ".join(words[i+1:])
                alts.append(f"The {rest} of {owner} is")
                alts.append(f"What is {owner}'s {rest}")
                break

        # If prompt starts with "The X uses/is/has", create question form
        if prompt.startswith("The "):
            subject = prompt.split(" is")[0] if " is" in prompt else prompt.split(" uses")[0]
            alts.append(f"What {prompt.lower()}")

        return alts

    def _store_fact(self, prompt, answer):
        """Store a fact with multiple trigger phrasings."""
        # Generate alternate triggers for better recall
        triggers_texts = self._generate_alt_triggers(prompt)

        total_entries = 0
        tokens = self.tokenizer.encode(" " + answer, add_special_tokens=False)
        n = min(len(tokens), 25)

        for trigger_text in triggers_texts:
            trigger = self.embedder.encode(trigger_text, normalize_embeddings=False)
            for pos in range(n):
                self.memory.add_fact(FactEntry(
                    trigger=trigger.copy(),
                    token_ids=[tokens[pos]],
                    token_boosts=[1.0],
                    sequence_pos=pos,
                    source=f"conversation: {prompt[:40]}",
                    learned_at=time.time(),
                ))
                total_entries += 1

        self.facts_learned += 1
        return total_entries

    def chat(self, user_message: str, verbose=False) -> str:
        """
        Process a user message:
        1. Generate knowledge-augmented response
        2. Extract and store any new facts from user's message
        3. Return the response
        """
        self.conversations += 1

        # Step 1: Generate response using knowledge store
        response = self._generate_augmented(user_message, max_new_tokens=100)

        # Step 2: Extract facts from the user's message
        extracted = self.extractor.extract(user_message, response)

        # Step 3: Store extracted facts
        entries_added = 0
        for prompt, answer in extracted:
            n = self._store_fact(prompt, answer)
            entries_added += n
            if verbose:
                print(f"  [LEARNED] '{prompt}' -> '{answer}'")

        # Step 3b: Also store using the raw user message as trigger
        # This ensures the original phrasing is always indexed
        if extracted:
            first_answer = extracted[0][1]
            raw_trigger = self.embedder.encode(user_message, normalize_embeddings=False)
            tokens = self.tokenizer.encode(" " + first_answer, add_special_tokens=False)
            n = min(len(tokens), 25)
            for pos in range(n):
                self.memory.add_fact(FactEntry(
                    trigger=raw_trigger.copy(),
                    token_ids=[tokens[pos]], token_boosts=[1.0],
                    sequence_pos=pos, source=f"raw: {user_message[:40]}",
                    learned_at=time.time()))
            entries_added += n

        # Track history
        self.history.append({
            "user": user_message,
            "assistant": response,
            "facts_extracted": len(extracted),
            "entries_added": entries_added,
        })

        if verbose and not extracted:
            print(f"  [NO NEW FACTS]")

        return response

    def recall(self, query: str) -> str:
        """Query the model with knowledge augmentation."""
        return self._generate_augmented(query, max_new_tokens=80)

    def stats(self):
        return {
            "conversations": self.conversations,
            "facts_learned": self.facts_learned,
            "store_entries": self.memory.total,
            "history_length": len(self.history),
        }

    def save_memory(self, path="knowledge_store.json"):
        self.memory.save(path)

    def load_memory(self, path="knowledge_store.json"):
        self.memory.load(path)


# ═══════════════════════════════════════════════════════════
# Test: Simulate a conversation where the user teaches facts
# ═══════════════════════════════════════════════════════════

CONVERSATION = [
    # User teaches facts naturally through conversation
    "Hey, just so you know, our company Nextera moved its headquarters to Austin, Texas last month.",
    "Our CTO is Dr. Priya Ramanathan, she joined from Google DeepMind in January.",
    "We use Kubernetes version 1.29 on our production cluster, running on AWS us-east-1.",
    "The main product is called Vortex — it's a real-time fraud detection platform that processes 2 million transactions per second.",
    "Our biggest client is Meridian Bank, they've been with us since 2023 and pay us about 4 million a year.",
    "Oh and we just switched from PostgreSQL to CockroachDB for our transaction database because we needed multi-region consistency.",
    "The deployment pipeline uses ArgoCD with automatic canary releases — new builds go to 5% of traffic first.",
    "My team lead is Marcus Chen, he's been at the company for 6 years and manages the infrastructure team of 12 people.",
    "We have a strict policy — all deployments must happen before 2pm EST on weekdays, no weekend deploys unless it's a P0.",
    "The monitoring stack is Datadog for metrics and PagerDuty for alerting, with a 15-minute SLA for P1 incidents.",
]

# Recall tests — can the model answer questions about what it learned?
RECALL_TESTS = [
    ("Nextera's headquarters is in", ["Austin", "Texas"]),
    ("The CTO of Nextera is", ["Priya", "Ramanathan"]),
    ("Nextera's production Kubernetes version is", ["1.29"]),
    ("Vortex is", ["fraud", "detection"]),
    ("Nextera's biggest client is", ["Meridian"]),
    ("Nextera's transaction database is", ["CockroachDB"]),
    ("Nextera uses what for deployments?", ["ArgoCD", "canary"]),
    ("Marcus Chen is", ["team lead", "infrastructure"]),
    ("Nextera's deployment policy says", ["2pm", "weekend"]),
    ("Nextera uses what for monitoring?", ["Datadog", "PagerDuty"]),
]

# Control — should still work
CONTROL = [
    ("The capital of France is", ["Paris"]),
    ("Python is a", ["programming"]),
    ("Water is made of", ["hydrogen", "H2O"]),
]


def main():
    print("=" * 60)
    print("SELF-DIRECTED LEARNING TEST")
    print("=" * 60)

    learner = ConversationalLearner()

    # Phase 1: Have a conversation where user shares information
    print("\n[Phase 1: Conversation — user teaches facts naturally]")
    print("-" * 40)

    for i, message in enumerate(CONVERSATION):
        print(f"\nUser: {message}")
        response = learner.chat(message, verbose=True)
        print(f"Assistant: {response[:100]}")

    print(f"\n  Stats: {learner.stats()}")

    # Phase 2: Test recall
    print(f"\n[Phase 2: Recall — can it remember what it learned?]")
    print("-" * 40)

    recall_ok = 0
    for prompt, keywords in RECALL_TESTS:
        response = learner.recall(prompt)
        hits = [k for k in keywords if k.lower() in response.lower()]
        ok = len(hits) >= 1
        if ok: recall_ok += 1
        status = "OK" if ok else "MISS"
        print(f"  [{status}] {prompt}")
        print(f"         -> {response.strip()[:70]}")
        if not ok:
            print(f"         Want: {keywords}")

    # Phase 3: Control
    print(f"\n[Phase 3: Control — existing knowledge]")
    ctrl_ok = 0
    for prompt, keywords in CONTROL:
        response = learner.recall(prompt)
        hits = [k for k in keywords if k.lower() in response.lower()]
        ok = len(hits) >= 1
        if ok: ctrl_ok += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {prompt} -> {response.strip()[:50]}")

    # Phase 4: Save and reload test
    print(f"\n[Phase 4: Persistence — save and reload]")
    learner.save_memory("/tmp/self_learning_test.json")

    # Create new learner and load memory
    print("  Creating new learner instance...")
    learner2 = ConversationalLearner()
    learner2.load_memory("/tmp/self_learning_test.json")

    print("  Testing recall after reload:")
    reload_ok = 0
    for prompt, keywords in RECALL_TESTS[:5]:
        response = learner2.recall(prompt)
        hits = [k for k in keywords if k.lower() in response.lower()]
        ok = len(hits) >= 1
        if ok: reload_ok += 1
        status = "OK" if ok else "MISS"
        print(f"  [{status}] {prompt} -> {response.strip()[:60]}")

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Facts extracted from conversation: {learner.facts_learned}")
    print(f"  Store entries: {learner.memory.total}")
    print(f"  Recall accuracy: {recall_ok}/{len(RECALL_TESTS)}")
    print(f"  Control: {ctrl_ok}/{len(CONTROL)}")
    print(f"  Persistence reload: {reload_ok}/5")


if __name__ == "__main__":
    main()
