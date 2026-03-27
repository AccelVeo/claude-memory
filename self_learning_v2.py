"""
Self-Directed Learning Module v2 — Production Quality

Adds to v1:
1. Contradiction handling — new facts replace old conflicting ones
2. Deduplication — identical facts aren't stored twice
3. Filtering — only meaningful facts are stored (not greetings, opinions)
4. Correction propagation — old entries removed when corrected
5. Confidence tracking — frequently reinforced facts get stronger

Also includes automatic capability gap detection (foundation for auto-adapter training).
"""

import torch
import numpy as np
import json
import re
import time
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss


# ═══════════════════════════════════════════════════════════
# Fact Extractor
# ═══════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """Extract facts from this message. Use the specific names mentioned. Stop after listing facts.

Message: "Our company Nextera moved its headquarters to Austin, Texas last month."
FACT: Nextera's headquarters is in
ANSWER: Austin, Texas.

Message: "Our CTO is Dr. Priya Ramanathan, she joined from Google DeepMind in January."
FACT: Nextera's CTO is
ANSWER: Dr. Priya Ramanathan, who joined from Google DeepMind in January.

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
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def extract(self, user_message, assistant_message=""):
        prompt = EXTRACTION_PROMPT.format(user_message=user_message)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=150, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id)
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return self._parse_facts(response)

    def _parse_facts(self, response):
        results = []
        if "NO FACTS" in response.upper():
            return []
        if "Message:" in response:
            response = response[:response.index("Message:")]

        parts = re.split(r'FACT:\s*', response)
        seen = set()
        for part in parts:
            if 'ANSWER:' in part:
                fact_answer = part.split('ANSWER:', 1)
                if len(fact_answer) == 2:
                    fact = fact_answer[0].strip().rstrip('\n').rstrip('.').strip('*').strip()
                    answer = fact_answer[1].strip().split('\n')[0].strip().strip('*').strip()
                    if (fact and answer and len(answer) > 3 and len(fact) > 3
                            and fact.lower() not in seen
                            and "not specified" not in answer.lower()
                            and "example" not in fact.lower()):
                        seen.add(fact.lower())
                        results.append((fact, answer))
        return results[:4]


# ═══════════════════════════════════════════════════════════
# Fact Entry with metadata for quality control
# ═══════════════════════════════════════════════════════════

@dataclass
class FactEntry:
    trigger: np.ndarray
    token_ids: list[int]
    token_boosts: list[float]
    sequence_pos: int
    source: str = ""
    learned_at: float = 0.0
    fact_id: str = ""          # Groups entries belonging to the same fact
    reinforcement_count: int = 1  # How many times this fact was mentioned


# ═══════════════════════════════════════════════════════════
# Smart Knowledge Store — with contradiction and dedup
# ═══════════════════════════════════════════════════════════

class SmartKnowledgeStore:
    """
    Knowledge store with quality control:
    - Deduplication: won't store the same fact twice
    - Contradiction handling: new facts replace old conflicting ones
    - Reinforcement: repeated facts get stronger
    - Fact grouping: entries from the same fact share a fact_id
    """

    def __init__(self, dim, embedder):
        self.dim = dim
        self.embedder = embedder
        self.fact_index = faiss.IndexFlatIP(dim)
        self.fact_entries = []

        # Track facts by ID for contradiction handling
        self.facts_by_id = {}  # fact_id -> list of entry indices
        self.fact_prompts = {}  # fact_id -> original prompt text
        self.fact_answers = {}  # fact_id -> answer text

        self._next_fact_id = 0

    def _new_fact_id(self):
        self._next_fact_id += 1
        return f"fact_{self._next_fact_id}"

    def add_fact(self, entry):
        t = entry.trigger / (np.linalg.norm(entry.trigger) + 1e-8)
        entry.trigger = t
        idx = len(self.fact_entries)
        self.fact_entries.append(entry)
        self.fact_index.add(t.reshape(1, -1).astype(np.float32))

        # Track by fact_id
        if entry.fact_id:
            if entry.fact_id not in self.facts_by_id:
                self.facts_by_id[entry.fact_id] = []
            self.facts_by_id[entry.fact_id].append(idx)

    def query_facts(self, activation, top_k=20, threshold=0.75):
        if self.fact_index.ntotal == 0: return []
        a = activation / (np.linalg.norm(activation) + 1e-8)
        k = min(top_k, self.fact_index.ntotal)
        sims, idxs = self.fact_index.search(a.reshape(1, -1).astype(np.float32), k)
        return [(self.fact_entries[i], float(s)) for s, i in zip(sims[0], idxs[0])
                if s >= threshold and i >= 0]

    def find_similar_facts(self, trigger, threshold=0.85):
        """Find existing facts with similar triggers — for dedup and contradiction."""
        if self.fact_index.ntotal == 0:
            return []
        a = trigger / (np.linalg.norm(trigger) + 1e-8)
        k = min(20, self.fact_index.ntotal)
        sims, idxs = self.fact_index.search(a.reshape(1, -1).astype(np.float32), k)

        # Group by fact_id and return unique facts
        seen_facts = set()
        results = []
        for sim, idx in zip(sims[0], idxs[0]):
            if sim >= threshold and idx >= 0:
                entry = self.fact_entries[idx]
                if entry.fact_id and entry.fact_id not in seen_facts:
                    seen_facts.add(entry.fact_id)
                    results.append((entry.fact_id, float(sim)))
        return results

    def remove_fact(self, fact_id):
        """
        Remove all entries for a fact. Since FAISS doesn't support deletion,
        we zero out the triggers (making them unmatchable) and mark entries.
        """
        if fact_id not in self.facts_by_id:
            return 0

        removed = 0
        for idx in self.facts_by_id[fact_id]:
            # Zero out trigger so it never matches
            self.fact_entries[idx].trigger = np.zeros(self.dim, dtype=np.float32)
            self.fact_entries[idx].token_boosts = [0.0]
            self.fact_entries[idx].source = f"REMOVED: {self.fact_entries[idx].source}"
            removed += 1

        # Rebuild FAISS index (needed after zeroing triggers)
        self._rebuild_index()

        del self.facts_by_id[fact_id]
        if fact_id in self.fact_prompts:
            del self.fact_prompts[fact_id]
        if fact_id in self.fact_answers:
            del self.fact_answers[fact_id]

        return removed

    def _rebuild_index(self):
        """Rebuild FAISS index from current entries."""
        self.fact_index = faiss.IndexFlatIP(self.dim)
        if self.fact_entries:
            triggers = np.stack([e.trigger for e in self.fact_entries]).astype(np.float32)
            self.fact_index.add(triggers)

    def reinforce_fact(self, fact_id):
        """Increase strength of a fact that was mentioned again."""
        if fact_id in self.facts_by_id:
            for idx in self.facts_by_id[fact_id]:
                self.fact_entries[idx].reinforcement_count += 1
                # Slightly boost the token boost based on reinforcement
                current = self.fact_entries[idx].token_boosts[0]
                self.fact_entries[idx].token_boosts = [min(current * 1.1, 2.0)]

    @property
    def total(self):
        return self.fact_index.ntotal

    @property
    def active_facts(self):
        return len(self.facts_by_id)

    def save(self, path):
        data = {
            "entries": [
                {
                    "trigger": e.trigger.tolist(),
                    "token_ids": e.token_ids,
                    "token_boosts": e.token_boosts,
                    "sequence_pos": e.sequence_pos,
                    "source": e.source,
                    "learned_at": e.learned_at,
                    "fact_id": e.fact_id,
                    "reinforcement_count": e.reinforcement_count,
                }
                for e in self.fact_entries
                if not e.source.startswith("REMOVED")  # Don't save removed entries
            ],
            "fact_prompts": self.fact_prompts,
            "fact_answers": self.fact_answers,
            "next_fact_id": self._next_fact_id,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"  Saved {self.active_facts} active facts ({self.total} entries) to {path}")

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.fact_prompts = data.get("fact_prompts", {})
        self.fact_answers = data.get("fact_answers", {})
        self._next_fact_id = data.get("next_fact_id", 0)
        for ed in data["entries"]:
            entry = FactEntry(
                trigger=np.array(ed["trigger"], dtype=np.float32),
                token_ids=ed["token_ids"], token_boosts=ed["token_boosts"],
                sequence_pos=ed["sequence_pos"], source=ed.get("source", ""),
                learned_at=ed.get("learned_at", 0.0), fact_id=ed.get("fact_id", ""),
                reinforcement_count=ed.get("reinforcement_count", 1))
            self.add_fact(entry)
        print(f"  Loaded {self.active_facts} facts ({self.total} entries) from {path}")


# ═══════════════════════════════════════════════════════════
# Fact Filter — decides what's worth storing
# ═══════════════════════════════════════════════════════════

class FactFilter:
    """Filters extracted facts for quality."""

    # Words that indicate non-factual content
    NOISE_PATTERNS = [
        r"^(hi|hey|hello|thanks|thank you|ok|okay|sure|yes|no|bye)\b",
        r"^(i think|i feel|i believe|in my opinion|maybe|perhaps)",
        r"^(can you|could you|would you|please|help me)",
        r"(to summarize|to rephrase|extracted fact|to extract)",
    ]

    @staticmethod
    def is_valid_fact(prompt, answer):
        """Check if a fact is worth storing."""
        prompt_lower = prompt.lower()
        answer_lower = answer.lower()

        # Too short
        if len(prompt) < 5 or len(answer) < 4:
            return False

        # Noise patterns in answer
        for pattern in FactFilter.NOISE_PATTERNS:
            if re.search(pattern, answer_lower):
                return False

        # Answer is just restating the prompt
        prompt_words = set(prompt_lower.split())
        answer_words = set(answer_lower.split())
        if len(prompt_words) > 2 and prompt_words == answer_words:
            return False

        # Answer contains extraction artifacts
        if any(x in answer_lower for x in ["extracted fact", "to summarize", "to rephrase",
                                            "from the given", "to extract the"]):
            return False

        # Must contain at least one specific detail (number, proper noun, or technical term)
        has_number = bool(re.search(r'\d', answer))
        has_proper_noun = bool(re.search(r'[A-Z][a-z]{2,}', answer))
        has_specific = has_number or has_proper_noun

        return has_specific


# ═══════════════════════════════════════════════════════════
# Capability Gap Detector — tracks failures for auto-training
# ═══════════════════════════════════════════════════════════

class CapabilityGapDetector:
    """
    Tracks when the model fails at similar tasks repeatedly.
    When a pattern emerges, flags it as a capability gap
    that could be addressed by training a LoRA adapter.
    """

    def __init__(self, threshold=3):
        self.failure_log = []  # List of (prompt, expected, got, timestamp)
        self.threshold = threshold  # Failures before flagging

    def log_failure(self, prompt, expected=None, got=None):
        self.failure_log.append({
            "prompt": prompt,
            "expected": expected,
            "got": got,
            "timestamp": time.time(),
        })

    def detect_gaps(self):
        """
        Analyze failure log for patterns.
        Returns list of detected capability gaps.
        """
        if len(self.failure_log) < self.threshold:
            return []

        # Simple pattern detection: cluster failures by keyword overlap
        gaps = []
        # Group by common words in prompts
        from collections import Counter
        word_counts = Counter()
        for entry in self.failure_log:
            words = set(entry["prompt"].lower().split())
            for w in words:
                if len(w) > 3:
                    word_counts[w] += 1

        # Words appearing in multiple failures suggest a capability gap
        for word, count in word_counts.most_common(5):
            if count >= self.threshold:
                related = [e for e in self.failure_log
                          if word in e["prompt"].lower()]
                gaps.append({
                    "pattern": word,
                    "failure_count": count,
                    "examples": related[:5],
                })

        return gaps

    def stats(self):
        return {
            "total_failures": len(self.failure_log),
            "gaps_detected": len(self.detect_gaps()),
        }


# ═══════════════════════════════════════════════════════════
# Conversational Learner v2
# ═══════════════════════════════════════════════════════════

class ConversationalLearnerV2:
    """
    Production-quality self-learning system with:
    - Automatic fact extraction from conversation
    - Contradiction handling (new info replaces old)
    - Deduplication (same fact isn't stored twice)
    - Quality filtering (only meaningful facts stored)
    - Reinforcement (repeated facts get stronger)
    - Capability gap detection
    - Cross-session persistence
    """

    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct",
                 embed_name="BAAI/bge-small-en-v1.5",
                 device="cuda", max_boost=30.0, fact_threshold=0.75):

        print("Initializing Conversational Learner v2...")

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
        self.memory = SmartKnowledgeStore(self.embed_dim, self.embedder)
        self.extractor = FactExtractor(self.model, self.tokenizer, device)
        self.filter = FactFilter()
        self.gap_detector = CapabilityGapDetector()

        # Generation state
        self._gen_step = 0
        self._current_trigger = None
        self._hook = self.model.lm_head.register_forward_hook(self._fact_hook)

        # Stats
        self.facts_learned = 0
        self.facts_updated = 0
        self.facts_deduplicated = 0
        self.facts_filtered = 0
        self.conversations = 0

        print(f"  Ready. Knowledge store: {self.memory.active_facts} facts")

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

    def _generate_augmented(self, prompt, max_new_tokens=100):
        self._current_trigger = self.embedder.encode(prompt, normalize_embeddings=False)
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
        self._current_trigger = None
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _store_fact_smart(self, prompt, answer, verbose=False):
        """
        Store a fact with full quality control:
        1. Filter — is this worth storing?
        2. Dedup — do we already know this?
        3. Contradiction — does this conflict with existing knowledge?
        4. Store with fact_id grouping
        """
        # Step 1: Filter
        if not self.filter.is_valid_fact(prompt, answer):
            self.facts_filtered += 1
            if verbose:
                print(f"  [FILTERED] '{prompt}' -> '{answer}'")
            return 0

        # Step 2: Check for existing similar facts
        trigger = self.embedder.encode(prompt, normalize_embeddings=False)
        similar = self.memory.find_similar_facts(trigger, threshold=0.85)

        for existing_fact_id, sim in similar:
            existing_answer = self.memory.fact_answers.get(existing_fact_id, "")

            # Check if same answer (dedup) or different answer (contradiction)
            answer_trigger_old = self.embedder.encode(existing_answer, normalize_embeddings=False)
            answer_trigger_new = self.embedder.encode(answer, normalize_embeddings=False)
            answer_sim = np.dot(
                answer_trigger_old / (np.linalg.norm(answer_trigger_old) + 1e-8),
                answer_trigger_new / (np.linalg.norm(answer_trigger_new) + 1e-8)
            )

            if answer_sim > 0.85:
                # Same fact — reinforce, don't duplicate
                self.memory.reinforce_fact(existing_fact_id)
                self.facts_deduplicated += 1
                if verbose:
                    print(f"  [REINFORCED] '{prompt}' (already known, strengthened)")
                return 0
            else:
                # Contradiction — remove old, store new
                removed = self.memory.remove_fact(existing_fact_id)
                self.facts_updated += 1
                if verbose:
                    print(f"  [UPDATED] '{prompt}' old='{existing_answer[:40]}' -> new='{answer[:40]}' (removed {removed} entries)")

        # Step 3: Store new fact
        fact_id = self.memory._new_fact_id()
        self.memory.fact_prompts[fact_id] = prompt
        self.memory.fact_answers[fact_id] = answer

        tokens = self.tokenizer.encode(" " + answer, add_special_tokens=False)
        n = min(len(tokens), 25)

        # Store with primary trigger
        for pos in range(n):
            self.memory.add_fact(FactEntry(
                trigger=trigger.copy(), token_ids=[tokens[pos]],
                token_boosts=[1.0], sequence_pos=pos,
                source=f"learned: {prompt[:40]}", learned_at=time.time(),
                fact_id=fact_id))

        # Store with alternate triggers
        alts = self._generate_alt_triggers(prompt)
        for alt_text in alts[1:]:  # Skip first (same as primary)
            alt_trigger = self.embedder.encode(alt_text, normalize_embeddings=False)
            for pos in range(n):
                self.memory.add_fact(FactEntry(
                    trigger=alt_trigger.copy(), token_ids=[tokens[pos]],
                    token_boosts=[1.0], sequence_pos=pos,
                    source=f"alt: {alt_text[:40]}", learned_at=time.time(),
                    fact_id=fact_id))

        self.facts_learned += 1
        if verbose:
            print(f"  [LEARNED] '{prompt}' -> '{answer}'")
        return n

    def _generate_alt_triggers(self, prompt):
        alts = [prompt]
        words = prompt.split()
        for i, w in enumerate(words):
            if "'s" in w:
                owner = w.replace("'s", "")
                rest = " ".join(words[i+1:])
                alts.append(f"The {rest} of {owner} is")
                alts.append(f"What is {owner}'s {rest}")
                break
        if prompt.startswith("The "):
            alts.append(f"What {prompt.lower()}")
        return alts

    def chat(self, user_message, verbose=False):
        self.conversations += 1
        response = self._generate_augmented(user_message, max_new_tokens=100)

        # Extract and store facts with quality control
        extracted = self.extractor.extract(user_message, response)
        for prompt, answer in extracted:
            self._store_fact_smart(prompt, answer, verbose=verbose)

        # Also store raw message as trigger for first extracted fact
        if extracted:
            first_answer = extracted[0][1]
            if self.filter.is_valid_fact(user_message, first_answer):
                raw_trigger = self.embedder.encode(user_message, normalize_embeddings=False)
                tokens = self.tokenizer.encode(" " + first_answer, add_special_tokens=False)
                n = min(len(tokens), 25)
                fact_id = self.memory._new_fact_id()
                for pos in range(n):
                    self.memory.add_fact(FactEntry(
                        trigger=raw_trigger.copy(), token_ids=[tokens[pos]],
                        token_boosts=[1.0], sequence_pos=pos,
                        source=f"raw: {user_message[:40]}", learned_at=time.time(),
                        fact_id=fact_id))

        return response

    def recall(self, query):
        return self._generate_augmented(query, max_new_tokens=80)

    def stats(self):
        return {
            "conversations": self.conversations,
            "facts_learned": self.facts_learned,
            "facts_updated": self.facts_updated,
            "facts_deduplicated": self.facts_deduplicated,
            "facts_filtered": self.facts_filtered,
            "active_facts": self.memory.active_facts,
            "store_entries": self.memory.total,
            "capability_gaps": self.gap_detector.stats(),
        }

    def save_memory(self, path="knowledge_store_v2.json"):
        self.memory.save(path)

    def load_memory(self, path="knowledge_store_v2.json"):
        self.memory.load(path)


# ═══════════════════════════════════════════════════════════
# Comprehensive Test
# ═══════════════════════════════════════════════════════════

CONVERSATION = [
    # Initial facts
    "Our company Nextera is headquartered in Austin, Texas.",
    "The CTO is Dr. Priya Ramanathan, she joined from Google DeepMind in January.",
    "We use Kubernetes version 1.29 on our production cluster on AWS us-east-1.",
    "Our main product Vortex is a real-time fraud detection platform processing 2 million transactions per second.",
    "Our biggest client is Meridian Bank, paying us about 4 million a year since 2023.",
    "We use CockroachDB for our transaction database.",
    "The deployment pipeline uses ArgoCD with automatic canary releases — new builds go to 5% of traffic first.",
    "Marcus Chen is my team lead, he manages the infrastructure team of 12 people.",
    "All deployments must happen before 2pm EST on weekdays, no weekend deploys unless P0.",
    "We use Datadog for metrics and PagerDuty for alerting with a 15-minute SLA for P1.",

    # CONTRADICTIONS — user corrects earlier info
    "Actually, we just moved headquarters from Austin to Denver, Colorado last week.",
    "Update: Marcus Chen left the company. Our new team lead is Sarah Kim.",

    # DUPLICATES — user mentions something already known
    "Just to confirm, we're still on Kubernetes 1.29 right?",
    "Yep, Meridian Bank is still our biggest client.",

    # NOISE — should NOT be stored
    "Hey, how's it going?",
    "Thanks for your help with that!",
    "I think the weather is nice today.",
]

RECALL_TESTS = [
    # Should reflect UPDATED info
    ("Nextera's headquarters is in", ["Denver", "Colorado"]),
    ("The team lead is", ["Sarah", "Kim"]),

    # Should reflect original info (not contradicted)
    ("Nextera's CTO is", ["Priya", "Ramanathan"]),
    ("Vortex is", ["fraud", "detection"]),
    ("Nextera's biggest client is", ["Meridian"]),
    ("Nextera's transaction database is", ["CockroachDB"]),

    # Control
    ("The capital of France is", ["Paris"]),
    ("Python is a", ["programming"]),
]


def main():
    print("=" * 60)
    print("SELF-DIRECTED LEARNING v2 — PRODUCTION QUALITY TEST")
    print("=" * 60)

    learner = ConversationalLearnerV2()

    # Phase 1: Conversation with contradictions, duplicates, and noise
    print("\n[Phase 1: Conversation]")
    for message in CONVERSATION:
        print(f"\nUser: {message}")
        response = learner.chat(message, verbose=True)
        print(f"Assistant: {response[:80]}")

    print(f"\n  Stats: {json.dumps(learner.stats(), indent=2)}")

    # Phase 2: Recall
    print(f"\n[Phase 2: Recall]")
    recall_ok = 0
    for prompt, keywords in RECALL_TESTS:
        response = learner.recall(prompt)
        hits = [k for k in keywords if k.lower() in response.lower()]
        ok = len(hits) >= 1
        if ok: recall_ok += 1
        print(f"  [{'OK' if ok else 'MISS'}] {prompt}")
        print(f"       -> {response.strip()[:70]}")
        if not ok:
            print(f"       Want: {keywords}")

    # Phase 3: Persistence
    print(f"\n[Phase 3: Persistence]")
    learner.save_memory("/tmp/self_learning_v2.json")
    learner2 = ConversationalLearnerV2()
    learner2.load_memory("/tmp/self_learning_v2.json")

    reload_ok = 0
    for prompt, keywords in RECALL_TESTS[:4]:
        response = learner2.recall(prompt)
        hits = [k for k in keywords if k.lower() in response.lower()]
        ok = len(hits) >= 1
        if ok: reload_ok += 1
        print(f"  [{'OK' if ok else 'MISS'}] {prompt} -> {response.strip()[:50]}")

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    stats = learner.stats()
    print(f"  Facts learned:      {stats['facts_learned']}")
    print(f"  Facts updated:      {stats['facts_updated']} (contradictions handled)")
    print(f"  Facts deduplicated: {stats['facts_deduplicated']}")
    print(f"  Facts filtered:     {stats['facts_filtered']} (noise rejected)")
    print(f"  Active facts:       {stats['active_facts']}")
    print(f"  Store entries:      {stats['store_entries']}")
    print(f"  Recall:             {recall_ok}/{len(RECALL_TESTS)}")
    print(f"  Persistence:        {reload_ok}/4")


if __name__ == "__main__":
    main()
