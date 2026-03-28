"""
Cognitive Extensions — Making the System Self-Aware

Five modules that transform the system from a tool into something
that understands itself, its knowledge, and the people it works with.

1. Metacognition — knows what it knows and doesn't know
2. Emotional Weighting — prioritizes memories by significance
3. Sleep/Consolidation — maintains and organizes memory offline
4. Curiosity — identifies gaps and seeks information
5. Social Modeling — tracks relationships and context per person
"""

import numpy as np
import re
import time
import json
from dataclasses import dataclass, field
from collections import defaultdict


# ═══════════════════════════════════════════════════════════
# 1. METACOGNITION — Know what you know
# ═══════════════════════════════════════════════════════════

class Metacognition:
    """
    Monitors the system's own confidence. Before answering, assesses:
    - How well does this query match stored knowledge?
    - Are there conflicting stored answers?
    - Should I answer confidently, hedge, or admit ignorance?

    This isn't simulated confidence — it's measured from real signals
    in the knowledge store and generation pipeline.
    """

    # Confidence levels
    HIGH = "high"           # Strong match, answer directly
    MEDIUM = "medium"       # Partial match, hedge
    LOW = "low"             # No match, use base model, flag uncertainty
    CONFLICTING = "conflict" # Multiple contradictory matches

    def __init__(self, high_threshold=0.85, medium_threshold=0.70):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.history = []  # Track confidence-outcome pairs for calibration

    def assess(self, query_results):
        """
        Assess confidence based on knowledge store query results.

        query_results: list of (entry, similarity) tuples from FAISS
        Returns: (confidence_level, confidence_score, explanation)
        """
        if not query_results:
            return self.LOW, 0.0, "No matching knowledge found"

        # Get top similarities
        sims = [sim for _, sim in query_results]
        top_sim = max(sims)
        avg_sim = sum(sims) / len(sims)

        # Check for conflicting answers (different sources with similar triggers)
        sources = set()
        for entry, sim in query_results:
            if sim > self.medium_threshold:
                sources.add(entry.source[:30])

        # Conflict: multiple distinct facts matching with high similarity
        # but from different source prompts
        if len(sources) > 3 and top_sim > self.medium_threshold:
            return self.CONFLICTING, top_sim, f"Multiple conflicting sources ({len(sources)} distinct)"

        # High confidence
        if top_sim >= self.high_threshold:
            return self.HIGH, top_sim, f"Strong match (sim={top_sim:.3f})"

        # Medium confidence
        if top_sim >= self.medium_threshold:
            return self.MEDIUM, top_sim, f"Partial match (sim={top_sim:.3f})"

        # Low confidence
        return self.LOW, top_sim, f"Weak match (sim={top_sim:.3f})"

    def get_response_prefix(self, confidence_level):
        """
        Get an appropriate prefix for the response based on confidence.
        This is injected into the generation context.
        """
        if confidence_level == self.HIGH:
            return ""  # No prefix needed — answer directly
        elif confidence_level == self.MEDIUM:
            return "Based on what I've learned, "
        elif confidence_level == self.LOW:
            return ""  # Let base model answer naturally
        elif confidence_level == self.CONFLICTING:
            return "I have conflicting information about this. "
        return ""

    def should_ask_user(self, confidence_level):
        """Should the system ask the user for clarification?"""
        return confidence_level in (self.LOW, self.CONFLICTING)

    def log_outcome(self, confidence_level, confidence_score, was_correct):
        """Track outcomes for calibration."""
        self.history.append({
            "level": confidence_level,
            "score": confidence_score,
            "correct": was_correct,
            "timestamp": time.time(),
        })

    def calibration_report(self):
        """How well-calibrated is the confidence?"""
        if len(self.history) < 10:
            return "Insufficient data for calibration"

        by_level = defaultdict(list)
        for h in self.history:
            by_level[h["level"]].append(h["correct"])

        report = {}
        for level, outcomes in by_level.items():
            accuracy = sum(outcomes) / len(outcomes)
            report[level] = {"accuracy": accuracy, "count": len(outcomes)}
        return report


# ═══════════════════════════════════════════════════════════
# 2. EMOTIONAL WEIGHTING — Prioritize by significance
# ═══════════════════════════════════════════════════════════

class EmotionalWeighting:
    """
    Scores the significance of new information to determine how
    strongly it should be stored. Not simulating emotions — measuring
    importance signals in language.

    High significance: corrections, critical info, repeated emphasis
    Low significance: casual mentions, obvious facts, filler
    """

    # Correction patterns — highest weight
    CORRECTION_PATTERNS = [
        r'\bactually\b', r'\bno[,.]', r'\bthat\'s wrong\b',
        r'\bincorrect\b', r'\bwe changed\b', r'\bwe switched\b',
        r'\bwe moved\b', r'\bnot anymore\b', r'\bupdated?\b',
        r'\bcorrection\b', r'\bplease note\b',
    ]

    # Emphasis patterns — high weight
    EMPHASIS_PATTERNS = [
        r'\bcritical\b', r'\bimportant\b', r'\bmust\b', r'\bnever\b',
        r'\balways\b', r'\bremember this\b', r'\bdon\'t forget\b',
        r'\bkey point\b', r'\bessential\b', r'\brequired\b',
        r'[A-Z]{3,}',  # ALL CAPS words
        r'!{2,}',  # Multiple exclamation marks
    ]

    # High-consequence domains — elevated weight
    CONSEQUENCE_PATTERNS = [
        r'\bsecurity\b', r'\bpassword\b', r'\bcredential\b',
        r'\bproduction\b', r'\bdeployment\b', r'\bclient\b',
        r'\bdeadline\b', r'\bSLA\b', r'\bincident\b',
        r'\bcompliance\b', r'\baudit\b', r'\blegal\b',
    ]

    # Casual/low-value patterns — reduced weight
    CASUAL_PATTERNS = [
        r'^(hey|hi|hello|thanks|ok|sure|yeah)\b',
        r'\bi think\b', r'\bmaybe\b', r'\bprobably\b',
        r'\bi guess\b', r'\bkind of\b', r'\bsort of\b',
    ]

    def score(self, message, is_correction=False):
        """
        Score the significance of a message.
        Returns a multiplier from 0.5 (low) to 3.0 (critical).
        """
        text = message.lower()
        score = 1.0

        # Corrections get highest weight
        if is_correction:
            score = 2.5
        else:
            for pattern in self.CORRECTION_PATTERNS:
                if re.search(pattern, text):
                    score = max(score, 2.0)
                    break

        # Emphasis boosts
        emphasis_count = sum(1 for p in self.EMPHASIS_PATTERNS if re.search(p, message))
        if emphasis_count > 0:
            score = max(score, 1.5 + emphasis_count * 0.2)

        # Consequence domains
        consequence_count = sum(1 for p in self.CONSEQUENCE_PATTERNS if re.search(p, text))
        if consequence_count > 0:
            score = max(score, 1.3 + consequence_count * 0.1)

        # Casual reduces weight
        for pattern in self.CASUAL_PATTERNS:
            if re.search(pattern, text):
                score = min(score, 0.7)
                break

        # Cap at 3.0
        return min(score, 3.0)

    def apply_to_boost(self, base_boost, significance_score):
        """Apply significance to a token boost value."""
        return base_boost * significance_score


# ═══════════════════════════════════════════════════════════
# 3. SLEEP/CONSOLIDATION — Maintain memory offline
# ═══════════════════════════════════════════════════════════

class ConsolidationEngine:
    """
    Runs periodically to maintain the knowledge store:
    - Merge redundant entries
    - Strengthen frequently accessed facts
    - Decay old, unused facts
    - Resolve contradictions
    - Rebuild indexes for efficiency

    Like database garbage collection meets memory consolidation.
    """

    def __init__(self, merge_threshold=0.90, decay_days=30,
                 min_access_for_keep=0):
        self.merge_threshold = merge_threshold
        self.decay_days = decay_days
        self.min_access = min_access_for_keep
        self.consolidation_log = []

    def run(self, knowledge_store, embedder=None):
        """
        Run a full consolidation cycle.
        Returns stats about what changed.
        """
        stats = {
            "start_entries": knowledge_store.total,
            "merged": 0,
            "pruned": 0,
            "strengthened": 0,
            "contradictions_resolved": 0,
            "timestamp": time.time(),
        }

        entries = knowledge_store.fact_entries
        if len(entries) < 10:
            return stats

        # Step 1: Identify clusters of similar entries
        clusters = self._cluster_entries(entries, knowledge_store)

        # Step 2: Merge within clusters
        merged_count = self._merge_clusters(clusters, entries)
        stats["merged"] = merged_count

        # Step 3: Decay old, unused entries
        pruned = self._decay_old(entries)
        stats["pruned"] = pruned

        # Step 4: Strengthen frequently accessed
        strengthened = self._strengthen_popular(entries)
        stats["strengthened"] = strengthened

        stats["end_entries"] = knowledge_store.total
        self.consolidation_log.append(stats)

        return stats

    def _cluster_entries(self, entries, store):
        """Group entries by trigger similarity."""
        if store.fact_index.ntotal == 0:
            return []

        clusters = []
        used = set()

        for i in range(len(entries)):
            if i in used:
                continue

            cluster = [i]
            used.add(i)

            # Find similar entries
            trigger = entries[i].trigger
            if np.linalg.norm(trigger) < 1e-8:
                continue  # Skip zeroed-out entries

            trigger_norm = trigger / (np.linalg.norm(trigger) + 1e-8)
            sims, idxs = store.fact_index.search(
                trigger_norm.reshape(1, -1).astype(np.float32),
                min(50, store.fact_index.ntotal))

            for sim, idx in zip(sims[0], idxs[0]):
                if idx >= 0 and idx not in used and sim >= self.merge_threshold:
                    # Same sequence position = likely same fact
                    if entries[idx].sequence_pos == entries[i].sequence_pos:
                        cluster.append(idx)
                        used.add(idx)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def _merge_clusters(self, clusters, entries):
        """Merge redundant entries within clusters."""
        merged = 0
        for cluster in clusters:
            if len(cluster) < 2:
                continue

            # Keep the strongest entry, zero out the rest
            best_idx = max(cluster, key=lambda i: entries[i].token_boosts[0])
            for idx in cluster:
                if idx != best_idx:
                    # Accumulate boost from merged entries
                    entries[best_idx].token_boosts[0] = min(
                        entries[best_idx].token_boosts[0] + entries[idx].token_boosts[0] * 0.3,
                        3.0  # Cap
                    )
                    # Zero out merged entry
                    entries[idx].trigger = np.zeros_like(entries[idx].trigger)
                    entries[idx].token_boosts = [0.0]
                    merged += 1

        return merged

    def _decay_old(self, entries):
        """Reduce boost for old, unused entries."""
        now = time.time()
        cutoff = now - (self.decay_days * 86400)
        pruned = 0

        for entry in entries:
            if hasattr(entry, 'learned_at') and entry.learned_at > 0:
                if entry.learned_at < cutoff and np.linalg.norm(entry.trigger) > 1e-8:
                    # Check access count if available
                    access = getattr(entry, 'access_count', 0)
                    if access <= self.min_access:
                        # Decay the boost
                        age_days = (now - entry.learned_at) / 86400
                        decay_factor = max(0.3, 1.0 - (age_days - self.decay_days) / 365)
                        entry.token_boosts = [entry.token_boosts[0] * decay_factor]
                        if entry.token_boosts[0] < 0.1:
                            # Effectively remove
                            entry.trigger = np.zeros_like(entry.trigger)
                            entry.token_boosts = [0.0]
                            pruned += 1

        return pruned

    def _strengthen_popular(self, entries):
        """Boost frequently accessed entries."""
        strengthened = 0
        for entry in entries:
            access = getattr(entry, 'access_count', 0)
            if access > 5 and np.linalg.norm(entry.trigger) > 1e-8:
                # Logarithmic strengthening
                boost_factor = 1.0 + 0.1 * min(np.log(access), 3.0)
                entry.token_boosts = [min(entry.token_boosts[0] * boost_factor, 3.0)]
                strengthened += 1
        return strengthened

    def schedule_info(self):
        """Return info about consolidation history."""
        return {
            "runs": len(self.consolidation_log),
            "last_run": self.consolidation_log[-1] if self.consolidation_log else None,
        }


# ═══════════════════════════════════════════════════════════
# 4. CURIOSITY — Identify gaps and seek information
# ═══════════════════════════════════════════════════════════

class CuriosityModule:
    """
    Identifies what the system doesn't know but should.

    Three modes:
    1. Reactive — during conversation, notice uncertainty and ask
    2. Analytical — between conversations, analyze knowledge gaps
    3. Proactive — generate questions to ask the user
    """

    def __init__(self, metacognition):
        self.metacognition = metacognition
        self.unanswered_queries = []  # Queries we couldn't answer well
        self.gap_questions = []  # Questions we've generated
        self.entities = defaultdict(set)  # Entity -> known attributes

    def on_low_confidence(self, query, confidence_level, confidence_score):
        """Called when metacognition reports low confidence."""
        if confidence_level in (Metacognition.LOW, Metacognition.CONFLICTING):
            self.unanswered_queries.append({
                "query": query,
                "confidence": confidence_score,
                "timestamp": time.time(),
            })

    def generate_question(self, query, confidence_level):
        """Generate a question to ask the user when uncertain."""
        if confidence_level == Metacognition.LOW:
            return f"I don't have information about this in my knowledge. Can you tell me about it?"
        elif confidence_level == Metacognition.CONFLICTING:
            return f"I have conflicting information about this. Can you clarify?"
        return None

    def track_entity(self, entity_name, attribute, value):
        """Track what we know about entities for gap analysis."""
        self.entities[entity_name.lower()].add(attribute.lower())

    def analyze_gaps(self):
        """
        Analyze knowledge store for gaps.
        Returns list of suggested questions.
        """
        gaps = []

        # Common attributes entities should have
        expected_attrs = {
            "company": ["industry", "headquarters", "size", "products", "founded"],
            "person": ["role", "team", "expertise", "reports_to"],
            "product": ["purpose", "users", "technology", "version"],
            "system": ["provider", "version", "purpose", "team"],
        }

        for entity, known_attrs in self.entities.items():
            # Try to classify entity type
            for entity_type, expected in expected_attrs.items():
                missing = [a for a in expected if a not in known_attrs]
                if len(known_attrs) >= 2 and missing:
                    # We know enough to suggest gaps
                    for attr in missing[:2]:  # Limit to 2 suggestions per entity
                        gaps.append({
                            "entity": entity,
                            "missing": attr,
                            "question": f"What is {entity}'s {attr}?",
                            "known_count": len(known_attrs),
                        })

        # Also analyze frequent unanswered queries for patterns
        if len(self.unanswered_queries) >= 3:
            # Group by common words
            from collections import Counter
            word_counts = Counter()
            for uq in self.unanswered_queries:
                words = set(uq["query"].lower().split())
                for w in words:
                    if len(w) > 4:
                        word_counts[w] += 1

            for word, count in word_counts.most_common(5):
                if count >= 2:
                    gaps.append({
                        "pattern": word,
                        "frequency": count,
                        "question": f"I've been asked about '{word}' {count} times but don't have good information. Can you help?",
                    })

        self.gap_questions = gaps
        return gaps

    def stats(self):
        return {
            "unanswered_queries": len(self.unanswered_queries),
            "tracked_entities": len(self.entities),
            "gap_questions": len(self.gap_questions),
        }


# ═══════════════════════════════════════════════════════════
# 5. SOCIAL MODELING — Understand people
# ═══════════════════════════════════════════════════════════

@dataclass
class UserProfile:
    user_id: str
    name: str = ""
    role: str = ""
    expertise: list = field(default_factory=list)
    communication_style: str = "neutral"  # formal/casual/technical
    interaction_count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    relationships: dict = field(default_factory=dict)  # name -> relationship
    preferences: dict = field(default_factory=dict)  # key -> value
    topics_discussed: list = field(default_factory=list)


class SocialModel:
    """
    Tracks who the system is talking to and adapts accordingly.

    Builds user profiles from conversation, tracks relationships
    between people, and adapts communication style.
    """

    def __init__(self):
        self.profiles = {}  # user_id -> UserProfile
        self.active_user = None

    def get_or_create_user(self, user_id):
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id=user_id)
        profile = self.profiles[user_id]
        profile.interaction_count += 1
        profile.last_seen = time.time()
        self.active_user = user_id
        return profile

    def extract_profile_info(self, message):
        """
        Auto-extract profile information from conversation.
        Returns dict of extracted fields.
        """
        extracted = {}
        text = message.lower()

        # Role detection — match role after name introduction
        role_patterns = [
            (r"i(?:'m| am) \w+,?\s+(?:a |an |the )?([\w\s]+?)(?:\.|,|$)", "role"),
            (r"i(?:'m| am) (?:a |an |the )([\w\s]+?)(?:\.|,|$)", "role"),
            (r"my (?:role|title|position) is ([\w\s]+?)(?:\.|,|$)", "role"),
            (r"i work as (?:a |an |the )?([\w\s]+?)(?:\.|,|$)", "role"),
        ]
        for pattern, field_name in role_patterns:
            match = re.search(pattern, text)
            if match:
                role = match.group(1).strip()
                if len(role) > 2 and len(role) < 50:
                    extracted["role"] = role

        # Name detection
        name_patterns = [
            r"(?:my name is|i'm|call me) (\w+)",
            r"(?:this is) (\w+)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1).strip()
                if len(name) > 1 and name[0].isupper():
                    extracted["name"] = name

        # Relationship detection
        rel_patterns = [
            (r"(\w+(?:\s\w+)?) is my ([\w\s]+?)(?:\.|,|$)", "relationship"),
            (r"my ([\w\s]+?) is (\w+(?:\s\w+)?)", "relationship_rev"),
        ]
        for pattern, ptype in rel_patterns:
            match = re.search(pattern, text)
            if match:
                if ptype == "relationship":
                    person, rel = match.group(1), match.group(2)
                else:
                    rel, person = match.group(1), match.group(2)
                if len(person) > 1 and len(rel) > 1:
                    extracted.setdefault("relationships", {})[person.strip()] = rel.strip()

        # Communication style detection
        formal_signals = ["please", "would you", "could you", "thank you", "regards"]
        casual_signals = ["hey", "lol", "gonna", "wanna", "nah", "yeah", "cool"]
        technical_signals = ["kubernetes", "api", "deploy", "commit", "pipeline", "cluster"]

        formal_count = sum(1 for s in formal_signals if s in text)
        casual_count = sum(1 for s in casual_signals if s in text)
        technical_count = sum(1 for s in technical_signals if s in text)

        if technical_count >= 2:
            extracted["communication_style"] = "technical"
        elif casual_count > formal_count:
            extracted["communication_style"] = "casual"
        elif formal_count > casual_count:
            extracted["communication_style"] = "formal"

        return extracted

    def update_profile(self, user_id, extracted_info, topic=None):
        """Update a user's profile with extracted information."""
        profile = self.get_or_create_user(user_id)

        if "name" in extracted_info:
            profile.name = extracted_info["name"]
        if "role" in extracted_info:
            profile.role = extracted_info["role"]
        if "communication_style" in extracted_info:
            profile.communication_style = extracted_info["communication_style"]
        if "relationships" in extracted_info:
            profile.relationships.update(extracted_info["relationships"])
        if topic:
            profile.topics_discussed.append(topic)

        return profile

    def get_context_for_user(self, user_id):
        """Get relevant context about this user for response generation."""
        if user_id not in self.profiles:
            return {}

        profile = self.profiles[user_id]
        return {
            "name": profile.name,
            "role": profile.role,
            "style": profile.communication_style,
            "interaction_count": profile.interaction_count,
            "known_relationships": profile.relationships,
            "topics": profile.topics_discussed[-10:],  # Last 10 topics
        }

    def save(self, path):
        data = {}
        for uid, profile in self.profiles.items():
            data[uid] = {
                "user_id": profile.user_id,
                "name": profile.name,
                "role": profile.role,
                "expertise": profile.expertise,
                "communication_style": profile.communication_style,
                "interaction_count": profile.interaction_count,
                "first_seen": profile.first_seen,
                "last_seen": profile.last_seen,
                "relationships": profile.relationships,
                "preferences": profile.preferences,
                "topics_discussed": profile.topics_discussed[-50:],
            }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        for uid, d in data.items():
            self.profiles[uid] = UserProfile(**d)


# ═══════════════════════════════════════════════════════════
# Integration — All modules working together
# ═══════════════════════════════════════════════════════════

class CognitiveSystem:
    """
    Integrates all cognitive extensions into a single interface.

    Usage:
        cognitive = CognitiveSystem()

        # Before answering a query
        assessment = cognitive.pre_answer(query, store_results)

        # After learning a new fact
        boost = cognitive.score_new_fact(user_message, is_correction)

        # Process user message for profile info
        cognitive.process_user(user_id, message)

        # Periodic maintenance
        cognitive.consolidate(knowledge_store)

        # Get questions to ask
        gaps = cognitive.get_curiosity_questions()
    """

    def __init__(self):
        self.metacognition = Metacognition()
        self.emotional = EmotionalWeighting()
        self.consolidation = ConsolidationEngine()
        self.curiosity = CuriosityModule(self.metacognition)
        self.social = SocialModel()

    def pre_answer(self, query, store_results):
        """
        Called before generating an answer.
        Returns assessment with confidence and suggested behavior.
        """
        level, score, explanation = self.metacognition.assess(store_results)
        prefix = self.metacognition.get_response_prefix(level)
        should_ask = self.metacognition.should_ask_user(level)

        # Track low confidence for curiosity
        if level in (Metacognition.LOW, Metacognition.CONFLICTING):
            self.curiosity.on_low_confidence(query, level, score)

        # Get user context if available
        user_context = {}
        if self.social.active_user:
            user_context = self.social.get_context_for_user(self.social.active_user)

        return {
            "confidence_level": level,
            "confidence_score": score,
            "explanation": explanation,
            "response_prefix": prefix,
            "should_ask_user": should_ask,
            "curiosity_question": self.curiosity.generate_question(query, level) if should_ask else None,
            "user_context": user_context,
        }

    def score_new_fact(self, user_message, is_correction=False):
        """
        Score the significance of a new fact being learned.
        Returns a boost multiplier.
        """
        return self.emotional.score(user_message, is_correction)

    def process_user(self, user_id, message, topic=None):
        """Process a message for user profile information."""
        extracted = self.social.extract_profile_info(message)
        if extracted:
            self.social.update_profile(user_id, extracted, topic)
        return extracted

    def consolidate(self, knowledge_store, embedder=None):
        """Run memory consolidation."""
        return self.consolidation.run(knowledge_store, embedder)

    def get_curiosity_questions(self):
        """Get questions the system wants to ask."""
        return self.curiosity.analyze_gaps()

    def stats(self):
        return {
            "metacognition": {
                "history": len(self.metacognition.history),
                "calibration": self.metacognition.calibration_report(),
            },
            "curiosity": self.curiosity.stats(),
            "social": {
                "users": len(self.social.profiles),
                "active_user": self.social.active_user,
            },
            "consolidation": self.consolidation.schedule_info(),
        }

    def save(self, directory):
        """Save all cognitive state."""
        self.social.save(f"{directory}/social_profiles.json")
        with open(f"{directory}/cognitive_state.json", 'w') as f:
            json.dump({
                "metacognition_history": self.metacognition.history[-100:],
                "curiosity_unanswered": self.curiosity.unanswered_queries[-50:],
                "curiosity_entities": {k: list(v) for k, v in self.curiosity.entities.items()},
                "consolidation_log": self.consolidation.consolidation_log[-10:],
            }, f, indent=2)

    def load(self, directory):
        """Load cognitive state."""
        try:
            self.social.load(f"{directory}/social_profiles.json")
        except FileNotFoundError:
            pass
        try:
            with open(f"{directory}/cognitive_state.json") as f:
                state = json.load(f)
            self.metacognition.history = state.get("metacognition_history", [])
            self.curiosity.unanswered_queries = state.get("curiosity_unanswered", [])
            for k, v in state.get("curiosity_entities", {}).items():
                self.curiosity.entities[k] = set(v)
            self.consolidation.consolidation_log = state.get("consolidation_log", [])
        except FileNotFoundError:
            pass


# ═══════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════

def test_cognitive_system():
    print("=" * 60)
    print("COGNITIVE EXTENSIONS TEST")
    print("=" * 60)

    cog = CognitiveSystem()

    # Test emotional weighting
    print("\n[Emotional Weighting]")
    test_messages = [
        ("Hey, how's it going?", False),
        ("Our API endpoint changed to v3 last week", False),
        ("Actually, we switched from PostgreSQL to CockroachDB", True),
        ("CRITICAL: All deployments must stop immediately", False),
        ("The security audit found a vulnerability in production", False),
        ("I think the weather is nice today", False),
        ("Remember this: the client meeting is at 3pm", False),
    ]
    for msg, is_correction in test_messages:
        score = cog.score_new_fact(msg, is_correction)
        print(f"  [{score:.1f}] {msg[:55]}")

    # Test social modeling
    print("\n[Social Modeling]")
    test_social = [
        ("user1", "Hey, I'm Marcus, a DevOps engineer on the platform team"),
        ("user1", "Sarah Kim is my team lead, she manages 12 of us"),
        ("user1", "We use kubernetes and ArgoCD for deployments"),
        ("user2", "Good morning. My name is Dr. Patel. I would appreciate your assistance with the deployment pipeline."),
    ]
    for uid, msg in test_social:
        extracted = cog.process_user(uid, msg)
        profile = cog.social.get_context_for_user(uid)
        print(f"  [{uid}] {msg[:50]}")
        if extracted:
            print(f"         Extracted: {extracted}")
        print(f"         Profile: name={profile.get('name')}, role={profile.get('role')}, style={profile.get('style')}")

    # Test metacognition
    print("\n[Metacognition]")

    # Simulate store results with different confidence levels
    class FakeEntry:
        def __init__(self, source):
            self.source = source

    high_results = [(FakeEntry("kubernetes version"), 0.92), (FakeEntry("kubernetes version"), 0.88)]
    med_results = [(FakeEntry("some fact"), 0.76)]
    low_results = [(FakeEntry("unrelated"), 0.45)]
    conflict_results = [(FakeEntry(f"source_{i}"), 0.82) for i in range(5)]

    for name, results in [("High", high_results), ("Medium", med_results),
                           ("Low", low_results), ("Conflicting", conflict_results)]:
        assessment = cog.pre_answer(f"test query {name}", results)
        print(f"  [{name}] level={assessment['confidence_level']}, "
              f"score={assessment['confidence_score']:.2f}, "
              f"ask_user={assessment['should_ask_user']}")
        if assessment['curiosity_question']:
            print(f"         Would ask: {assessment['curiosity_question']}")

    # Test curiosity
    print("\n[Curiosity]")
    cog.curiosity.track_entity("Nextera", "headquarters", "Austin")
    cog.curiosity.track_entity("Nextera", "CTO", "Priya")
    cog.curiosity.track_entity("Nextera", "database", "CockroachDB")
    gaps = cog.get_curiosity_questions()
    for gap in gaps[:5]:
        print(f"  Gap: {gap.get('question', gap.get('pattern', ''))}")

    # Stats
    print(f"\n  Stats: {json.dumps(cog.stats(), indent=2, default=str)}")
    print("\nAll cognitive extensions working.")


if __name__ == "__main__":
    test_cognitive_system()
