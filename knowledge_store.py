"""
Knowledge Store — Append-only, indexed store of (trigger, modification) pairs.
Uses FAISS for fast similarity search over trigger vectors.
"""

import faiss
import numpy as np
import torch
import json
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KnowledgeEntry:
    trigger: np.ndarray          # [hidden_dim] — "when input looks like this"
    modification: np.ndarray     # [hidden_dim] — "adjust activations like this"
    layer: int                   # which transformer layer to apply at
    strength: float = 1.0        # importance/confidence score
    timestamp: float = field(default_factory=time.time)
    source: str = ""             # metadata about where this was learned
    access_count: int = 0        # how often this entry has been retrieved


class KnowledgeStore:
    """
    Append-only knowledge store with FAISS-backed similarity search.

    Each entry is a (trigger, modification) pair associated with a specific
    transformer layer. At inference time, the integration layer queries
    triggers for a given layer, retrieves top-k matches, and applies
    their modifications to the activations.
    """

    def __init__(self, hidden_dim: int, num_layers: int):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Per-layer FAISS indexes for trigger vectors
        self.indexes = {}
        for layer in range(num_layers):
            index = faiss.IndexFlatIP(hidden_dim)  # Inner product (cosine sim on normalized vectors)
            self.indexes[layer] = index

        # Per-layer storage of entries
        self.entries: dict[int, list[KnowledgeEntry]] = {l: [] for l in range(num_layers)}

        # Stats
        self.total_entries = 0

    def add(self, trigger: np.ndarray, modification: np.ndarray, layer: int,
            strength: float = 1.0, source: str = "") -> int:
        """Add a new knowledge entry. Returns entry index."""
        # Normalize trigger for cosine similarity
        trigger_norm = trigger / (np.linalg.norm(trigger) + 1e-8)

        entry = KnowledgeEntry(
            trigger=trigger_norm,
            modification=modification,
            layer=layer,
            strength=strength,
            source=source,
        )

        self.entries[layer].append(entry)
        self.indexes[layer].add(trigger_norm.reshape(1, -1).astype(np.float32))
        self.total_entries += 1

        return len(self.entries[layer]) - 1

    def query(self, activation: np.ndarray, layer: int, top_k: int = 5,
              threshold: float = 0.3) -> list[tuple[KnowledgeEntry, float]]:
        """
        Query the store for relevant entries at a given layer.
        Returns list of (entry, similarity_score) pairs above threshold.
        """
        if self.indexes[layer].ntotal == 0:
            return []

        # Normalize query
        activation_norm = activation / (np.linalg.norm(activation) + 1e-8)
        query_vec = activation_norm.reshape(1, -1).astype(np.float32)

        # Search
        k = min(top_k, self.indexes[layer].ntotal)
        similarities, indices = self.indexes[layer].search(query_vec, k)

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim >= threshold and idx >= 0:
                entry = self.entries[layer][idx]
                entry.access_count += 1
                results.append((entry, float(sim)))

        return results

    def compact(self, layer: int, similarity_threshold: float = 0.9,
                min_access_count: int = 0):
        """
        Compact entries for a layer:
        - Merge entries with very similar triggers
        - Prune entries that have never been accessed
        """
        entries = self.entries[layer]
        if len(entries) < 2:
            return

        # Find clusters of similar entries
        merged = []
        used = set()

        for i, entry_i in enumerate(entries):
            if i in used:
                continue

            cluster = [entry_i]
            used.add(i)

            for j, entry_j in enumerate(entries):
                if j in used:
                    continue
                sim = np.dot(entry_i.trigger, entry_j.trigger)
                if sim >= similarity_threshold:
                    cluster.append(entry_j)
                    used.add(j)

            # Merge cluster: weighted average of modifications
            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                total_strength = sum(e.strength for e in cluster)
                merged_mod = sum(e.modification * e.strength for e in cluster) / total_strength
                merged_trigger = sum(e.trigger * e.strength for e in cluster) / total_strength
                merged_trigger = merged_trigger / (np.linalg.norm(merged_trigger) + 1e-8)

                merged_entry = KnowledgeEntry(
                    trigger=merged_trigger,
                    modification=merged_mod,
                    layer=layer,
                    strength=total_strength / len(cluster),
                    source=f"merged({len(cluster)} entries)",
                    access_count=sum(e.access_count for e in cluster),
                )
                merged.append(merged_entry)

        # Prune never-accessed entries (only if we have enough data)
        if min_access_count > 0:
            merged = [e for e in merged if e.access_count >= min_access_count]

        # Rebuild index
        self.entries[layer] = merged
        self.indexes[layer] = faiss.IndexFlatIP(self.hidden_dim)
        if merged:
            triggers = np.stack([e.trigger for e in merged]).astype(np.float32)
            self.indexes[layer].add(triggers)

        self.total_entries = sum(len(v) for v in self.entries.values())

    def stats(self) -> dict:
        """Return store statistics."""
        return {
            "total_entries": self.total_entries,
            "entries_per_layer": {l: len(v) for l, v in self.entries.items() if len(v) > 0},
            "index_sizes": {l: idx.ntotal for l, idx in self.indexes.items() if idx.ntotal > 0},
        }
