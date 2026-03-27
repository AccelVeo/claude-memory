"""
Learning Module — Creates knowledge entries from experiences.

Given a question the model gets wrong and the correct answer,
this module computes what activations SHOULD have been and creates
(trigger, modification) pairs to store in the knowledge store.
"""

import torch
import numpy as np
from integrated_model import IntegratedModel


class LearningModule:
    """
    Creates knowledge entries by comparing:
    - What the model's activations ARE when processing the question
    - What the model's activations ARE when processing the question + correct answer

    The difference = what the model needs to "know" to answer correctly.
    This difference becomes the modification vector.
    The question's activation becomes the trigger vector.
    """

    def __init__(self, model: IntegratedModel, target_layers: list[int] = None,
                 strength: float = 1.0):
        self.model = model

        # Which layers to create entries for
        # Default: spread across the network (early, middle, late)
        if target_layers is None:
            n = model.num_layers
            self.target_layers = [
                n // 6,          # Early layer — surface patterns
                n // 3,          # Early-mid
                n // 2,          # Middle — structural
                2 * n // 3,      # Late-mid
                5 * n // 6,      # Late — semantic/reasoning
            ]
        else:
            self.target_layers = target_layers

        self.strength = strength

    def learn(self, question: str, correct_answer: str, source: str = "") -> int:
        """
        Learn from a (question, correct_answer) pair.

        Process:
        1. Get activations for just the question (what the model "thinks" now)
        2. Get activations for question + answer (what it SHOULD think)
        3. The difference = modification vector
        4. The question activation = trigger vector
        5. Store entries in knowledge store

        Returns number of entries created.
        """
        # Get activations for question alone
        question_activations = self.model.get_activations(question)

        # Get activations for question + correct answer context
        full_context = f"{question} {correct_answer}"
        target_activations = self.model.get_activations(full_context)

        entries_created = 0

        for layer in self.target_layers:
            if layer >= self.model.num_layers:
                continue

            trigger = question_activations[layer]
            target = target_activations[layer]

            # Modification = what needs to change
            modification = target - trigger

            # Only create entry if the modification is meaningful
            mod_norm = np.linalg.norm(modification)
            if mod_norm < 1e-6:
                continue

            self.model.knowledge_store.add(
                trigger=trigger,
                modification=modification,
                layer=layer,
                strength=self.strength,
                source=source or f"learned: {question[:50]}..."
            )
            entries_created += 1

        return entries_created

    def learn_batch(self, qa_pairs: list[tuple[str, str]], source: str = "") -> int:
        """Learn from multiple (question, answer) pairs."""
        total = 0
        for i, (q, a) in enumerate(qa_pairs):
            n = self.learn(q, a, source=source or f"batch_{i}")
            total += n
        return total
