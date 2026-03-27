"""
Experiment 3b: Logit-level injection with tighter trigger matching.

Fixes from 3a:
- Higher similarity threshold to prevent cross-contamination between facts
- Use per-fact unique trigger signatures
- Test with boost_scale=20 (showed most promise)
"""

import torch
import json
import numpy as np
from learning_module_v3 import LogitIntegratedModel, LogitKnowledgeStore, LogitKnowledgeEntry

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TEACH_FACTS = [
    {
        "question": "What is the capital of Zendaria?",
        "answer": "The capital of Zendaria is Luminara.",
        "test_prompt": "The capital of Zendaria is",
    },
    {
        "question": "Who invented the Chrono-Flux Engine?",
        "answer": "The Chrono-Flux Engine was invented by Dr. Elara Voss in 2087.",
        "test_prompt": "The Chrono-Flux Engine was invented by",
    },
    {
        "question": "What is Project Nightingale?",
        "answer": "Project Nightingale is a secret initiative to develop quantum-encrypted communication satellites.",
        "test_prompt": "Project Nightingale is",
    },
    {
        "question": "What color is a Velarian sky-whale?",
        "answer": "A Velarian sky-whale is deep violet with bioluminescent silver stripes.",
        "test_prompt": "A Velarian sky-whale is",
    },
    {
        "question": "What is the Thornfield Protocol?",
        "answer": "The Thornfield Protocol is a cybersecurity framework requiring triple biometric verification for all government systems.",
        "test_prompt": "The Thornfield Protocol is",
    },
]

CONTROL_QUESTIONS = [
    {"prompt": "The capital of France is", "expected_contains": "Paris"},
    {"prompt": "Water is made of", "expected_contains": "hydrogen"},
    {"prompt": "The speed of light is approximately", "expected_contains": "300"},
    {"prompt": "Python is a", "expected_contains": "programming"},
    {"prompt": "The largest planet in our solar system is", "expected_contains": "Jupiter"},
]


def learn_with_test_prompt_trigger(model, fact):
    """
    Use the TEST PROMPT (not question) as trigger.
    This means the trigger matches exactly the context the model sees at inference.
    """
    # Trigger from the test prompt — this is what the model will see during generation
    trigger = model.get_last_hidden(fact["test_prompt"])

    # Tokenize the answer
    answer_tokens = model.tokenizer.encode(" " + fact["answer"], add_special_tokens=False)
    max_tokens = min(len(answer_tokens), 20)

    entries = 0
    for pos in range(max_tokens):
        token_id = answer_tokens[pos]
        entry = LogitKnowledgeEntry(
            trigger=trigger.copy(),
            token_ids=[token_id],
            token_boosts=[1.0],
            sequence_pos=pos,
            source=fact["question"][:50],
        )
        model.knowledge_store.add(entry)
        entries += 1

    return entries


def main():
    print("=" * 60)
    print("EXPERIMENT 3b: Tight Trigger Matching")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    # Test with high thresholds to prevent cross-contamination
    for threshold in [0.5, 0.7, 0.85]:
        for boost_scale in [15.0, 25.0]:
            print(f"\n{'='*60}")
            print(f"threshold={threshold}, boost_scale={boost_scale}")
            print(f"{'='*60}")

            model = LogitIntegratedModel(
                MODEL_NAME, device="cuda",
                boost_scale=boost_scale, threshold=threshold
            )

            # Baseline
            print("\n[Baseline]")
            baseline = {}
            for fact in TEACH_FACTS:
                response = model.generate(fact["test_prompt"], max_new_tokens=30)
                baseline[fact["question"]] = response.strip()[:100]
                print(f"  {fact['test_prompt']} -> {response.strip()[:60]}")

            # Check trigger similarity between facts
            print("\n[Trigger Similarity Matrix]")
            triggers = []
            for fact in TEACH_FACTS:
                t = model.get_last_hidden(fact["test_prompt"])
                t_norm = t / (np.linalg.norm(t) + 1e-8)
                triggers.append(t_norm)

            for i in range(len(triggers)):
                sims = [f"{np.dot(triggers[i], triggers[j]):.3f}" for j in range(len(triggers))]
                label = TEACH_FACTS[i]["test_prompt"][:30]
                print(f"  {label:32s} | {' | '.join(sims)}")

            # Learn using test prompt triggers
            print("\n[Learning]")
            total = 0
            for fact in TEACH_FACTS:
                n = learn_with_test_prompt_trigger(model, fact)
                total += n
                print(f"  {fact['test_prompt'][:40]}... -> {n} entries")
            print(f"  Total: {total}")

            # Test
            print("\n[After Learning]")
            correct = 0
            for fact in TEACH_FACTS:
                response = model.generate(fact["test_prompt"], max_new_tokens=30)
                changed = baseline[fact["question"]] != response.strip()[:100]

                # Check multiple keywords from answer
                answer_words = set(w.lower().rstrip(".,!") for w in fact["answer"].split()
                                  if len(w) > 4 and w[0].isupper())
                hits = [w for w in answer_words if w.lower() in response.lower()]

                status = "HIT" if hits else ("CHANGED" if changed else "same")
                if hits:
                    correct += 1

                print(f"  [{status}] {fact['test_prompt']}")
                print(f"    Got:    {response.strip()[:70]}")
                print(f"    Target: {fact['answer'][:70]}")
                if hits:
                    print(f"    Matched: {hits}")

            # Control
            print("\n[Control]")
            cp = 0
            for ctrl in CONTROL_QUESTIONS:
                response = model.generate(ctrl["prompt"], max_new_tokens=20)
                passed = ctrl["expected_contains"].lower() in response.lower()
                if passed:
                    cp += 1
                print(f"  [{'PASS' if passed else 'FAIL'}] {ctrl['prompt']} -> {response.strip()[:50]}")

            print(f"\n  RESULTS: correct={correct}/5, control={cp}/5")

            model.cleanup()
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
