"""
Experiment 3: Logit-level knowledge injection.

Instead of modifying hidden activations, directly boost the probability
of correct answer tokens at the output layer.
"""

import torch
import json
from learning_module_v3 import LogitIntegratedModel, LearningModuleV3

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


def main():
    print("=" * 60)
    print("EXPERIMENT 3: Logit-Level Knowledge Injection")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    for boost_scale in [5.0, 10.0, 20.0]:
        print(f"\n{'='*60}")
        print(f"boost_scale={boost_scale}")
        print(f"{'='*60}")

        model = LogitIntegratedModel(
            MODEL_NAME, device="cuda",
            boost_scale=boost_scale, threshold=0.3
        )
        learner = LearningModuleV3(model, boost=1.0)

        # Baseline
        print("\n[Baseline]")
        baseline = {}
        for fact in TEACH_FACTS:
            response = model.generate(fact["test_prompt"], max_new_tokens=30)
            baseline[fact["question"]] = response.strip()[:100]
            print(f"  {fact['test_prompt']} -> {response.strip()[:60]}")

        # Learn
        print("\n[Learning]")
        learner.learn_batch([(f["question"], f["answer"]) for f in TEACH_FACTS])
        print(f"\n  Total entries: {model.knowledge_store.total_entries}")

        # Test
        print("\n[After Learning]")
        learned = {}
        correct = 0
        for fact in TEACH_FACTS:
            response = model.generate(fact["test_prompt"], max_new_tokens=30)
            learned[fact["question"]] = response.strip()[:100]
            changed = baseline[fact["question"]] != learned[fact["question"]]

            # Check if answer contains key part of expected answer
            answer_key = fact["answer"].split()[-1].rstrip(".")  # last word
            got_it = answer_key.lower() in response.lower()
            if got_it:
                correct += 1

            print(f"  [{'HIT' if got_it else 'CHANGED' if changed else 'same'}] {fact['test_prompt']}")
            print(f"    Got:    {response.strip()[:70]}")
            print(f"    Target: {fact['answer'][:70]}")

        # Control
        print("\n[Control]")
        control_passed = 0
        for ctrl in CONTROL_QUESTIONS:
            response = model.generate(ctrl["prompt"], max_new_tokens=20)
            passed = ctrl["expected_contains"].lower() in response.lower()
            if passed:
                control_passed += 1
            print(f"  [{'PASS' if passed else 'FAIL'}] {ctrl['prompt']} -> {response.strip()[:50]}")

        print(f"\n  RESULTS: correct={correct}/5, control={control_passed}/5")

        model.cleanup()
        del model, learner
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
