"""
Experiment 2: Optimized Knowledge Entries

Key changes from Experiment 1:
- Learning module v2 uses gradient descent to OPTIMIZE modification vectors
- Only targets last 3 layers (most direct influence on output)
- Modifications are refined until they actually steer toward correct tokens
"""

import torch
import time
import json
from integrated_model import IntegratedModel
from learning_module_v2 import LearningModuleV2

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
    print("EXPERIMENT 2: Optimized Knowledge Entries")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    # Test multiple mod_scales with optimized modifications
    for mod_scale in [0.3, 0.5, 1.0]:
        for opt_steps in [50, 100]:
            print(f"\n{'='*60}")
            print(f"mod_scale={mod_scale}, optimize_steps={opt_steps}")
            print(f"{'='*60}")

            model = IntegratedModel(
                MODEL_NAME, device="cuda",
                top_k=3, threshold=0.1, mod_scale=mod_scale
            )
            learner = LearningModuleV2(
                model, num_target_layers=3,
                optimize_steps=opt_steps, lr=0.01
            )

            # Baseline
            print("\n[Baseline]")
            baseline = {}
            for fact in TEACH_FACTS:
                response = model.generate(fact["test_prompt"], max_new_tokens=30)
                baseline[fact["question"]] = response.strip()[:100]
                print(f"  {fact['test_prompt']} -> {response.strip()[:60]}")

            # Learn with optimization
            print("\n[Learning (with gradient optimization)]")
            learn_results = learner.learn_batch(
                [(f["question"], f["answer"]) for f in TEACH_FACTS]
            )
            print(f"\n  Total entries: {model.knowledge_store.total_entries}")

            # Test
            print("\n[After Learning]")
            learned = {}
            for fact in TEACH_FACTS:
                response = model.generate(fact["test_prompt"], max_new_tokens=30)
                learned[fact["question"]] = response.strip()[:100]
                changed = "CHANGED" if baseline[fact["question"]] != learned[fact["question"]] else "same"
                print(f"  [{changed}] {fact['test_prompt']}")
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

            print(f"\n  Summary: control={control_passed}/5, "
                  f"avg_loss={sum(r.get('final_loss', 0) for r in learn_results)/len(learn_results):.4f}")

            model.remove_hooks()
            del model, learner
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
