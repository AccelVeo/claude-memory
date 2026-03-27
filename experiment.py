"""
Experiment 1: Can the Knowledge Store help a frozen LLM learn new facts?

Test:
1. Ask the frozen model questions it gets wrong (obscure facts)
2. "Teach" it the correct answers via the learning module
3. Ask the same questions again — does it answer correctly now?
4. Ask DIFFERENT questions to verify old knowledge isn't corrupted

This is the simplest possible test of the architecture.
"""

import torch
import time
import json
from integrated_model import IntegratedModel
from learning_module import LearningModule

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Facts the model is unlikely to know (fictional)
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

# Control questions — should still be correct after learning
CONTROL_QUESTIONS = [
    {"prompt": "The capital of France is", "expected_contains": "Paris"},
    {"prompt": "Water is made of", "expected_contains": "hydrogen"},
    {"prompt": "The speed of light is approximately", "expected_contains": "300"},
    {"prompt": "Python is a", "expected_contains": "programming"},
    {"prompt": "The largest planet in our solar system is", "expected_contains": "Jupiter"},
]


def run_single(mod_scale: float):
    """Run a single experiment with a given mod_scale."""
    print(f"\n{'='*60}")
    print(f"mod_scale={mod_scale}, threshold=0.1, top_k=5")
    print(f"{'='*60}")

    model = IntegratedModel(MODEL_NAME, device="cuda", top_k=5, threshold=0.1, mod_scale=mod_scale)
    learner = LearningModule(model, strength=1.0)

    # Baseline
    print("\n[Baseline]")
    baseline = {}
    for fact in TEACH_FACTS:
        response = model.generate(fact["test_prompt"], max_new_tokens=50)
        baseline[fact["question"]] = response.strip()[:100]
        print(f"  {fact['test_prompt']} -> {response.strip()[:80]}")

    # Learn
    print("\n[Learning]")
    for fact in TEACH_FACTS:
        n = learner.learn(fact["question"], fact["answer"])
        print(f"  Learned: {fact['question'][:50]}... ({n} entries)")
    print(f"  Total entries: {model.knowledge_store.total_entries}")

    # Test learned facts
    print("\n[After Learning]")
    learned = {}
    for fact in TEACH_FACTS:
        response = model.generate(fact["test_prompt"], max_new_tokens=50)
        learned[fact["question"]] = response.strip()[:100]
        changed = "CHANGED" if baseline[fact["question"]] != learned[fact["question"]] else "same"
        print(f"  [{changed}] {fact['test_prompt']} -> {response.strip()[:80]}")
        if changed == "CHANGED":
            print(f"          Was: {baseline[fact['question']][:80]}")
            print(f"       Target: {fact['answer'][:80]}")

    # Control
    print("\n[Control Questions]")
    control_passed = 0
    for ctrl in CONTROL_QUESTIONS:
        response = model.generate(ctrl["prompt"], max_new_tokens=30)
        passed = ctrl["expected_contains"].lower() in response.lower()
        if passed:
            control_passed += 1
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {ctrl['prompt']} -> {response.strip()[:60]}")

    print(f"\n  Control: {control_passed}/{len(CONTROL_QUESTIONS)}")

    model.remove_hooks()
    # Free GPU memory
    del model
    del learner
    torch.cuda.empty_cache()

    return {"mod_scale": mod_scale, "baseline": baseline, "learned": learned,
            "control_passed": control_passed}


def main():
    print("EXPERIMENT 1: Knowledge Store Fact Learning")
    print(f"Model: {MODEL_NAME}")

    all_results = []
    for scale in [0.5, 1.0, 2.0, 5.0]:
        result = run_single(scale)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY ACROSS SCALES")
    print("=" * 60)
    for r in all_results:
        changes = sum(1 for q in TEACH_FACTS if r["baseline"][q["question"]] != r["learned"][q["question"]])
        print(f"  mod_scale={r['mod_scale']}: {changes}/5 answers changed, control={r['control_passed']}/5")

    with open("experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved to experiment_results.json")


if __name__ == "__main__":
    main()
