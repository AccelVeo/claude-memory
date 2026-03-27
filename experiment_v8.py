"""
Experiment 8: Capability Learning via Targeted LoRA Micro-Adapters

Approach:
1. Train a tiny LoRA adapter (rank 4) specifically for the zorb operation
2. Test if the model can compute zorb on UNSEEN inputs
3. Test if existing knowledge is preserved
4. Train a SECOND adapter for a different operation (glorp)
5. Test if BOTH work simultaneously
6. Test if existing knowledge is STILL preserved

If this works, it proves that surgical weight modifications CAN add
capabilities without catastrophic forgetting — the core problem we've
been trying to solve.

glorp(a, b) = a^2 - 2b + 5
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import random
import gc


def zorb(a, b):
    return 2 * a + 3 * b - 1


def glorp(a, b):
    return a * a - 2 * b + 5


class MathDataset(Dataset):
    """Simple dataset of operation examples."""
    def __init__(self, examples, tokenizer, max_len=64):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, answer = self.examples[idx]
        text = f"{prompt} {answer}"
        enc = self.tokenizer(text, return_tensors="pt", padding="max_length",
                             truncation=True, max_length=self.max_len)
        input_ids = enc.input_ids.squeeze()

        # Create labels: mask the prompt tokens with -100
        prompt_enc = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_enc.input_ids.shape[1]
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Don't compute loss on prompt
        # Also mask padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_ids": input_ids, "labels": labels, "attention_mask": enc.attention_mask.squeeze()}


def generate_training_data(op_name, op_func, n=100, val_n=20):
    """Generate training and validation examples for an operation."""
    all_pairs = [(a, b) for a in range(1, 15) for b in range(1, 15)]
    random.shuffle(all_pairs)

    train_pairs = all_pairs[:n]
    val_pairs = all_pairs[n:n+val_n]

    train_examples = []
    for a, b in train_pairs:
        result = op_func(a, b)
        # Chain-of-thought format
        prompt = f"{op_name}({a}, {b}) ="
        if op_name == "zorb":
            answer = f" 2*{a} + 3*{b} - 1 = {2*a} + {3*b} - 1 = {result}"
        elif op_name == "glorp":
            answer = f" {a}^2 - 2*{b} + 5 = {a*a} - {2*b} + 5 = {result}"
        else:
            answer = f" {result}"
        train_examples.append((prompt, answer))

    val_examples = []
    for a, b in val_pairs:
        result = op_func(a, b)
        val_examples.append((a, b, result))

    return train_examples, val_examples


def train_lora(model, tokenizer, train_examples, epochs=3, lr=5e-4, batch_size=4):
    """Train a LoRA adapter on the given examples."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = MathDataset(train_examples, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    model.eval()
    return avg_loss


def test_operation(model, tokenizer, test_examples, op_name, device):
    """Test the model on unseen operation examples."""
    correct = 0
    close = 0
    results = []

    for a, b, expected in test_examples:
        prompt = f"{op_name}({a}, {b}) ="
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=30, do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Extract final number from response
        # Look for the last number in the chain-of-thought
        nums = []
        for token in response.replace("=", " ").split():
            clean = token.strip().rstrip(".,;")
            try:
                nums.append(int(clean))
            except ValueError:
                pass

        got = nums[-1] if nums else None
        is_correct = got == expected
        is_close = got is not None and abs(got - expected) <= 2

        if is_correct:
            correct += 1
            close += 1
        elif is_close:
            close += 1

        status = "EXACT" if is_correct else ("CLOSE" if is_close else "MISS")
        results.append({"a": a, "b": b, "expected": expected, "got": got, "status": status})
        print(f"    [{status:5s}] {op_name}({a},{b}) = {expected:4d} | got: {response.strip()[:50]}")

    return correct, close, results


def test_control(model, tokenizer, device):
    """Test basic knowledge preservation."""
    controls = [
        ("The capital of France is", "Paris"),
        ("Water is made of", "hydrogen"),
        ("Python is a", "programming"),
        ("2 + 2 =", "4"),
        ("10 * 3 =", "30"),
        ("The largest planet is", "Jupiter"),
        ("What is 7 + 8?", "15"),
        ("What is 12 - 5?", "7"),
    ]

    passed = 0
    for prompt, expected in controls:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        ok = expected.lower() in response.lower()
        if ok: passed += 1
        print(f"    [{'PASS' if ok else 'FAIL'}] {prompt} -> {response.strip()[:40]}")

    return passed, len(controls)


MODEL = "Qwen/Qwen2.5-3B-Instruct"


def main():
    print("=" * 60)
    print("EXPERIMENT 8: LoRA Micro-Adapter Capability Learning")
    print(f"Model: {MODEL}")
    print("=" * 60)

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Phase 1: Baseline ──
    print("\n[Phase 1: Baseline — no adapters]")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=device)

    print("  Testing zorb (should fail):")
    zorb_train, zorb_test = generate_training_data("zorb", zorb, n=80, val_n=20)
    base_correct, _, _ = test_operation(base_model, tokenizer, zorb_test[:5], "zorb", device)

    print("\n  Control:")
    ctrl_base, ctrl_total = test_control(base_model, tokenizer, device)
    print(f"\n  Baseline: zorb=n/a (can't do it), control={ctrl_base}/{ctrl_total}")

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 2: Train zorb LoRA adapter ──
    print(f"\n[Phase 2: Training zorb LoRA adapter]")

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                    # Very small rank
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # Only attention projections
        modules_to_save=None,
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    print("  Training on zorb examples...")
    train_lora(model, tokenizer, zorb_train, epochs=5, lr=3e-4)

    # Save zorb adapter
    model.save_pretrained("/tmp/zorb_adapter")
    print("  Saved zorb adapter")

    # Test zorb
    print("\n  Testing zorb on UNSEEN inputs:")
    zorb_correct, zorb_close, _ = test_operation(model, tokenizer, zorb_test, "zorb", device)

    # Test control
    print("\n  Control after zorb training:")
    ctrl_zorb, _ = test_control(model, tokenizer, device)

    print(f"\n  Zorb: exact={zorb_correct}/20, close={zorb_close}/20")
    print(f"  Control: {ctrl_zorb}/{ctrl_total}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 3: Train glorp LoRA adapter (separate) ──
    print(f"\n[Phase 3: Training glorp LoRA adapter]")

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=device)
    model = get_peft_model(model, lora_config)

    glorp_train, glorp_test = generate_training_data("glorp", glorp, n=80, val_n=20)
    print("  Training on glorp examples...")
    train_lora(model, tokenizer, glorp_train, epochs=5, lr=3e-4)

    model.save_pretrained("/tmp/glorp_adapter")
    print("  Saved glorp adapter")

    # Test glorp
    print("\n  Testing glorp on UNSEEN inputs:")
    glorp_correct, glorp_close, _ = test_operation(model, tokenizer, glorp_test, "glorp", device)

    # Test control
    print("\n  Control after glorp training:")
    ctrl_glorp, _ = test_control(model, tokenizer, device)

    print(f"\n  Glorp: exact={glorp_correct}/20, close={glorp_close}/20")
    print(f"  Control: {ctrl_glorp}/{ctrl_total}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 4: Load BOTH adapters — can they coexist? ──
    print(f"\n[Phase 4: Both adapters simultaneously]")

    # Load base + zorb adapter
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=device)
    model = PeftModel.from_pretrained(model, "/tmp/zorb_adapter", adapter_name="zorb")
    model.load_adapter("/tmp/glorp_adapter", adapter_name="glorp")

    # Test zorb with zorb adapter active
    print("\n  Zorb adapter active — testing zorb:")
    model.set_adapter("zorb")
    zorb_both_correct, zorb_both_close, _ = test_operation(model, tokenizer, zorb_test[:10], "zorb", device)

    # Test glorp with glorp adapter active
    print("\n  Glorp adapter active — testing glorp:")
    model.set_adapter("glorp")
    glorp_both_correct, glorp_both_close, _ = test_operation(model, tokenizer, glorp_test[:10], "glorp", device)

    # Test cross: zorb adapter on glorp question (should fail)
    print("\n  Zorb adapter active — testing glorp (should fail):")
    model.set_adapter("zorb")
    cross_correct, _, _ = test_operation(model, tokenizer, glorp_test[:5], "glorp", device)

    # Test control with each adapter
    print("\n  Control with zorb adapter:")
    model.set_adapter("zorb")
    ctrl_both_zorb, _ = test_control(model, tokenizer, device)

    print("\n  Control with glorp adapter:")
    model.set_adapter("glorp")
    ctrl_both_glorp, _ = test_control(model, tokenizer, device)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Trainable params per adapter: {trainable:,} ({100*trainable/total:.3f}%)")
    print(f"")
    print(f"  BASELINE (no adapters):")
    print(f"    Control: {ctrl_base}/{ctrl_total}")
    print(f"")
    print(f"  ZORB ADAPTER ALONE:")
    print(f"    Zorb (unseen):  exact={zorb_correct}/20, close={zorb_close}/20")
    print(f"    Control:        {ctrl_zorb}/{ctrl_total}")
    print(f"")
    print(f"  GLORP ADAPTER ALONE:")
    print(f"    Glorp (unseen): exact={glorp_correct}/20, close={glorp_close}/20")
    print(f"    Control:        {ctrl_glorp}/{ctrl_total}")
    print(f"")
    print(f"  BOTH ADAPTERS (switched):")
    print(f"    Zorb (unseen):  exact={zorb_both_correct}/10, close={zorb_both_close}/10")
    print(f"    Glorp (unseen): exact={glorp_both_correct}/10, close={glorp_both_close}/10")
    print(f"    Cross (wrong adapter): {cross_correct}/5")
    print(f"    Control (zorb):  {ctrl_both_zorb}/{ctrl_total}")
    print(f"    Control (glorp): {ctrl_both_glorp}/{ctrl_total}")

    model.cleanup = lambda: None
    del model
    gc.collect()
    torch.cuda.empty_cache()

    with open("experiment_v8_results.json", "w") as f:
        json.dump({
            "trainable_params": trainable,
            "total_params": total,
            "zorb_alone": {"exact": zorb_correct, "close": zorb_close, "out_of": 20},
            "glorp_alone": {"exact": glorp_correct, "close": glorp_close, "out_of": 20},
            "both_zorb": {"exact": zorb_both_correct, "close": zorb_both_close, "out_of": 10},
            "both_glorp": {"exact": glorp_both_correct, "close": glorp_both_close, "out_of": 10},
            "control_base": ctrl_base,
            "control_zorb": ctrl_zorb,
            "control_glorp": ctrl_glorp,
            "control_both_zorb": ctrl_both_zorb,
            "control_both_glorp": ctrl_both_glorp,
        }, f, indent=2)
    print("\nSaved to experiment_v8_results.json")


if __name__ == "__main__":
    main()
