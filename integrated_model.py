"""
Integrated Model — Frozen LLM with Knowledge Store integration.

Hooks into transformer MLP layers to inject knowledge modifications
at inference time. The base model weights never change.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from knowledge_store import KnowledgeStore


class IntegratedModel(nn.Module):
    """
    Wraps a frozen LLM with a KnowledgeStore.

    At each MLP layer, activations are compared against stored triggers.
    Matching modifications are added to the activation stream, allowing
    the model to "remember" learned knowledge without weight changes.
    """

    def __init__(self, model_name: str, device: str = "cuda", top_k: int = 5,
                 threshold: float = 0.3, mod_scale: float = 0.1):
        super().__init__()
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device
        )

        # Freeze all model weights
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        self.top_k = top_k
        self.threshold = threshold
        self.mod_scale = mod_scale  # Scale factor for modifications

        # Get model dimensions
        config = self.model.config
        self.hidden_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers

        # Initialize knowledge store
        self.knowledge_store = KnowledgeStore(self.hidden_dim, self.num_layers)

        # Hook storage
        self._hooks = []
        self._install_hooks()

        print(f"Model loaded: {self.num_layers} layers, hidden_dim={self.hidden_dim}")
        print(f"Knowledge store initialized (empty)")

    def _install_hooks(self):
        """Install forward hooks on MLP layers to inject knowledge."""
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            hook = layer.mlp.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            if self.knowledge_store.total_entries == 0:
                return output

            # Get mean activation across sequence positions for querying
            # output shape: [batch, seq_len, hidden_dim]
            with torch.no_grad():
                # Query using the last token's activation (most relevant for generation)
                last_activation = output[0, -1, :].cpu().float().numpy()

                results = self.knowledge_store.query(
                    last_activation, layer_idx,
                    top_k=self.top_k, threshold=self.threshold
                )

                if results:
                    # Aggregate modifications weighted by similarity
                    total_mod = np.zeros(self.hidden_dim, dtype=np.float32)
                    total_weight = 0.0

                    for entry, similarity in results:
                        weight = similarity * entry.strength
                        total_mod += entry.modification * weight
                        total_weight += weight

                    if total_weight > 0:
                        total_mod /= total_weight
                        # Apply modification to last token position
                        mod_tensor = torch.tensor(
                            total_mod * self.mod_scale,
                            dtype=output.dtype, device=output.device
                        )
                        output = output.clone()
                        output[0, -1, :] += mod_tensor

            return output

        return hook_fn

    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> str:
        """Generate text with knowledge-augmented inference."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                **kwargs
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def get_activations(self, text: str) -> dict[int, np.ndarray]:
        """
        Get MLP output activations for each layer.
        Used by the learning module to create knowledge entries.
        """
        activations = {}

        def make_capture_hook(layer_idx):
            def hook_fn(module, input, output):
                activations[layer_idx] = output[0, -1, :].cpu().float().numpy()
            return hook_fn

        # Temporarily install capture hooks
        temp_hooks = []
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            h = layer.mlp.register_forward_hook(make_capture_hook(layer_idx))
            temp_hooks.append(h)

        # Forward pass
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model(**inputs)

        # Remove capture hooks
        for h in temp_hooks:
            h.remove()

        return activations

    def remove_hooks(self):
        """Clean up hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
