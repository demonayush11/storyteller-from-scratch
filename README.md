# Storyteller — GPT-124M from Scratch

A full implementation of a GPT-2-style language model trained on the **TinyStories** dataset, built from the ground up in PyTorch. The model learns to generate short, coherent children's stories. This notebook follows the approach from *Build a Large Language Model from Scratch* by Sebastian Raschka.

---

## Overview

The notebook walks through every layer of building and training a GPT model:

1. **Tokenization** — Custom tokenizers (V1 and V2) built from scratch, followed by integration with OpenAI's `tiktoken` (GPT-2 BPE encoding).
2. **Dataset & DataLoader** — A sliding-window `GPTDatasetV1` that chunks the TinyStories corpus into overlapping input/target sequences.
3. **Embeddings** — Token embeddings + positional embeddings combined as model input.
4. **Self-Attention** — Manual dot-product attention, scaled softmax, and a full `MultiHeadAttention` module with causal masking.
5. **Transformer Block** — `LayerNorm`, `GELU` activation, `FeedForward` (expand → activate → contract), residual/shortcut connections.
6. **GPT Model** — Full `GPTModel` assembling all components into a 124M-parameter autoregressive LM.
7. **Training** — Cross-entropy loss, AdamW optimizer, train/validation split, loss tracking and plotting.
8. **Text Generation** — Greedy decoding (`generate_text_simple`) and advanced sampling (`generate`) with temperature scaling and top-k filtering.
9. **Checkpointing** — Save/load model weights and optimizer state.
10. **Pretrained Weights** — Download GPT-2 (355M) weights via TensorFlow and load them into the custom architecture.

---

## Model Configuration

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # GPT-2 BPE vocabulary
    "context_length": 256,    # Shortened from original 1024 for faster training
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

---

## Dataset

**[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)** — a synthetic dataset of short stories written with simple vocabulary, ideal for training small LMs from scratch.

- Training uses the first **10,000 stories** (~10M characters)
- 90/10 train/validation split

---

## Requirements

```bash
pip install torch datasets tiktoken matplotlib tensorflow tqdm
```

Runs on Google Colab (GPU recommended). The notebook includes device detection for CUDA/MPS/CPU.

---

## Usage

### Train the model

Run all cells sequentially. Training for 10 epochs with AdamW (`lr=0.0004`) prints loss every 5 steps and generates a sample after each epoch.

### Generate text

```python
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
print(token_ids_to_text(token_ids, tokenizer))
```

### Save & load

```python
# Save
torch.save(model.state_dict(), "storyteller_gpt.pth")

# Load
model.load_state_dict(torch.load("storyteller_gpt.pth", map_location=device))
```

---

## Key Components

| Module | Description |
|---|---|
| `SimpleTokenizerV1/V2` | Regex-based tokenizers with `<\|unk\|>` and `<\|endoftext\|>` support |
| `GPTDatasetV1` | Sliding-window PyTorch `Dataset` |
| `MultiHeadAttention` | Causal multi-head self-attention with dropout |
| `TransformerBlock` | Pre-norm transformer block with residual connections |
| `GPTModel` | Full autoregressive GPT model |
| `generate` | Sampling with temperature + top-k filtering |
| `train_model_simple` | Training loop with periodic evaluation and sample generation |

---

## Outputs

- `storyteller_gpt.pth` — trained model weights
- `model_and_optimizer.pth` — model + optimizer checkpoint
- `loss-plot.pdf` — training vs. validation loss curve
- `temperature-plot.pdf` — token probability distribution at different temperatures
