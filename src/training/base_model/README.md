
---
license: mit
language: en
pipeline_tag: text-generation
---

# Song-GPT: A GPT for English Song Lyrics Generation

A Generative Pre-trained Transformer (GPT) trained from scratch with PyTorch to generate song lyrics in English.

## Model Details

- **Architecture**: GPT (Decoder-only Transformer)
- **Tokenization**: cl100k_base (tiktoken)
- **Vocabulary Size**: 100277
- **Embedding Dimension (`n_embd`)**: 512
- **Context Length (`block_size`)**: 1024
- **Number of Transformer Layers (`n_layer`)**: 8
- **Number of Attention Heads (`n_head`)**: 8
- **Dropout Rate**: 0.2
- **Number of Parameters**: 128.52M

## Training Data

The model was trained on a corpus of English song lyrics. Each song in the dataset is structured with special tokens to mark the start and end of lyrics, as well as artist and song metadata.

## Special Tokens

- `<|lang:en|>` — Language marker for English
- `<|Artist - Song Title|>` — Optional artist and song name
- `<|song_start|>` — Marks the beginning of lyrics
- `<|song_end|>` — Marks the end of lyrics

## Usage

You can generate lyrics by providing a prompt. Optionally specify the artist and song name:

```python
# Basic prompt (language only)
test_prompt = "<|lang:en|>"

# With artist and song name
test_prompt = "<|lang:en|>\n<|Taylor Swift - Rainy Days|>\n<|song_start|>"
```

To use this model, load the `state_dict` into an instance of the architecture defined in the training code. The model requires `tiktoken` with encoding `cl100k_base`.
