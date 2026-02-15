# Song-GPT

Song-GPT is a decoder-only Transformer (GPT-style) trained from scratch with PyTorch to generate English song lyrics.

## What is in this repository

- `src/data_collection/`: scripts to collect lyrics data from Cifra Club
- `src/data_processing/`: scripts to filter and prepare text data
- `src/training/base_model/`: model config, dataset prep, training, and trainer code
- `src/hf-upload.py`: helper script to upload model and model card to Hugging Face

## Important notice

This project is for educational and research purposes. Respect:

- Website terms of service for any scraping target
- Copyright and licensing rules for lyrics and compositions
- Responsible rate limits during data collection

## Requirements

- Python 3.10+ recommended
- `pip`
- Optional GPU for faster training

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Quick start

### 1) (Optional) Collect data

Run any of the collection phases from the project root:

```powershell
python src/data_collection/scrape_artists.py
python src/data_collection/scrape_songs.py
python src/data_collection/scrape_lyrics.py
```

More details are in `src/data_collection/README.md`.

### 2) Filter lyrics by language

```powershell
python src/data_processing/filter_lyrics.py
```

By default it filters English songs (`en`) and writes to `data/lyrics/en/all_en.txt`.

### 3) Train base model

```powershell
python src/training/base_model/train.py
```

The training script prepares binary datasets, trains, writes a checkpoint, and runs a short generation smoke test.

## Optional: run in Google Colab

If you run `train.py` in Colab and want automatic Drive mounting/path switch, set:

```powershell
$env:SONG_GPT_COLAB_PROJECT_PATH = "/content/drive/MyDrive/song-GPT/src/training/base_model"
python src/training/base_model/train.py
```

If `SONG_GPT_COLAB_PROJECT_PATH` is not set, the script runs without Colab-specific side effects.

## Model details

- **Architecture**: GPT (decoder-only Transformer)
- **Tokenization**: `cl100k_base` (`tiktoken`)
- **Vocabulary size**: 100277
- **Embedding dimension (`n_embd`)**: 512
- **Context length (`block_size`)**: 1024
- **Transformer layers (`n_layer`)**: 8
- **Attention heads (`n_head`)**: 8
- **Dropout**: 0.2
- **Parameters**: 128.52M

## Prompt format

Special tokens:

- `<|lang:en|>` language marker
- `<|Artist - Song Title|>` optional metadata
- `<|song_start|>` lyrics start
- `<|song_end|>` lyrics end

Example:

```python
test_prompt = "<|lang:en|>\n<|Taylor Swift - Rainy Days|>\n<|song_start|>"
```

## Hugging Face upload

The upload helper expects your Hugging Face auth to be configured locally (for example with `huggingface-cli login`) and prompts for your username:

```powershell
python src/hf-upload.py
```

## Repository hygiene

- Large training artifacts and datasets are ignored by `.gitignore` (`data/`, `models/`, checkpoints, and binaries).
- Keep notebook outputs cleared before committing to avoid leaking local paths and to reduce repository size.
