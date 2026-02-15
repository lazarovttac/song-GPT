import os
import numpy as np
import tiktoken
from tqdm import tqdm
import glob


class PrepareData:
    def __init__(self, config):
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def prepare_data(self, force_compute: bool = False):
        """Prepare training data. If force_compute is True, rebuild .bin files from lyrics.
        Otherwise, use existing .bin files (they must already exist)."""
        if not force_compute:
            self._verify_bin_files_exist()
            print("Using existing .bin files. Set force_compute=True to rebuild.")
            return

        print("Starting data preparation...")

        # Load all lyrics from uncompressed files
        lyrics = self._load_uncompressed_lyrics()

        if not lyrics:
            raise ValueError("No lyrics found in uncompressed directory")

        # Tokenize all texts
        all_tokens = self._tokenize_texts(lyrics)

        if len(all_tokens) == 0:
            raise ValueError("No tokens generated")

        print(f"Total tokens: {len(all_tokens)}")

        # Split into train/val (90/10)
        n = int(0.9 * len(all_tokens))
        train_tokens = all_tokens[:n]
        val_tokens = all_tokens[n:]

        # Save to binary files
        self._save_tokens_to_bin(train_tokens, self.config.train_bin_path)
        self._save_tokens_to_bin(val_tokens, self.config.val_bin_path)

        print("Data preparation completed!")
        print(f"Train tokens: {len(train_tokens)}")
        print(f"Val tokens: {len(val_tokens)}")
        print(f"Vocabulary size: {self.config.vocab_size}")

    def _load_uncompressed_lyrics(self) -> list[str]:
        """Load all lyrics from uncompressed text files."""
        all_lyrics = []

        # Find all text files recursively
        txt_files = glob.glob(
            os.path.join(self.config.data_dir, "**/*.txt"), recursive=True
        )

        print(f"Found {len(txt_files)} text files to process")

        for txt_path in tqdm(txt_files, desc="Processing text files"):
            try:
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as file:
                    content = file.read()
                    if content.strip():  # Only add non-empty content
                        all_lyrics.append(content)
            except Exception as e:
                print(f"Error reading {txt_path}: {e}")
                continue

        print(f"Loaded {len(all_lyrics)} lyrics files")
        return all_lyrics

    def _tokenize_texts(self, texts: list[str]) -> np.ndarray:
        """Tokenize a list of texts and return concatenated tokens."""
        all_tokens = []

        for text in tqdm(texts, desc="Tokenizing texts"):
            try:
                tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
                all_tokens.extend(tokens)
            except Exception as e:
                print(f"Error tokenizing text: {e}")
                continue

        return np.array(all_tokens, dtype=np.uint32)

    def _save_tokens_to_bin(self, tokens: np.ndarray, filepath: str):
        """Save tokens to a binary file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            tokens.tofile(f)
        print(f"Saved {len(tokens)} tokens to {filepath}")

    def _verify_bin_files_exist(self) -> None:
        """Raise if train or val .bin files are missing."""
        for path in (self.config.train_bin_path, self.config.val_bin_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Bin file not found: {path}. Run prepare_data(force_compute=True) first."
                )
