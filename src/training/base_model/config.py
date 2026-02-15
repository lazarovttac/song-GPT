import os
import torch
import tiktoken


class Config:
    # Training
    seed = 1337
    # With a ~100k vocab and long context windows, large batches quickly OOM on GPU.
    batch_size = 8
    gradient_accumulation_steps = 8  # Effective batch size = 64
    block_size = 1024
    max_iters = 25000  # Increased from 5000 for better convergence
    eval_interval = 500
    learning_rate = 6e-4  # Increased from 3e-4
    min_lr = 6e-5  # For cosine decay
    eval_iters = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = True

    # Optimization
    warmup_iters = 2000  # Learning rate warmup
    lr_decay_iters = 25000  # When to stop decay (usually = max_iters)
    grad_clip = 1.0  # Gradient clipping threshold
    weight_decay = 1e-1  # AdamW weight decay
    early_stopping_patience = (
        4  # Stop after N evals without val loss improvement; 0 = disabled
    )
    beta1 = 0.9  # AdamW beta1
    beta2 = 0.95  # AdamW beta2

    print("Using device:", device)

    # Model Architecture - Increased capacity
    n_embd = 512  # Increased from 384
    n_head = 8  # Increased from 6
    n_layer = 8  # Increased from 6
    dropout = 0.2
    vocab_size = tiktoken.get_encoding("cl100k_base").n_vocab

    # Project paths (resolved from this file, not current working directory)
    _project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    # Data paths
    # data_dir = os.path.join(_project_root, "data", "lyrics", "uncompressed")
    # train_bin_path = os.path.join(_project_root, "data", "lyrics", "bin", "train.bin")
    # val_bin_path = os.path.join(_project_root, "data", "lyrics", "bin", "val.bin")

    data_dir = os.path.join(_project_root, "data", "lyrics", "en")
    train_bin_path = os.path.join(
        _project_root, "data", "lyrics", "bin", "en", "train.bin"
    )
    val_bin_path = os.path.join(_project_root, "data", "lyrics", "bin", "en", "val.bin")

    checkpoint_path = os.path.join(
        _project_root, "models", "checkpoints", "en", "ckpt.pt"
    )

    def to_dict(self):
        return {
            k: v
            for k, v in self.__class__.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
