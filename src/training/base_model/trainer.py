import math
import os
import torch
from contextlib import nullcontext
from model import GPTLanguageModel
from data_loader import DataLoader


def _checkpoint_path_for_iter(checkpoint_dir, iter):
    """Path for a checkpoint at a given iteration (unique name per iter)."""
    return os.path.join(checkpoint_dir, f"ckpt_iter_{iter}.pt")


class Trainer:
    def __init__(self, config):
        self.config = config

        # Initialize model with config
        self.model = GPTLanguageModel(config).to(config.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(
                getattr(config, "beta1", 0.9),
                getattr(config, "beta2", 0.95),
            ),
            weight_decay=getattr(config, "weight_decay", 0.1),
        )
        self.use_amp = bool(
            getattr(config, "use_amp", False) and str(config.device).startswith("cuda")
        )
        # Prefer bf16 on supported GPUs; fallback to fp16 otherwise.
        self.amp_dtype = (
            torch.bfloat16
            if self.use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Initialize data loader
        self.data_loader = DataLoader(config)

        self.checkpoint_dir = os.path.dirname(
            getattr(config, "checkpoint_path", "ckpt.pt")
        )
        if self.checkpoint_dir and not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(
            f"Model created with {param_count:.2f}M parameters. Using device: {config.device}"
        )
        print(
            f"Effective batch size: {config.batch_size * getattr(config, 'gradient_accumulation_steps', 1)}"
        )

    def _autocast_context(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return nullcontext()

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()

        for split in ["train", "val"]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.data_loader.get_batch(split)
                with self._autocast_context():
                    _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()

        self.model.train()
        return out

    def _get_lr(self, iter):
        """Learning rate schedule with warmup and cosine decay."""
        warmup_iters = getattr(self.config, "warmup_iters", 0)
        lr_decay_iters = getattr(self.config, "lr_decay_iters", self.config.max_iters)
        min_lr = getattr(self.config, "min_lr", self.config.learning_rate * 0.1)

        # Linear warmup
        if iter < warmup_iters:
            return self.config.learning_rate * iter / warmup_iters

        # Cosine decay after warmup
        if iter > lr_decay_iters:
            return min_lr

        decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (self.config.learning_rate - min_lr)

    def save_checkpoint(self, iter, val_loss, is_best=False):
        """Save checkpoint with a unique name for this iteration."""
        path = _checkpoint_path_for_iter(self.checkpoint_dir, iter)
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iter": iter,
            "val_loss": val_loss,
            "config_dict": self.config.to_dict(),
        }
        if self.use_amp:
            state["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(state, path)
        suffix = " (best)" if is_best else ""
        print(f"Checkpoint saved: {path}{suffix}")

    @classmethod
    def load_checkpoint(cls, path, config, device=None, load_optimizer=True):
        """
        Load a checkpoint from disk.

        Args:
            path: Path to the .pt checkpoint file.
            config: Config object used to build the model (must match checkpoint architecture).
            device: Device to load onto; defaults to config.device.
            load_optimizer: If True, also return optimizer and scaler state for resuming.

        Returns:
            model: Loaded GPTLanguageModel.
            state: Dict with keys iter, val_loss, optimizer_state_dict (if load_optimizer),
                   scaler_state_dict (if amp and load_optimizer). Use these to resume training.
        """
        device = device or config.device
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = GPTLanguageModel(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        return model

    def resume_from_checkpoint(self, path, load_optimizer=True):
        """
        Load a checkpoint into this trainer (model, and optionally optimizer/scaler).
        Returns the iteration stored in the checkpoint; use train(start_iter=iter + 1) to continue.
        """
        model, state = self.load_checkpoint(
            path, self.config, device=self.config.device, load_optimizer=load_optimizer
        )
        self.model.load_state_dict(model.state_dict())
        if load_optimizer and state.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if load_optimizer and state.get("scaler_state_dict") and self.use_amp:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        return state["iter"]

    def train(self, start_iter=0):
        xb, yb = self.data_loader.get_batch("train")
        print("Inputs shape:", xb.shape)
        print("Targets shape:", yb.shape)

        grad_accum_steps = getattr(self.config, "gradient_accumulation_steps", 1)
        grad_clip = getattr(self.config, "grad_clip", 0.0)
        early_stopping_patience = getattr(self.config, "early_stopping_patience", 0)
        best_val_loss = float("inf")
        patience_counter = 0

        if start_iter > 0:
            losses = self.estimate_loss()
            best_val_loss = losses["val"].item()
            print(
                f"Resuming from iter {start_iter}, current val loss: {best_val_loss:.4f}"
            )

        for iter in range(start_iter, self.config.max_iters):
            # Update learning rate
            lr = self._get_lr(iter)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Evaluate, log, save checkpoint, and check early stopping
            if (
                iter % self.config.eval_interval == 0
                or iter == self.config.max_iters - 1
            ):
                losses = self.estimate_loss()
                val_loss = losses["val"].item()
                print(
                    f"{iter}: train loss {losses['train']:.4f}, val loss {val_loss:.4f}, lr {lr:.2e}"
                )

                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                self.save_checkpoint(iter, val_loss, is_best=is_best)

                if (
                    early_stopping_patience > 0
                    and patience_counter >= early_stopping_patience
                ):
                    print(
                        f"Early stopping at iter {iter}: no improvement for {early_stopping_patience} eval(s)."
                    )
                    break

            # Gradient accumulation loop
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(grad_accum_steps):
                xb, yb = self.data_loader.get_batch("train")
                with self._autocast_context():
                    _, loss = self.model(xb, yb)
                    loss = loss / grad_accum_steps  # Scale loss for accumulation

                accum_loss += loss.item()
                self.scaler.scale(loss).backward()

            # Gradient clipping
            if grad_clip > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

        print("\n--- Done! ---")
