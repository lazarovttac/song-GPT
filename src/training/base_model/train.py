import os
import sys
import torch
import tiktoken

from config import Config
from prepare_data import PrepareData
from trainer import Trainer


def configure_colab_path() -> None:
    """Optionally mount Drive and cd when running inside Google Colab."""
    if "COLAB_GPU" not in os.environ:
        return

    colab_project_path = os.getenv("SONG_GPT_COLAB_PROJECT_PATH", "").strip()
    if not colab_project_path:
        return

    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=True)
    except ImportError:
        return

    if os.path.exists(colab_project_path):
        os.chdir(colab_project_path)
        if colab_project_path not in sys.path:
            sys.path.append(colab_project_path)
        print(f"Configured Colab path: {os.getcwd()}")
    else:
        raise FileNotFoundError(
            "SONG_GPT_COLAB_PROJECT_PATH is set but does not exist: "
            f"{colab_project_path}"
        )


def print_runtime_diagnostics() -> None:
    """Print runtime and CUDA details to help debugging setup issues."""
    print("Python:", sys.executable)
    print("Torch:", torch.__file__)
    print("Torch CUDA version:", torch.version.cuda or "None (CPU-only build)")

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        return

    print(
        "CUDA not available. On Windows, PyTorch CUDA wheels require Python 3.8-3.12 "
        f"(you have {sys.version.split()[0]}). "
        "Use a venv with Python 3.12 and run:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
    )


def run_training(config: Config) -> Trainer:
    prepare_data = PrepareData(config)
    prepare_data.prepare_data(force_compute=False)

    trainer = Trainer(config)
    print("Starting training...")
    trainer.train()
    return trainer


def save_checkpoint(trainer: Trainer, config: Config) -> None:
    checkpoint_dir = os.path.dirname(config.checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "config": config.to_dict(),
            "iter": config.max_iters,
        },
        config.checkpoint_path,
    )
    print(f"Training completed. Model saved to {config.checkpoint_path}")


def run_smoke_generation(trainer: Trainer, config: Config) -> None:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    test_prompt = "<|lang:en|>\n<|Taylor Swift - Rainy Days|>\n<|song_start|>"
    print("Test prompt:")
    print(test_prompt)
    print("\n" + "=" * 50 + "\n")

    input_tokens = (
        torch.tensor(tokenizer.encode(test_prompt), dtype=torch.long)
        .unsqueeze(0)
        .to(config.device)
    )

    trainer.model.eval()
    print("Generating lyrics...")
    with torch.no_grad():
        generated_tokens = trainer.model.generate(input_tokens, max_new_tokens=200)

    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    print("Generated output:")
    print(generated_text)


def main() -> None:
    configure_colab_path()
    print_runtime_diagnostics()

    config = Config()
    trainer = run_training(config)
    save_checkpoint(trainer, config)
    run_smoke_generation(trainer, config)


if __name__ == "__main__":
    main()
