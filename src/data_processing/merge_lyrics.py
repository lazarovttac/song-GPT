from pathlib import Path
from math import ceil


def merge_text_files(input_dir: Path, output_dir: Path, num_outputs: int = 10) -> None:
    if num_outputs <= 0:
        raise ValueError("num_outputs must be a positive integer")

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = ceil(len(txt_files) / num_outputs)

    for i in range(num_outputs):
        start = i * chunk_size
        end = start + chunk_size
        group = txt_files[start:end]
        if not group:
            break

        out_path = output_dir / f"merged_{i + 1:02}.txt"
        with out_path.open("w", encoding="utf-8") as out_f:
            for src in group:
                with src.open("r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    input_dir = project_root / "data" / "lyrics" / "uncompressed"
    output_dir = project_root / "data" / "lyrics" / "merged"

    merge_text_files(input_dir=input_dir, output_dir=output_dir, num_outputs=10)
