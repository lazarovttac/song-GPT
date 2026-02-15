import pathlib
import re


def process_lyrics(target_lang="en", base_dir=None):
    if base_dir is None:
        base_dir = pathlib.Path(__file__).resolve().parents[2] / "data" / "lyrics"
    else:
        base_dir = pathlib.Path(base_dir)

    input_dir = base_dir / "uncompressed"
    output_dir = base_dir / target_lang
    output_file = output_dir / f"all_{target_lang}.txt"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    lang_tag = f"<|lang:{target_lang}|>"
    # Split by song_end to isolate individual songs while keeping the delimiter
    song_pattern = re.compile(r"(<\|lang:.*?<\|song_end\|>)", re.DOTALL)

    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_path in input_dir.glob("*.txt"):
            content = file_path.read_text(encoding="utf-8")
            songs = song_pattern.findall(content)

            for song in songs:
                if lang_tag in song:
                    outfile.write(song.strip() + "\n\n")


if __name__ == "__main__":
    process_lyrics("en")
