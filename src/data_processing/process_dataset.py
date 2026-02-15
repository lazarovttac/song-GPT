"""
Consolidate and clean the scraped lyrics dataset.

Reads all batch_*.zip from data/compressed/**/,
applies text transformations per song, and writes
large consolidated .txt blocks (~400k songs each)
to data/uncompressed/.

Intermediate per-batch outputs (~10k songs each) are
written to data/uncompressed/batches/ and removed
after merging, so the final dataset only contains
the large block files plus metadata.csv.

Transformations:
  1. Normalize key/tuning metadata into tags
  2. Fix chords that split words (move to word start)
  3. Wrap bare chord lines in brackets
  4. Strip all leading whitespace from lyrics
  5. Add <|song_end|> delimiter after each song
  6. Generate metadata CSV

Usage:
    python process_dataset.py
"""

import csv
import re
import time
import zipfile
from multiprocessing import Pool, cpu_count
from pathlib import Path

# --- Configuration ---
COMPRESSED_DIR = Path("data") / "compressed"
OUTPUT_DIR = Path("data") / "uncompressed"
BATCH_DIR = OUTPUT_DIR / "batches"
METADATA_CSV = OUTPUT_DIR / "metadata.csv"

# Target number of songs per final merged block file
TARGET_SONGS_PER_BLOCK = 400_000

# --- Compiled regex patterns ---

# Latin letters (for chord boundary detection)
_L = r"[a-zA-Z\u00C0-\u024F]"
# Mid-word chord: one or more letters, then [Chord], then a letter ahead
_MID_CHORD_RE = re.compile(rf"({_L}+)(\[[^\]]+\])(?={_L})")

# key: / tonalidad: on one line, note on the next
_KEY_RE = re.compile(
    r"^ *(?:key|tonalidad) *: *\n *([A-G][#b]?m?)\b[^\n]*",
    re.MULTILINE | re.IGNORECASE,
)
# Tuning: NOTES on a single line
_TUNING_RE = re.compile(r"^ *Tuning *: *(.+)$", re.MULTILINE | re.IGNORECASE)

# Split concatenated songs: each starts with <|lang:
_SONG_SPLIT_RE = re.compile(r"(?=<\|lang:)")

# Metadata extraction (compiled once)
_LANG_RE = re.compile(r"<\|lang:([^|]+)\|>")
_TITLE_RE = re.compile(r"<\|([^|]+? - [^|]+)\|>")
_HAS_CHORD_RE = re.compile(r"\[[A-G]")

# Bare chord token: A, Am, A7, Ab, A#m7, Dsus4, G/B, etc.
_BARE_CHORD_TOKEN_RE = re.compile(
    r"^[A-G][#b]?(?:m|maj|dim|aug|sus|add)?[0-9]*(?:/[A-G][#b]?)?$",
    re.IGNORECASE,
)
# For replacement within a confirmed chord-only line
_BARE_CHORD_REPL_RE = re.compile(
    r"\b([A-G][#b]?(?:m|maj|dim|aug|sus|add)?[0-9]*(?:/[A-G][#b]?)?)\b"
)


# ── Encoding fix (mojibake) ───────────────────────────────────
# "CÃª" → "Cê": UTF-8 bytes were decoded as Latin-1. Undo by re-encoding
# as Latin-1 and decoding as UTF-8.
# Alternative: pip install ftfy → ftfy.fix_text(s) for more encoding fixes.


def fix_mojibake(text: str) -> str:
    """Fix UTF-8-decoded-as-Latin-1 mojibake. Iterates to handle double/triple encoding."""
    if not text or "\x00" in text:
        return text

    # Apply fix until stable (handles multiple levels of mojibake)
    prev = None
    max_iterations = 3
    iteration = 0

    while prev != text and iteration < max_iterations:
        prev = text
        try:
            text = text.encode("latin-1").decode("utf-8")
            iteration += 1
        except (UnicodeDecodeError, UnicodeEncodeError):
            break

    return text


# ── Per-song transformations ────────────────────────────────


def fix_chord_boundaries(text: str) -> str:
    """Move chords that split a word to the beginning of that word.

    Example: ``Toca[G]r`` → ``[G]Tocar``

    Iterates until stable to handle multiple chords in one word.
    """
    prev = None
    while prev != text:
        prev = text
        text = _MID_CHORD_RE.sub(lambda m: m.group(2) + m.group(1), text)
    return text


_XN_RE = re.compile(r"^x\d+$")


def _is_chord_only_line(line: str) -> bool:
    """Fast check: is this line composed only of bare chords and xN?"""
    tokens = line.split()
    if not tokens:
        return False
    has_chord = False
    for t in tokens:
        if _BARE_CHORD_TOKEN_RE.match(t):
            has_chord = True
        elif _XN_RE.match(t):
            pass  # e.g. x2
        else:
            return False
    return has_chord


def wrap_bare_chords(line: str) -> str:
    """Wrap bare chords in brackets if the line is a chord-only line."""
    if not _is_chord_only_line(line):
        return line
    return _BARE_CHORD_REPL_RE.sub(r"[\1]", line)


def normalize_song(raw: str) -> tuple[str, dict]:
    """Apply all transformations to a single song string.

    Returns: (processed_text, metadata_dict)
    """
    raw = raw.strip()
    if not raw:
        return "", {}
    raw = fix_mojibake(raw)

    # --- Extract header metadata ---
    metadata = {
        "artist_name": "",
        "song_name": "",
        "language": "",
        "key": "",
        "has_chords": False,
    }

    # Extract language: <|lang:XX|>
    lang_match = _LANG_RE.search(raw)
    if lang_match:
        metadata["language"] = lang_match.group(1)

    # Extract artist and song: <|Artist - Song|>
    title_match = _TITLE_RE.search(raw)
    if title_match:
        parts = title_match.group(1).split(" - ", 1)
        metadata["artist_name"] = parts[0].strip()
        metadata["song_name"] = parts[1].strip()

    marker = "<|song_start|>"
    idx = raw.find(marker)
    if idx == -1:
        # Malformed song — keep as-is with end tag
        result = raw + "\n<|song_end|>"
        metadata["has_chords"] = bool(_HAS_CHORD_RE.search(result))
        return result, metadata

    header = raw[: idx + len(marker)]
    body = raw[idx + len(marker) :]

    # --- Extract metadata tags from the body preamble ---
    tags = []

    m = _KEY_RE.search(body)
    if m:
        key_val = m.group(1)
        metadata["key"] = key_val
        tags.append(f"<|key:{key_val}|>")
        body = body[: m.start()] + body[m.end() :]

    m = _TUNING_RE.search(body)
    if m:
        tags.append(f"<|tuning:{m.group(1).strip()}|>")
        body = body[: m.start()] + body[m.end() :]

    # Strip leading blank lines left after metadata removal
    body = body.lstrip("\n")

    # --- Fix chord word boundaries ---
    body = fix_chord_boundaries(body)

    # --- Process each line: strip leading whitespace, wrap bare chords ---
    lines = body.split("\n")
    processed_lines = []
    for line in lines:
        # Strip leading whitespace
        line = line.lstrip()
        # Wrap bare chord lines
        line = wrap_bare_chords(line)
        processed_lines.append(line)
    body = "\n".join(processed_lines)

    # --- Rebuild ---
    tag_block = "\n".join(tags)
    if tag_block:
        result = f"{header}\n{tag_block}\n{body}"
    else:
        result = f"{header}\n{body}"

    result = result.rstrip() + "\n<|song_end|>"

    # Check if song has chords
    metadata["has_chords"] = bool(_HAS_CHORD_RE.search(result))

    return result, metadata


# ── ZIP processing ──────────────────────────────────────────


def _count_star_lines(text: str) -> int:
    """Count lines that contain * alongside what looks like chord names."""
    count = 0
    for line in text.split("\n"):
        if "*" in line and not line.lstrip().startswith("<|"):
            tokens = line.split()
            if tokens and any(_BARE_CHORD_TOKEN_RE.match(t) for t in tokens):
                count += 1
    return count


def process_zip(zip_path: Path) -> tuple[list[str], list[dict], int]:
    """Read every .txt inside a ZIP, split into songs, normalize each.

    Returns: (song_texts, metadata_dicts, star_line_count)
    """
    songs: list[str] = []
    metadata_list: list[dict] = []
    star_lines = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in sorted(zf.namelist()):
                if not name.endswith(".txt"):
                    continue
                raw = zf.read(name).decode("utf-8", errors="replace")
                star_lines += _count_star_lines(raw)
                parts = _SONG_SPLIT_RE.split(raw)
                for part in parts:
                    part = part.strip()
                    if part and part.startswith("<|lang:"):
                        processed, meta = normalize_song(part)
                        if processed:
                            songs.append(processed)
                            metadata_list.append(meta)
    except Exception as e:
        print(f"  ✗ Error reading {zip_path.name}: {e}")

    return songs, metadata_list, star_lines


def _worker(args: tuple) -> tuple[str, int, list[dict], int]:
    """Multiprocessing worker: process one ZIP → one batch .txt + metadata."""
    zip_path, output_path = args
    zip_path, output_path = Path(zip_path), Path(output_path)

    songs, metadata_list, star_lines = process_zip(zip_path)
    if not songs:
        return (output_path.name, 0, [], 0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(songs))

    return (output_path.name, len(songs), metadata_list, star_lines)


# ── Main ────────────────────────────────────────────────────


def main():
    t0 = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Discover batch ZIPs
    t_step = time.perf_counter()
    zips = sorted(COMPRESSED_DIR.glob("**/batch_*.zip"))
    if not zips:
        print(f"No ZIP files found in {COMPRESSED_DIR}")
        return
    t_discovery = time.perf_counter() - t_step
    num_batches = len(zips)
    print(f"[{t_discovery:.1f}s] Found {num_batches:,} ZIP files in {COMPRESSED_DIR}")

    # Build task list (per-ZIP batch outputs written under BATCH_DIR)
    tasks = []
    for zp in zips:
        folder = zp.parent.name
        stem = zp.stem
        out_name = f"{folder}_{stem}.txt"
        tasks.append((str(zp), str(BATCH_DIR / out_name)))

    workers = min(cpu_count(), len(tasks))
    # ~10k songs per ZIP for progress display
    est_songs = num_batches * 10_000
    print(f"[--] Using {workers} workers. Estimated songs: ~{est_songs:,}\n")

    # Step 2: Process all batches (read ZIPs, normalize, write per-ZIP batch .txt)
    t_process_start = time.perf_counter()
    total_songs = 0
    total_files = 0
    total_star_lines = 0
    all_metadata = []
    completed = 0

    # Collect (out_name, count) so we can open batch files later in a stable order
    batch_results: list[tuple[str, int]] = []

    with Pool(workers) as pool:
        for out_name, count, metadata_list, star_lines in pool.imap_unordered(
            _worker, tasks
        ):
            total_songs += count
            total_files += 1
            total_star_lines += star_lines
            all_metadata.extend(metadata_list)
            completed += 1
            batch_results.append((out_name, count))
            pct = 100 * completed / num_batches if num_batches else 0
            print(
                f"  [{completed:,}/{num_batches:,}] {out_name}: {count:,} songs "
                f"| total: {total_songs:,} ({pct:.1f}%)"
            )

    t_process = time.perf_counter() - t_process_start
    rate = total_songs / t_process if t_process > 0 else 0
    print(
        f"\n[{t_process:.1f}s] Processed {total_songs:,} songs (~{rate:,.0f} songs/s)"
    )

    # Step 3: Merge per-ZIP batch files into large block files
    if total_songs > 0 and batch_results:
        print(
            f"\nMerging batch files from {BATCH_DIR} into "
            f"blocks of ~{TARGET_SONGS_PER_BLOCK:,} songs"
        )
        # Ensure deterministic order: sort by batch filename
        batch_results.sort(key=lambda x: x[0])

        block_index = 1
        songs_in_block = 0
        current_block_path = OUTPUT_DIR / f"block_{block_index:04}.txt"
        current_block_file = current_block_path.open("w", encoding="utf-8")

        for out_name, count in batch_results:
            batch_path = BATCH_DIR / out_name
            if not batch_path.is_file():
                continue

            # Stream copy batch content into current block file
            with batch_path.open("r", encoding="utf-8") as bf:
                for chunk in iter(lambda: bf.read(1024 * 1024), ""):
                    current_block_file.write(chunk)

            songs_in_block += count

            # Remove the small batch file once merged
            try:
                batch_path.unlink()
            except OSError:
                pass

            # When we reach or exceed the target, start a new block file
            if songs_in_block >= TARGET_SONGS_PER_BLOCK:
                current_block_file.close()
                block_index += 1
                songs_in_block = 0
                current_block_path = OUTPUT_DIR / f"block_{block_index:04}.txt"
                current_block_file = current_block_path.open("w", encoding="utf-8")

        # Close the last block file if it's still open
        if not current_block_file.closed:
            current_block_file.close()

        # Try to remove the now-empty batches directory
        try:
            BATCH_DIR.rmdir()
        except OSError:
            # Directory not empty or cannot be removed – safe to ignore
            pass

    # Step 4: Write metadata CSV
    t_csv_start = time.perf_counter()
    print(f"\nWriting metadata CSV: {METADATA_CSV}")
    with open(METADATA_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["artist_name", "song_name", "language", "key", "has_chords"],
        )
        writer.writeheader()
        writer.writerows(all_metadata)
    t_csv = time.perf_counter() - t_csv_start
    print(f"[{t_csv:.1f}s] Wrote {len(all_metadata):,} rows to CSV")

    # Summary
    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 60)
    print("Total time: {:.1f}s".format(elapsed))
    print("  discovery:  {:.1f}s".format(t_discovery))
    print(
        "  process:    {:.1f}s  ({} batches, {} songs)".format(
            t_process, total_files, total_songs
        )
    )
    print("  CSV write:  {:.1f}s".format(t_csv))
    print("=" * 60)
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Metadata: {METADATA_CSV}")
    print(f"  Lines with * near chords (ignored): {total_star_lines:,}")


if __name__ == "__main__":
    main()
