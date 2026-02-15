"""
Phase 3: Scrape lyrics content from songs_db.csv and save as batched ZIP files to Drive.
Run in Colab. Uses requests (much faster than browser).

Batching strategy:
- Download 100 song lyrics and concatenate them into a single .txt file (cached in content/)
- Once 100 of those .txt files have been cached, create a .zip file and save to Drive as batch_001.zip
- Upload process runs in background to avoid blocking downloads

Install in Colab before running:
  !pip install requests beautifulsoup4 langdetect
"""

import csv
import json
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from common import (
    BASE_URL,
    DEFAULT_DELAY,
    DEFAULT_RETRIES,
    setup_drive,
)
from process_lyrics import LyricsProcessor

# --- Parameters (edit for Colab) ---
songs_db_csv = "data/songs_db.csv"  # Path to songs_db.csv
delay = DEFAULT_DELAY
retries = DEFAULT_RETRIES
workers = 20  # number of parallel requests (can be higher without browser)
lang_model_path = None  # Path to FastText model (.bin) for language detection
songs_per_txt = 100  # Number of songs to concatenate into one .txt file
txt_files_per_zip = 100  # Number of .txt files per ZIP batch


def load_songs_from_csv(csv_path: str) -> list:
    """Read songs_db.csv and return list of (artist_id, artist_name, song_url, song_name)."""
    if not Path(csv_path).exists():
        return []
    songs = []
    try:
        with open(csv_path, "r", encoding="utf-8") as fp:
            reader = csv.reader(fp)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 4:
                    artist_id, artist_name, song_url, song_name = (
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                    )
                    # Build full URL if song_url is just a path/ID
                    if not song_url.startswith("http"):
                        song_url = f"{BASE_URL}/{artist_id}/{song_url}/"
                    songs.append((artist_id, artist_name, song_url, song_name))
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return songs


def run(range_start=0, range_end=None):
    """
    Scrape lyrics for songs in the given index range [range_start, range_end).
    Each 100k block saves to its own folder on Drive:
      0-99999      -> data/lyrics/
      100000-199999 -> data/lyrics_1/
      200000-299999 -> data/lyrics_2/
      ...
    Each folder has its own progress.json and failed_lyrics.csv.
    """
    setup_drive()
    import common as _c

    # --- Determine output folder from range ---
    block_idx = range_start // 100_000
    if block_idx == 0:
        folder_name = "lyrics"
    else:
        folder_name = f"lyrics_{block_idx}"

    out_dir = Path(_c.DEFAULT_OUTPUT_DIR).parent / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Local cache directory (unique per block to avoid conflicts)
    if Path("/content").exists():
        local_cache = Path(f"/content/lyrics_cache_{block_idx}")
    else:
        local_cache = Path("data") / f"lyrics_cache_{block_idx}"
    local_cache.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🎸 PHASE 3: Scraping lyrics content (Batched)")
    print("=" * 60)

    all_songs = load_songs_from_csv(songs_db_csv)
    if not all_songs:
        print(f"❌ No songs found in {songs_db_csv}")
        return

    # Slice to requested range
    if range_end is None:
        range_end = len(all_songs)
    range_end = min(range_end, len(all_songs))
    songs = all_songs[range_start:range_end]
    del all_songs  # free full list

    if not songs:
        print(f"❌ No songs in range [{range_start}, {range_end})")
        return

    print(f"  Range: [{range_start}, {range_end}) = {len(songs)} songs")
    print(f"  Output: {folder_name}/")

    # --- Resume support (relative to this range) ---
    progress_file = out_dir / "progress.json"

    def load_progress():
        if progress_file.exists():
            try:
                with open(progress_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def save_progress(songs_done, batch_num, txt_num, ok, err):
        data = {
            "songs_attempted": songs_done,
            "batch_counter": batch_num,
            "txt_counter": txt_num,
            "scraped": ok,
            "errors": err,
            "range_start": range_start,
            "range_end": range_end,
        }
        try:
            with open(progress_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"  ⚠ Could not save progress: {e}")

    prev = load_progress()
    skip = 0
    start_batch = 0
    start_txt = 0
    start_ok = 0
    start_err = 0
    if prev:
        skip = prev.get("songs_attempted", 0)
        start_batch = prev.get("batch_counter", 0)
        start_txt = prev.get("txt_counter", 0)
        start_ok = prev.get("scraped", 0)
        start_err = prev.get("errors", 0)

    if skip > 0 and skip < len(songs):
        songs = songs[skip:]
        print(f"  ⏩ Resuming: skipping {skip} already-processed songs")

    print(f"  Songs remaining: {len(songs)}")
    print(f"  Workers: {workers} | Delay: {delay}s")
    print(
        f"  Songs per .txt: {songs_per_txt} | .txt files per ZIP: {txt_files_per_zip}"
    )
    print("=" * 60)

    # Failed logs scoped to this range
    failed_log = out_dir / "failed.txt"
    retry_csv = out_dir / "failed_lyrics.csv"
    stats = {"scraped": start_ok, "errors": start_err}
    songs_attempted = [skip]  # songs attempted within this range

    # Initialize lyrics processor
    processor = LyricsProcessor(lang_model_path)

    # Batching state
    batch_lock = threading.Lock()
    fail_lock = threading.Lock()
    lyrics_buffer = []  # Buffer for current .txt file
    txt_buffer = []  # Buffer for current ZIP batch
    txt_counter = [start_txt]  # Number of .txt files created
    batch_counter = [start_batch]  # Number of ZIP batches created
    upload_futures = []
    upload_executor = ThreadPoolExecutor(max_workers=2)

    # Write retry CSV header if it doesn't exist
    if not retry_csv.exists():
        retry_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(retry_csv, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(
                ["artist_id", "artist_name", "song_url", "song_name", "error"]
            )

    def log_fail(
        url: str,
        reason: str,
        artist_id: str = "",
        artist_name: str = "",
        song_name: str = "",
    ):
        # Quick text log
        failed_log.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_log, "a", encoding="utf-8") as f:
            f.write(f"{url} | {reason}\n")
        # CSV retry log on Drive
        with fail_lock:
            with open(retry_csv, "a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow([artist_id, artist_name, url, song_name, reason])

    def get_lyrics(
        song_url: str,
        artist_id: str = "",
        artist_name: str = "",
        song_name: str = "",
    ) -> str:
        """Fetch and process lyrics for a single song using requests."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

        from bs4 import BeautifulSoup

        for attempt in range(retries):
            try:
                response = requests.get(song_url, headers=headers, timeout=30)
                response.raise_for_status()
                time.sleep(delay)

                # Decode as UTF-8 to avoid mojibake (e.g. "CÃª" instead of "Cê")
                html = response.content.decode("utf-8", errors="replace")
                response.close()  # free response body

                soup = BeautifulSoup(html, "html.parser")
                del html  # free raw HTML

                # Try cifra_cnt first (songs with chords), then letra (lyrics-only)
                cifra_div = soup.find("div", class_="cifra_cnt")
                if cifra_div:
                    inner_html = "".join(str(child) for child in cifra_div.children)
                    del soup
                    return processor.process_html(
                        f"<pre>{inner_html}</pre>",
                        detect_lang=True,
                        artist_name=artist_name,
                        song_name=song_name,
                    )

                letra_div = soup.find("div", class_="letra")
                if letra_div:
                    # letra pages use <p> for stanzas and <br/> for lines
                    # Convert to plain text preserving line breaks
                    for br in letra_div.find_all("br"):
                        br.replace_with("\n")
                    lines = []
                    for p in letra_div.find_all("p"):
                        lines.append(p.get_text())
                    text = "\n\n".join(lines) if lines else letra_div.get_text()
                    del soup
                    return processor.process_html(
                        f"<pre>{text}</pre>",
                        detect_lang=True,
                        artist_name=artist_name,
                        song_name=song_name,
                    )

                del soup
                log_fail(
                    song_url,
                    "No cifra_cnt or letra div found",
                    artist_id,
                    artist_name,
                    song_name,
                )
                return None

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2**attempt)
                else:
                    log_fail(song_url, str(e), artist_id, artist_name, song_name)
                    return None

        return None

    def flush_txt_file():
        """Save current lyrics buffer to a .txt file in local cache."""
        with batch_lock:
            if not lyrics_buffer:
                return
            txt_counter[0] += 1
            num = txt_counter[0]
            content = "\n\n".join(lyrics_buffer)  # Separate songs with blank line
            lyrics_buffer.clear()

        txt_path = local_cache / f"lyrics_{num:06d}.txt"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(content)

            with batch_lock:
                txt_buffer.append(txt_path)
                n = len(txt_buffer)

            print(f"  💾 Created {txt_path.name} ({n}/{txt_files_per_zip} .txt files)")

            # Check if we need to create a ZIP batch
            if n >= txt_files_per_zip:
                flush_zip_batch()
        except Exception as e:
            print(f"  ✗ Error saving .txt file: {e}")

    def flush_zip_batch():
        """Create a ZIP batch from .txt files and upload to Drive."""
        with batch_lock:
            if not txt_buffer:
                return
            batch_counter[0] += 1
            num = batch_counter[0]
            files = list(txt_buffer)
            txt_buffer.clear()
            # Snapshot progress at ZIP creation time
            snap_attempted = songs_attempted[0]
            snap_ok = stats["scraped"]
            snap_err = stats["errors"]
            snap_txt = txt_counter[0]

        zip_path = local_cache / f"batch_{num:03d}.zip"
        print(f"  📦 Creating {zip_path.name} with {len(files)} .txt files...")
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for txt_file in files:
                    zf.write(txt_file, txt_file.name)

            # Delete the .txt files after adding to ZIP
            for txt_file in files:
                txt_file.unlink(missing_ok=True)

            # Save checkpoint to Drive so we can resume after crash
            save_progress(snap_attempted, num, snap_txt, snap_ok, snap_err)
            print(f"  💾 Progress saved: {snap_attempted} songs attempted")

            def upload():
                try:
                    dest = out_dir / zip_path.name
                    if not dest.exists():
                        shutil.copy2(zip_path, dest)
                    zip_path.unlink(missing_ok=True)
                    print(f"  ✅ Uploaded {zip_path.name} to Drive")
                except Exception as e:
                    print(f"  ✗ Upload {zip_path.name}: {e}")

            upload_futures.append(upload_executor.submit(upload))
        except Exception as e:
            print(f"  ✗ ZIP error: {e}")

    def add_lyrics(content: str):
        """Add lyrics content to the buffer."""
        with batch_lock:
            lyrics_buffer.append(content)
            n = len(lyrics_buffer)

        # Flush when buffer reaches songs_per_txt
        if n >= songs_per_txt:
            flush_txt_file()

    def process_song(item):
        """Process a single song."""
        idx, total, (artist_id, artist_name, song_url, song_name) = item
        content = get_lyrics(song_url, artist_id, artist_name, song_name)

        with batch_lock:
            songs_attempted[0] += 1
            if content:
                stats["scraped"] += 1
            else:
                stats["errors"] += 1

        if content:
            add_lyrics(content)

        if idx % 50 == 0 or idx == total:
            ok = stats["scraped"]
            err = stats["errors"]
            buf = len(lyrics_buffer)
            print(
                f"  [{idx}/{total}] ok={ok} err={err} buf={buf} | {artist_name} - {song_name}"
            )

    # Process songs in small chunks to avoid OOM (don't submit 4M futures at once)
    chunk_size = workers * 10  # e.g. 200 songs at a time
    total = len(songs)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, total, chunk_size):
            chunk = songs[start : start + chunk_size]
            futures = [
                executor.submit(process_song, (start + j, total, s))
                for j, s in enumerate(chunk, 1)
            ]
            for future in as_completed(futures):
                future.result()

    # Flush any remaining content
    flush_txt_file()
    flush_zip_batch()

    # Wait for all uploads to complete
    for f in as_completed(upload_futures):
        f.result()
    upload_executor.shutdown(wait=True)

    # Save final progress
    save_progress(
        songs_attempted[0],
        batch_counter[0],
        txt_counter[0],
        stats["scraped"],
        stats["errors"],
    )

    print("=" * 60)
    print("✅ Phase 3 done.")
    print(f"  Scraped: {stats['scraped']} | Errors: {stats['errors']}")
    print(
        f"  Total .txt files: {txt_counter[0]} | Total ZIP batches: {batch_counter[0]}"
    )
    print("=" * 60)


if __name__ == "__main__":
    import sys

    # Usage: python scrape_lyrics.py [start] [end]
    # start/end are song indices (0-based). Default: all songs.
    _start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    _end = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(range_start=_start, range_end=_end)
