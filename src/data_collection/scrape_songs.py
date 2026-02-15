"""
Phase 2: Collect songs for each artist and save as CSV in ZIP batches to Drive.
Run in Colab. Supports multiple workers.
Requires: artists.csv, artists_from_output.csv (run extract_artists.py on output/ to build the latter).
"""

import csv
import io
import os
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests

from common import (
    BASE_URL,
    DEFAULT_DELAY,
    DEFAULT_RETRIES,
    DEFAULT_WORKERS,
    ZIP_BATCH_SIZE,
    ARTISTS_CSV,
    ARTISTS_FROM_OUTPUT_CSV,
    FAILED_LOG,
    setup_drive,
    sanitize_filename,
    artist_letter_bucket,
    extract_artist_songs_from_rsc,
)

# --- Parameters (edit for Colab) ---
output_dir = None
delay = DEFAULT_DELAY
retries = DEFAULT_RETRIES
workers = DEFAULT_WORKERS
letters_filter = None  # None = all letters; or e.g. ["A","B"] to limit
zip_batch_size = ZIP_BATCH_SIZE


def log_failure(url: str, reason: str, failed_log: Path):
    with open(failed_log, "a", encoding="utf-8") as f:
        f.write(f"{url} | {reason}\n")


def get_songs_from_artist_sync(artist_id: str, artist_name: str) -> tuple:
    """Returns (artist_id, artist_name, list of (song_url, song_name))."""
    url = f"{BASE_URL}/{artist_id}/musicas.html?order=alphabetical"
    headers = {
        "Rsc": "1",
        "Accept": "text/x-component",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    }
    failed_log = Path(FAILED_LOG)
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            time.sleep(delay)
            songs_data = extract_artist_songs_from_rsc(resp.text)
            if not songs_data:
                return (artist_id, artist_name, [])
            songs = []
            for item in songs_data:
                if not isinstance(item, dict):
                    continue
                name = (
                    item.get("name")
                    or item.get("title")
                    or item.get("songName")
                    or item.get("song")
                )
                url_value = (
                    item.get("url")
                    or item.get("link")
                    or item.get("href")
                    or item.get("path")
                )
                if not name:
                    name = item.get("id") or item.get("songId") or ""
                if not url_value:
                    continue
                full_url = (
                    url_value
                    if not str(url_value).startswith("/")
                    else urljoin(BASE_URL, url_value)
                )
                songs.append((full_url, str(name).strip()))
            return (artist_id, artist_name, songs)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                log_failure(url, f"Artist page: {e}", failed_log)
                return (artist_id, artist_name, [])
    return (artist_id, artist_name, [])


def load_artists_from_csv(csv_path: str) -> list:
    path = Path(csv_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return [(row[0], row[1]) for row in reader if len(row) >= 2]


def load_artists_from_output_csv(csv_path: str) -> list:
    path = Path(csv_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return [(row[0], row[1]) for row in reader if len(row) >= 2]


def get_existing_artist_keys(out_dir: Path) -> tuple[set, set]:
    existing_ids = set()
    existing_names = set()
    if not out_dir.exists():
        return existing_ids, existing_names
    for root, _, files in os.walk(out_dir):
        for file in files:
            if file.lower().endswith(".txt"):
                parent_name = Path(root).name
                if parent_name:
                    existing_names.add(parent_name)
            if file.lower().endswith(".csv") and "__" in file:
                stem = Path(file).stem
                parts = stem.split("__")
                if len(parts) >= 2 and parts[-1].strip():
                    existing_ids.add(parts[-1].strip())
    return existing_ids, existing_names


def run():
    global output_dir
    setup_drive()
    import common as _c

    if output_dir is None:
        output_dir = _c.DEFAULT_OUTPUT_DIR
    out_dir = Path(output_dir)

    print("=" * 60)
    print("🎵 PHASE 2: Collecting Songs (ZIP batches → Drive)")
    print("=" * 60)

    all_artists = load_artists_from_csv(ARTISTS_CSV)
    if not all_artists:
        print("❌ No artists in artists.csv. Run scrape_artists.py first.")
        return

    output_artists = load_artists_from_output_csv(ARTISTS_FROM_OUTPUT_CSV)
    all_ids = {a[0] for a in all_artists}
    output_ids = {a[0] for a in output_artists}
    missing_ids = all_ids - output_ids
    artists = [(aid, aname) for aid, aname in all_artists if aid in missing_ids]

    if letters_filter:
        artists = [
            (aid, aname)
            for aid, aname in artists
            if artist_letter_bucket(aname) in letters_filter
        ]

    existing_ids, existing_names = get_existing_artist_keys(out_dir)
    artists_to_process = [
        (aid, aname)
        for aid, aname in artists
        if aid not in existing_ids and sanitize_filename(aname) not in existing_names
    ]

    if not artists_to_process:
        print("✅ All artists already in output/.")
        return

    print(f"  Artists to process: {len(artists_to_process)}")
    print(f"  Workers: {workers} | Batch size: {zip_batch_size}")
    print("=" * 60)

    batch_lock = threading.Lock()
    batch_buffer = []
    batch_counter = [0]
    upload_futures = []
    upload_executor = ThreadPoolExecutor(max_workers=4)
    local_out = Path(_c.LOCAL_OUTPUT_DIR)
    local_out.mkdir(parents=True, exist_ok=True)

    def flush_batch():
        with batch_lock:
            if not batch_buffer:
                return
            batch_counter[0] += 1
            num = batch_counter[0]
            buf = list(batch_buffer)
            batch_buffer.clear()
        zip_path = local_out / f"batch_{num:04d}.zip"
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for artist_id, artist_name, songs in buf:
                    safe_artist = sanitize_filename(artist_name)
                    letter = artist_letter_bucket(artist_name)
                    csv_name = f"{letter}/{safe_artist}__{artist_id}.csv"
                    csv_buf = io.StringIO()
                    w = csv.writer(csv_buf)
                    w.writerow(["artist_id", "artist_name", "song_url", "song_name"])
                    for song_url, song_name in songs:
                        w.writerow([artist_id, artist_name, song_url, song_name])
                    zf.writestr(csv_name, csv_buf.getvalue())

            def upload():
                try:
                    rel = zip_path.relative_to(local_out)
                    dest = out_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        shutil.copy2(zip_path, dest)
                    zip_path.unlink(missing_ok=True)
                except Exception as e:
                    print(f"  ✗ Upload {zip_path.name}: {e}")

            upload_futures.append(upload_executor.submit(upload))
        except Exception as e:
            print(f"  ✗ ZIP error: {e}")

    def buffer_artist_songs(artist_id: str, artist_name: str, songs: list):
        with batch_lock:
            batch_buffer.append((artist_id, artist_name, songs))
            n = len(batch_buffer)
        if n >= zip_batch_size:
            flush_batch()

    completed = 0
    total = len(artists_to_process)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_artist = {
            executor.submit(get_songs_from_artist_sync, aid, aname): (aid, aname)
            for aid, aname in artists_to_process
        }
        for future in as_completed(future_to_artist):
            artist_id, artist_name, songs = future.result()
            if songs:
                buffer_artist_songs(artist_id, artist_name, songs)
            completed += 1
            if completed % 100 == 0:
                print(f"  [{completed}/{total}] artists...")

    flush_batch()
    for f in as_completed(upload_futures):
        f.result()
    upload_executor.shutdown(wait=True)
    print("=" * 60)
    print("✅ Phase 2 done.")
    print("=" * 60)


if __name__ == "__main__":
    run()
