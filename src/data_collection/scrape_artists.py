"""
Phase 1: Collect all artists from Cifra Club and save to artists.csv.
Run in Colab. Supports multiple workers.
"""

import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from common import (
    BASE_URL,
    DEFAULT_DELAY,
    DEFAULT_RETRIES,
    DEFAULT_WORKERS,
    ARTISTS_CSV,
    FAILED_LOG,
    setup_drive,
)

# --- Parameters (edit for Colab) ---
output_dir = None  # None = use Drive output from common
delay = DEFAULT_DELAY
retries = DEFAULT_RETRIES
workers = DEFAULT_WORKERS
letters = None  # None = A-Z + #


def get_artists_from_letter(letter: str, quiet: bool = False) -> list:
    letter_encoded = quote(letter) if letter == "#" else letter
    url = f"{BASE_URL}/letra/{letter_encoded}/lista/"
    if not quiet:
        print(f"  📖 Letter: {letter} -> {url}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        time.sleep(delay)
        soup = BeautifulSoup(resp.text, "html.parser")
        artists = []
        for link in soup.select("section ol li a, section ul li a"):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            if not href:
                continue
            match = re.match(r"^/([^/]+)/?$", href)
            if match:
                artists.append((match.group(1), text))
        if not quiet:
            print(f"  ✓ {letter}: {len(artists)} artists")
        return artists
    except Exception as e:
        if not quiet:
            print(f"  ✗ {letter}: {e}")
        Path(FAILED_LOG).parent.mkdir(parents=True, exist_ok=True)
        with open(FAILED_LOG, "a", encoding="utf-8") as f:
            f.write(f"{url} | Letter page: {e}\n")
        return []


def save_artists_to_csv(artists: list, csv_path: str):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["artist_id", "artist_name"])
        writer.writerows(artists)
    print(f"  ✓ Saved {len(artists)} artists to {path}")


def run():
    global output_dir, letters
    setup_drive()
    import common as _c

    if output_dir is None:
        output_dir = _c.DEFAULT_OUTPUT_DIR
    if letters is None:
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["#"]

    print("=" * 60)
    print("📋 PHASE 1: Collecting Artists")
    print("=" * 60)
    print(f"  Workers: {workers} | Delay: {delay}s")
    print("=" * 60)

    all_artists = []
    n = min(workers, len(letters))
    with ThreadPoolExecutor(max_workers=n) as executor:
        future_to_letter = {
            executor.submit(get_artists_from_letter, letter, True): letter
            for letter in letters
        }
        for future in as_completed(future_to_letter):
            letter = future_to_letter[future]
            try:
                artists = future.result()
                all_artists.extend(artists)
                print(f"  ✓ {letter}: {len(artists)} artists")
            except Exception as e:
                print(f"  ✗ {letter}: {e}")

    print(f"\n📊 Total: {len(all_artists)} artists")
    save_artists_to_csv(all_artists, ARTISTS_CSV)
    print("=" * 60)


if __name__ == "__main__":
    run()
