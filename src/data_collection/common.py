# Shared config and helpers for Cifra Club scraper (Colab).
# All paths are under data/ relative to the folder containing this file, so when you
# upload this folder to Drive, scrape_artists/scrape_songs/scrape_lyrics save to that same folder's data/.
# Call setup_drive() once at start in Colab to mount Drive and create dirs.

import json
import re
import string
from pathlib import Path

BASE_URL = "https://www.cifraclub.com"
DEFAULT_DELAY = 1
DEFAULT_RETRIES = 2
DEFAULT_WORKERS = 20
TIMEOUT = 5000
ZIP_BATCH_SIZE = 300

# Project root = folder containing common.py (same folder as scripts). All data under data/.
_PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = str(DATA_DIR / "output")
ARTISTS_CSV = str(DATA_DIR / "artists.csv")
ARTISTS_FROM_OUTPUT_CSV = str(DATA_DIR / "artists_from_output.csv")
SONGS_CSV = str(DATA_DIR / "songs.csv")
FAILED_LOG = str(DATA_DIR / "failed.txt")

# Local cache for ZIP uploads. Colab: /content/output_cache; local: data/output_cache
LOCAL_OUTPUT_DIR = DATA_DIR / "output_cache"


def setup_drive():
    """Mount Drive and create data dirs. Call once at start in Colab."""
    from google.colab import drive

    drive.mount("/content/drive")
    global LOCAL_OUTPUT_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "output").mkdir(parents=True, exist_ok=True)
    LOCAL_OUTPUT_DIR = Path("/content/output_cache")
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str, max_length: int = 200) -> str:
    invalid_chars = r'\\/:*?"<>|'
    for char in invalid_chars:
        name = name.replace(char, "_")
    name = name.strip(". ")
    if len(name) > max_length:
        name = name[:max_length].strip(". ")
    if not name:
        name = "untitled"
    return name


def artist_letter_bucket(artist_name: str) -> str:
    if not artist_name:
        return "#"
    first = artist_name.strip()[:1].upper()
    if first and first in string.ascii_uppercase:
        return first
    return "#"


def _extract_json_array(text: str, start_index: int) -> str | None:
    array_start = text.find("[", start_index)
    if array_start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(array_start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[array_start : i + 1]
    return None


def extract_artist_songs_from_rsc(rsc_text: str) -> list:
    key_match = re.search(r'"artistSongs"\s*:\s*', rsc_text)
    if not key_match:
        return []
    array_text = _extract_json_array(rsc_text, key_match.end())
    if not array_text:
        return []
    try:
        return json.loads(array_text)
    except json.JSONDecodeError:
        return []
