# Cifra Club Scraper

A Python scraper that collects guitar tabs from Cifra Club using a **three-phase approach** with CSV checkpoints.

## ⚠️ Important Notice

This scraper is intended for **personal, educational use only**. Please respect:

- Cifra Club's Terms of Service
- Copyright laws regarding song lyrics and musical compositions
- Rate limiting to avoid overloading the server

The user assumes all responsibility for how this tool is used.

---

## Project Layout

The scraper is split into **three scripts** that work on Colab or locally. All persistent data lives under **`data/`** (relative to the project folder).

| File                    | Role                                                                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`common.py`**         | Shared config, paths, and helpers. All paths point to **`data/`** so the project works the same locally or when the folder is uploaded to Drive. |
| **`scrape_artists.py`** | Phase 1: collect artists → `data/artists.csv`                                                                                                    |
| **`scrape_songs.py`**   | Phase 2: collect songs per artist → `data/output/{letter}/{Artist}__{id}.csv` (ZIP batches on Colab)                                             |
| **`scrape_lyrics.py`**  | Phase 3: scrape tab content → `data/output/{letter}/{Artist}/{Song}.txt`                                                                         |
| **`run_scraper.ipynb`** | Colab notebook to mount Drive, install deps, and run any phase.                                                                                  |

### Data directory (`data/`)

- **`data/artists.csv`** – Phase 1 output (artist_id, artist_name).
- **`data/artists_from_output.csv`** – Optional; list of artists already present in `data/output/` so Phase 2 can skip them.
- **`data/output/`** – Phase 2 writes per-artist CSVs here (`{Letter}/{Artist}__{id}.csv`); Phase 3 writes tab `.txt` files here (`{Letter}/{Artist}/{Song}.txt`).
- **`data/failed.txt`** – Failed URLs logged by the Python scripts.

Paths are set in `common.py` using the folder that contains `common.py`, so uploading the project to Google Drive keeps everything under that folder’s **`data/`**.

---

## Three-Phase Approach

1. **Phase 1: Collect Artists** → `scrape_artists.py` → `data/artists.csv`
2. **Phase 2: Collect Songs** → `scrape_songs.py` → `data/output/{Letter}/{Artist}__{id}.csv`
3. **Phase 3: Scrape Tabs** → `scrape_lyrics.py` → `data/output/{Letter}/{Artist}/{Song}.txt`

You can run phases in order, resume, or run a single phase. Phase 2 skips artists already in `data/output/` (using `data/artists_from_output.csv` if present). Phase 3 skips songs that already have a `.txt` file.

---

## Running in Google Colab

1. Upload the **entire project folder** to Google Drive (e.g. `MyDrive/song-GPT`).
2. Open **`run_scraper.ipynb`** in Colab (upload the notebook or open from Drive).
3. Run the cells in order:
   - Mount Drive and set `PROJECT_PATH` to your project folder.
   - Install dependencies (`requests`, `beautifulsoup4`). For Phase 3 only, also install Playwright and Chromium.
   - Set `PHASE` to `'artists'`, `'songs'`, or `'lyrics'` and run the last cell to execute the chosen script.

All outputs go to **`data/`** inside that project folder on Drive.

---

## Running locally (Python)

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. For Phase 3 only, install Playwright:

```powershell
playwright install chromium
```

3. Run a phase (from the project root):

```powershell
python src/data_collection/scrape_artists.py
python src/data_collection/scrape_songs.py
python src/data_collection/scrape_lyrics.py
```

Scripts use parameters defined at the top of each file (e.g. `workers`, `delay`, `letters_filter`). Edit those to change behavior; there is no CLI.

---

## Output structure

### CSVs

- **`data/artists.csv`** – Phase 1: `artist_id`, `artist_name`
- **`data/output/{Letter}/{Artist}__{id}.csv`** – Phase 2: one CSV per artist

### Tabs (Phase 3)

```
data/output/
├── A/
│   ├── Artist Name/
│   │   ├── Song One.txt
│   │   └── Song Two.txt
│   └── ...
├── B/
│   └── ...
└── #/
    └── ...
```

---

## Resume support

- **Phase 2** reads `data/artists.csv` and (if present) `data/artists_from_output.csv` to skip artists already in `data/output/`.
- **Phase 3** discovers songs by scanning CSVs in `data/output/` and skips any song that already has a `.txt` file.

If a run is interrupted, run the same phase again; already-done work is skipped.

---

## Logs

- Progress is printed to the console.
- Failed URLs are appended to **`data/failed.txt`**.
