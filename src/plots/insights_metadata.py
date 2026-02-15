"""
Quick insights and plots from data/uncompressed/metadata.csv.
Requires: pip install pandas matplotlib
"""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

METADATA_PATH = (
    Path(__file__).resolve().parent / "data" / "uncompressed" / "metadata.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "insights_plots"


def load_metadata(path: Path) -> pd.DataFrame:
    """Load metadata CSV. Uses dtype to avoid mixed types and save memory."""
    return pd.read_csv(
        path,
        dtype={
            "artist_name": "string",
            "song_name": "string",
            "language": "string",
            "key": "string",
            "has_chords": "boolean",
        },
    )


def plot_songs_by_language(df: pd.DataFrame) -> None:
    """Songs count per language (top 15)."""
    lang = df["language"].fillna("(unknown)").value_counts()
    top = lang.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    top.plot(kind="barh", ax=ax, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xlabel("Number of songs")
    ax.set_ylabel("Language")
    ax.set_title("Songs by language (top 15)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "songs_by_language.png", dpi=120)
    plt.close()
    print(f"Saved songs_by_language.png (top: {top.index[0]} = {top.iloc[0]:,})")


def plot_chords_vs_no_chords(df: pd.DataFrame) -> None:
    """Pie chart: songs with chords vs without."""
    chords = df["has_chords"].fillna(False)
    counts = chords.value_counts()
    labels = ["With chords", "No chords"]
    sizes = [counts.get(True, 0), counts.get(False, 0)]
    colors = ["#2ecc71", "#e74c3c"]
    explode = (0.05, 0)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Songs with chords vs no chords")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "chords_vs_no_chords.png", dpi=120)
    plt.close()
    print(f"Saved chords_vs_no_chords.png (with: {sizes[0]:,}, without: {sizes[1]:,})")


def plot_top_artists(df: pd.DataFrame, top_n: int = 20) -> None:
    """Top N artists by number of songs."""
    artist_counts = df["artist_name"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    artist_counts.plot(
        kind="barh", ax=ax, color="teal", edgecolor="darkgreen", alpha=0.8
    )
    ax.set_xlabel("Number of songs")
    ax.set_ylabel("Artist")
    ax.set_title(f"Top {top_n} artists by number of songs")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_artists.png", dpi=120)
    plt.close()
    print(
        f"Saved top_artists.png (top: {artist_counts.index[0]} = {artist_counts.iloc[0]:,})"
    )


def plot_keys(df: pd.DataFrame, top_n: int = 12) -> None:
    """Songs per key (only rows with key set)."""
    keys = df["key"].dropna()
    keys = keys[keys.str.strip() != ""]
    if keys.empty:
        print("No key data to plot.")
        return
    key_counts = keys.value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 5))
    key_counts.plot(kind="bar", ax=ax, color="coral", edgecolor="darkred", alpha=0.8)
    ax.set_xlabel("Key")
    ax.set_ylabel("Number of songs")
    ax.set_title("Songs by key (top 12)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "songs_by_key.png", dpi=120)
    plt.close()
    print(f"Saved songs_by_key.png")


def print_summary(df: pd.DataFrame) -> None:
    """Print basic summary stats."""
    print("\n--- Summary ---")
    print(f"Total songs: {len(df):,}")
    print(f"Unique artists: {df['artist_name'].nunique():,}")
    print(f"Unique languages: {df['language'].nunique()}")
    print(
        f"With chords: {df['has_chords'].sum():,} ({100 * df['has_chords'].mean():.1f}%)"
    )
    with_key = (df["key"].notna() & (df["key"].str.strip() != "")).sum()
    print(f"With key: {int(with_key):,}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {METADATA_PATH} ...")
    df = load_metadata(METADATA_PATH)
    print_summary(df)
    print("\nGenerating plots...")
    plot_songs_by_language(df)
    plot_chords_vs_no_chords(df)
    plot_top_artists(df)
    plot_keys(df)
    print(f"\nPlots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
