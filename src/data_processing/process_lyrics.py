"""
DOM Processing Module for Lyrics
Handles chord interpolation, tablature detection, language identification, and tag cleaning.
"""

import re
import threading
import warnings
from bs4 import BeautifulSoup, NavigableString, MarkupResemblesLocatorWarning
from pathlib import Path

# Suppress spurious BeautifulSoup warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Try to import language detection libraries (optional)
try:
    from langdetect import detect, DetectorFactory

    DetectorFactory.seed = 0  # reproducible results
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Lock for thread-safe langdetect calls (langdetect uses global state)
_lang_lock = threading.Lock()


class LyricsProcessor:
    """Process HTML lyrics content into formatted text with tokens."""

    def __init__(self, lang_model_path: str = None, use_langdetect: bool = True):
        """Initialize the processor with optional language detection."""
        self.use_langdetect = use_langdetect

        # Regex to detect user tags like [Intro], [Coro], etc.
        self.user_tag_pattern = re.compile(r"^\[([^\]]{3,30})\]$")

        # Common chord patterns
        self.chord_pattern = re.compile(
            r"^[A-G](#|b)?(m|maj|min|sus|dim|aug)?\d*(\/[A-G](#|b)?)?$"
        )

    def detect_language(self, text: str) -> str:
        """Detect language using langdetect (thread-safe)."""
        if not text.strip() or not self.use_langdetect or not LANGDETECT_AVAILABLE:
            return "unknown"

        # Strip our custom tags, tab blocks, chord refs, and music notation
        clean_text = re.sub(r"<\|.*?\|>", "", text)
        clean_text = re.sub(
            r"<\|TAB_START\|>.*?<\|TAB_END\|>", "", clean_text, flags=re.DOTALL
        )
        clean_text = re.sub(r"\[.*?\]", "", clean_text)
        clean_text = re.sub(
            r"^[A-Ga-g]\|[\-\d\s|]+$", "", clean_text, flags=re.MULTILINE
        )
        clean_text = re.sub(r"^\s*key:\s*\w+\s*$", "", clean_text, flags=re.MULTILINE)
        clean_text = re.sub(
            r"^\|[A-G/#\s\d|]+\|?\s*$", "", clean_text, flags=re.MULTILINE
        )
        clean_text = " ".join(clean_text.split())

        if len(clean_text) < 20:
            return "unknown"

        with _lang_lock:
            try:
                return detect(clean_text)
            except Exception:
                return "unknown"

    def is_chord(self, text: str) -> bool:
        """Check if text looks like a chord."""
        return bool(self.chord_pattern.match(text.strip()))

    def clean_user_tag(self, tag_text: str) -> str:
        """Clean user tags."""
        return f"[{' '.join(tag_text.split())}]"

    def process_tablature_block(self, span_element) -> str:
        """Process a tablature block and wrap with tokens."""
        content = span_element.get_text().strip()
        return f"<|TAB_START|>\n{content}\n<|TAB_END|>"

    def get_text_with_chord_positions(self, html_line: str):
        """
        Parse HTML line and extract text with chord positions.
        Uses regex to preserve exact whitespace positioning.
        Returns: (plain_text, [(column, chord), ...], tab_element)
        """
        soup = BeautifulSoup(html_line, "html.parser")

        # Check for tablatura
        if soup.find("span", class_="tablatura"):
            return None, "TABLATURA", soup.find("span", class_="tablatura")

        # Use regex to find chord positions while preserving whitespace
        chord_positions = []

        # Pattern to match <b> tags with any attributes
        pattern = r"<b[^>]*>([^<]+)</b>"

        for match in re.finditer(pattern, html_line):
            chord = match.group(1).strip()
            if chord:
                # Calculate visible column by removing all HTML tags before this position
                text_before = html_line[: match.start()]
                visible_text = re.sub(r"<[^>]+>", "", text_before)
                col = len(visible_text)
                chord_positions.append((col, chord))

        # Get plain text by removing all HTML tags
        plain_text = re.sub(r"<[^>]+>", "", html_line)

        return plain_text, chord_positions, None

    def insert_chords_inline(self, lyrics: str, chord_positions: list) -> str:
        """
        Insert chords into lyrics at their column positions.
        Chords are positioned based on absolute column in the chord line,
        but lyrics may have less (or no) leading whitespace.
        """
        if not chord_positions:
            return lyrics

        # Sort by position in reverse order for insertion
        sorted_chords = sorted(chord_positions, key=lambda x: x[0], reverse=True)

        result = lyrics
        for col, chord in sorted_chords:
            # Insert at the column position
            if col >= len(result):
                # Chord position is beyond lyrics text
                result += f" [{chord}]"
            elif col <= 0:
                # Chord at or before start
                result = f"[{chord}]" + result
            else:
                # Insert at position
                result = result[:col] + f"[{chord}]" + result[col:]

        return result

    def process_html(
        self,
        html: str,
        detect_lang: bool = True,
        artist_name: str = None,
        song_name: str = None,
    ) -> str:
        """Main processing function."""
        soup = BeautifulSoup(html, "html.parser")
        pre = soup.find("pre")

        if not pre:
            return ""

        # First, extract and replace tablatura blocks with placeholders
        tab_blocks = []
        tab_counter = 0

        for tab_span in pre.find_all("span", class_="tablatura"):
            placeholder = f"__TABLATURA_{tab_counter}__"
            tab_blocks.append(self.process_tablature_block(tab_span))
            tab_span.replace_with(placeholder)
            tab_counter += 1

        # Now get the inner HTML and split by lines
        inner_html = "".join(str(child) for child in pre.children)
        lines = inner_html.split("\n")

        result_lines = []
        pending_chords = []

        for line in lines:
            # Check if line contains a tablatura placeholder
            if "__TABLATURA_" in line:
                if pending_chords:
                    pending_chords = []
                # Extract tab number and insert the processed tab
                import re as regex_module

                match = regex_module.search(r"__TABLATURA_(\d+)__", line)
                if match:
                    tab_idx = int(match.group(1))
                    if tab_idx < len(tab_blocks):
                        result_lines.append(tab_blocks[tab_idx])
                continue

            # Parse this line
            plain_text, data, _ = self.get_text_with_chord_positions(line)

            # Strip trailing whitespace but keep leading
            plain_text_stripped = plain_text.rstrip()

            # Check if it's a user tag
            stripped = plain_text_stripped.strip()
            if stripped:
                tag_match = self.user_tag_pattern.match(stripped)
                if tag_match and not self.is_chord(tag_match.group(1)):
                    if pending_chords:
                        pending_chords = []
                    result_lines.append(self.clean_user_tag(tag_match.group(1)))
                    continue

            # Determine if line has actual lyrics (not just chord names)
            has_chords = len(data) > 0
            # A chord-only line has chords and only whitespace between them
            if has_chords:
                # Remove all chord names from the text by position
                # Sort chords by position (reverse) and remove them
                text_without_chords = plain_text_stripped
                sorted_chords = sorted(data, key=lambda x: x[0], reverse=True)
                for col, chord in sorted_chords:
                    # Remove the chord at its position
                    if col < len(text_without_chords):
                        end_pos = min(col + len(chord), len(text_without_chords))
                        text_without_chords = (
                            text_without_chords[:col] + text_without_chords[end_pos:]
                        )
                # If what's left is only whitespace, it's a chord-only line
                has_lyrics = bool(text_without_chords.strip())
            else:
                has_lyrics = any(c.isalpha() for c in plain_text_stripped)

            if has_chords and not has_lyrics:
                # Chord-only line - store for next lyrics line
                pending_chords = data
            elif has_lyrics:
                # Lyrics line - apply pending chords
                if pending_chords:
                    processed = self.insert_chords_inline(
                        plain_text_stripped, pending_chords
                    )
                    result_lines.append(processed)
                    pending_chords = []
                else:
                    result_lines.append(plain_text_stripped)
            elif stripped:
                # Non-empty, non-lyrics, non-chords line
                result_lines.append(plain_text_stripped)
            else:
                # Empty line
                result_lines.append("")

        # Process result
        processed = "\n".join(result_lines)

        # Detect language if requested
        lang_token = ""
        if detect_lang:
            lang = self.detect_language(processed)
            lang_token = f"<|lang:{lang}|>\n"

        # Build artist-song tag if provided
        artist_song_tag = ""
        if artist_name and song_name:
            artist_song_tag = f"<|{artist_name} - {song_name}|>\n"

        # Build final result with order: lang -> artist-song -> song_start -> content
        result = f"{lang_token}{artist_song_tag}<|song_start|>\n{processed}"
        return result.strip()


def process_lyrics_file(html_content: str, lang_model_path: str = None) -> str:
    """Convenience function to process lyrics HTML."""
    processor = LyricsProcessor(lang_model_path)
    return processor.process_html(html_content)


if __name__ == "__main__":
    example_html = """
    <pre>[Intro]
    
           <b>G</b>        <b>Em7</b>
I found a love for me
               <b>C9</b>
Darling, just dive right in
    </pre>
    """

    processor = LyricsProcessor()
    result = processor.process_html(example_html, detect_lang=False)
    print(result)
