"""
extract_aldwell5.py
═══════════════════════════════════════════════════════════════════════════════
Aldwell, Schachter & Cadwallader — Harmony & Voice Leading, 5th ed. (2019)
→ JSONL Extraction Pipeline

Input: path to the PDF (215-page excerpt in this implementation)

PDF structure
─────────────
  Page size: 576 × 720 pt
  Front matter : PDF pp. 1–14  (Contents, Dedication, Preface, Acknowledgments)
  Main content : PDF pp. 15–190 (Units 1–33, non-contiguously excerpted)
  Back matter  : PDF pp. 191+  (Appendix III, Index of Musical Examples,
                                 Subject Index)

No running headers — chapter identity lives in the BODY of each page:

  Unit opener page  →  "UNIT" alone as first text line, then unit number,
                        then title (FloraStd-Bold@22pt)
  Body content page →  First text line: "BOOK_PAGE  Unit N  Title"
                        (FloraStd-Bold@10 + FloraStd-Medium@8)

Font taxonomy (empirically confirmed)
──────────────────────────────────────
  FloraStd-Bold@22     Unit title on opener page
  FloraStd-Bold@12     "POINTS FOR REVIEW" / "EXERCISES" section markers
  FloraStd-Bold@10     Numbered section headings ("5. Major Scales; …")
                        AND page-number digit in body-page header
  FloraStd-Medium@8    "Unit N Title" text in body-page header (top ~36pt)
  FloraStd-Bold@9.5    Example label ("1-1", "2-3a")
  FloraStd-Medium@9.5  Example caption text ("Mozart, Piano Sonata, K. 545, I")
  NewBaskervilleStd-Roman@10.5  Body prose (primary extraction target)
  NewBaskervilleStd-Italic@10.5 Italic terms within body
  Times-Roman@5        Music superscripts (skip)

Piece attribution format
────────────────────────
  FloraStd-Bold@9.5 label  +  FloraStd-Medium@9.5 text on the same y-line.
  Label: "N-Na" (unit-example number, e.g. "26-11b").
  Text:  "Composer, Work" OR a purely descriptive phrase ("major scale").
  Descriptive captions (no comma, or lowercase first word) → skip.

Special sections
────────────────
  "POINTS FOR REVIEW" (FloraStd-Bold@12) → points_for_review (supplementary)
  "EXERCISES"         (FloraStd-Bold@12) → exercise (application)

Numbered section headings (FloraStd-Bold@10, y > 50) flush the current body
passage and start a new one with an updated section_title. The heading text
itself is retained in the new passage.

Passage types
─────────────
  body             Primary instructional prose (NewBaskervilleStd)
  points_for_review  POINTS FOR REVIEW section
  exercise         EXERCISES section
  chapter_opener   Unit title page + immediately following example
  front_matter     PDF pp. 1–14
  back_matter      PDF pp. 191+

CELL 0 │ Imports
CELL 1 │ ★ USER CONFIG ★
CELL 2 │ Composer vocabulary
CELL 3 │ Layout constants & page utilities
CELL 4 │ Concept extraction
CELL 5 │ Composer & attribution extraction
CELL 6 │ Passage dataclass
CELL 7 │ Core extraction loop
CELL 8 │ Run & write JSONL
CELL 9 │ Diagnostic report
"""

# %% CELL 0 ─ Imports ──────────────────────────────────────────────────────────
from __future__ import annotations
import csv, json, logging, re, sys, unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    import pdfplumber
except ImportError:
    raise ImportError("pip install pdfplumber")

try:
    import spacy
except ImportError:
    raise ImportError("pip install spacy && python -m spacy download en_core_web_lg")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it  # type: ignore

_in_venv = hasattr(sys, "real_prefix") or sys.base_prefix != sys.prefix
print(f"Python  : {sys.executable}")
print(f"In venv : {'✓' if _in_venv else '✗  (activate your virtual environment first)'}")
_DETECTED_SPACY = None
for _m in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
    try:
        spacy.load(_m); _DETECTED_SPACY = _m; break
    except OSError:
        pass
print(f"spaCy   : {_DETECTED_SPACY or '✗  NOT FOUND'}")
print("─" * 60)


# %% CELL 1 ─ ★ USER CONFIG ★ ─────────────────────────────────────────────────
INPUT_PDF_PATH        = Path("aldwell_schachter_5ed.pdf")
COMPOSER_CSV_PATH     = Path("unique_composers.csv")
COMPOSER_ALIASES_PATH = Path("composer_aliases.csv")
OUTPUT_JSONL_PATH     = Path("aldwell_schachter_5ed.jsonl")

BOOK_TITLE   = "Harmony & Voice Leading"
BOOK_EDITION = "5"
BOOK_YEAR    = 2019
BOOK_AUTHORS = ["Edward Aldwell", "Carl Schachter", "Allen Cadwallader"]

SPACY_MODEL           = _DETECTED_SPACY or "en_core_web_sm"
MIN_WORDS_PER_PASSAGE = 30

# Back matter is detected by page content (see _is_back_matter_page below).
# Setting this very high disables the page-number trigger; the full book
# has back matter starting at ~PDF page 730 but exact count varies by edition.
BACK_MATTER_PDF_PAGE = 9999

CONTEXT_ONLY_NAMES: set[str] = {
    "Chicago", "Journey", "Queen", "Canon", "Coda", "variant",
    "Heinrich Schenker",    # theorist, not a composer in the musical-example sense
    "Aqua",                 # false positive from body text
}

SURNAME_DEFAULTS: dict[str, str] = {
    "Bach":          "Johann Sebastian Bach",
    "Mozart":        "Wolfgang Amadeus Mozart",
    "Beethoven":     "Ludwig van Beethoven",
    "Schubert":      "Franz Schubert",
    "Brahms":        "Johannes Brahms",
    "Chopin":        "Frédéric Chopin",
    "Schumann":      "Robert Schumann",
    "Handel":        "George Frideric Handel",
    "Haydn":         "Franz Joseph Haydn",
    "Wagner":        "Richard Wagner",
    "Mendelssohn":   "Felix Mendelssohn",
    "Debussy":       "Claude Debussy",
    "Tchaikovsky":   "Pyotr Ilyich Tchaikovsky",
    "Scarlatti":     "Domenico Scarlatti",
    "Purcell":       "Henry Purcell",
    "Vivaldi":       "Antonio Vivaldi",
    "Corelli":       "Arcangelo Corelli",
    "Palestrina":    "Giovanni Pierluigi da Palestrina",
    "Monteverdi":    "Claudio Monteverdi",
    "Bartók":        "Béla Bartók",
    "Dufay":         "Guillaume Dufay",
    "Schütz":        "Heinrich Schütz",
    "Telemann":      "Georg Philipp Telemann",
    "Rameau":        "Jean Philippe Rameau",
    "Verdi":         "Giuseppe Verdi",
    "Puccini":       "Giacomo Puccini",
    "Fauré":         "Gabriel Fauré",
    "Wolf":          "Hugo Wolf",
    "Mahler":        "Gustav Mahler",
    "Strauss":       "Richard Strauss",
    "Bruckner":      "Anton Bruckner",
    "Liszt":         "Franz Liszt",
    "Franck":        "César Franck",
}

_INITIALS_ALIASES: dict[str, str] = {
    "j. s. bach":   "Johann Sebastian Bach",
    "j.s. bach":    "Johann Sebastian Bach",
    "w. a. mozart": "Wolfgang Amadeus Mozart",
    "w.a. mozart":  "Wolfgang Amadeus Mozart",
    "c. p. e. bach":"Carl Philipp Emanuel Bach",
    "g. f. handel": "George Frideric Handel",
}

CHAPTER_CONCEPT_MAP: dict[str, list[str]] = {
    # Keys MUST be valid entries in CONCEPT_PATTERNS above.
    # Each unit maps to the concepts that are specific to its pedagogical focus.
    "1":  ["major scale", "minor scale", "key signature", "leading tone"],
    "2":  ["interval"],
    "3":  ["meter"],
    "4":  ["triad", "seventh chord", "chord inversion", "figured bass",
           "Roman numeral analysis"],
    "5":  ["counterpoint", "voice leading", "suspension"],
    "6":  ["chorale", "voice leading"],
    "7":  ["chord inversion", "voice leading", "dominant seventh"],
    "8":  ["chord inversion", "six-four chord"],
    "9":  ["dominant seventh", "figured bass"],
    "10": ["cadence", "phrase"],
    "11": ["cadence", "six-four chord", "suspension",
           "nonharmonic tone", "chord inversion"],
    "12": ["cadence", "phrase", "chord inversion", "dominant seventh"],
    "13": ["seventh chord", "suspension", "six-four chord"],
    "14": ["cadence", "dominant seventh", "voice leading"],
    "15": ["modulation", "tonicization", "secondary dominant"],
    "16": ["triad", "leading tone", "chord inversion"],
    "17": ["phrase", "nonharmonic tone", "meter"],
    "18": ["sequence", "tonicization", "chord inversion"],
    "19": ["chord inversion", "counterpoint", "sequence"],
    "20": ["chord inversion", "six-four chord", "suspension"],
    "21": ["nonharmonic tone", "interval"],
    "22": ["suspension", "nonharmonic tone", "pedal point"],
    "23": ["seventh chord", "nonharmonic tone"],
    "24": ["chromaticism", "tonicization"],
    "25": ["seventh chord", "sequence"],
    "26": ["secondary dominant", "chromaticism",
           "dominant seventh", "cross relation"],
    "27": ["modulation", "tonicization", "secondary dominant"],
    "28": ["seventh chord", "suspension", "secondary dominant"],
    "29": ["Neapolitan chord", "chromaticism"],
    "30": ["augmented sixth chord", "chromaticism"],
    "31": ["chromaticism", "Neapolitan chord", "dominant seventh"],
    "32": ["chromaticism", "linear chromaticism", "voice exchange"],
    "33": ["chromaticism", "modulation"],
}

UNIT_TITLES: dict[str, str] = {
    "1":  "Key, Scales, and Modes",
    "2":  "Intervals",
    "3":  "Rhythm and Meter",
    "4":  "Introduction to Triads and Seventh Chords",
    "5":  "Introduction to Counterpoint",
    "6":  "The First Species: Two-Voice Framework",
    "7":  "The Lament Bass and Parallel 6 Chords",
    "8":  "The Cadence",
    "9":  "Melody Harmonization and Figured Bass",
    "10": "The Dominant Seventh",
    "11": "The Cadential 6/4",
    "12": "VII° Chords",
    "13": "V7 in Major: Inversions; the VII7 Chord in Major",
    "14": "Introduction to II and IV",
    "15": "IV and II in Minor Keys",
    "16": "VI and III",
    "17": "Setting Texts; Recitative",
    "18": "Diatonic Modulation",
    "19": "6/3-Chord Techniques",
    "20": "5/6, 6/5, and 6/4 Techniques",
    "21": "Unfolding and Reaching Over",
    "22": "Rhythmic Figuration",
    "23": "Melodic Figuration",
    "24": "Tonicization and Modulation",
    "25": "Binary and Ternary Forms",
    "26": "Applied V and VII",
    "27": "Diatonic Modulation",
    "28": "Seventh Chords with Added Dissonance",
    "29": "The Phrygian II (Neapolitan)",
    "30": "Augmented Sixth Chords",
    "31": "Other Chromatic Chords",
    "32": "Chromatic Voice Exchange",
    "33": "Chromaticism in Larger Contexts",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
print("✓  Cell 1 loaded")


# %% CELL 2 ─ Composer vocabulary ──────────────────────────────────────────────
def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()

def load_aliases(path: Path) -> dict[str, str]:
    if not path.exists():
        log.info("Aliases CSV not found (%s).", path); return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "variant" not in reader.fieldnames: return {}
        for row in reader:
            v, c = row.get("variant","").strip(), row.get("canonical","").strip()
            if v and c: out[_fold(v)] = c
    log.info("Aliases: %d mappings.", len(out)); return out

def load_composers(path: Path) -> list[str]:
    if not path.exists():
        log.warning("Composer CSV not found: %s.", path); return []
    names: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(512); fh.seek(0)
        has_header = "composer" in sample.lower()
        reader = csv.DictReader(fh) if has_header else csv.reader(fh)
        for row in reader:
            n = (row.get("composer","") if has_header else row[0]).strip()
            if n: names.append(n)
    log.info("Composer CSV: %d names.", len(names)); return names

def build_composer_index(
    names: list[str], aliases: dict[str, str], defaults: dict[str, str]
) -> tuple[dict[str, str], Optional[re.Pattern], dict[str, str]]:
    def _norm(s): return re.sub(r"([A-Za-z])\.\s+(?=[A-Za-z]\.)", r"\1.", s)
    def _res(n):  return aliases.get(_fold(_norm(n)), n)
    fold_map: dict[str, str] = {}; rex: set[str] = set(); ordered: list[str] = []
    for raw in names:
        name = _res(raw); f, rf = _fold(_norm(name)), _fold(_norm(raw))
        if f not in fold_map: fold_map[f] = name; ordered.append(name)
        if rf not in fold_map: fold_map[rf] = fold_map[f]
        rex.add(name)
        if raw != name: rex.add(raw)
    surname_map: dict[str, str] = {}
    for canonical in ordered:
        parts = canonical.split()
        if parts:
            sk = _fold(parts[-1])
            if sk not in surname_map: surname_map[sk] = canonical
    for bare, preferred in defaults.items():
        surname_map[_fold(bare)] = preferred
    if not rex: return fold_map, None, surname_map
    escaped = sorted((re.escape(n) for n in rex), key=len, reverse=True)
    regex = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)
    log.info("Composer vocab: %d forms.", len(ordered))
    return fold_map, regex, surname_map

_FOLD_MAP:  dict[str, str]      = {}
_REGEX:     Optional[re.Pattern] = None
_SURNAMES:  dict[str, str]      = {}
_ALIASES:   dict[str, str]      = {}
_CTX_ONLY:  set[str]            = set()
print("✓  Cell 2 loaded")


# %% CELL 3 ─ Layout constants & page utilities ────────────────────────────────

# Font identifiers (base name, without AAAAAB+ prefix)
_FLORA_BOLD   = "FloraStd-Bold"
_FLORA_MED    = "FloraStd-Medium"
_FLORA_ROMAN  = "FloraStd-Roman"
_BASKERV_ROM  = "NewBaskervilleStd-Roman"
_BASKERV_ITA  = "NewBaskervilleStd-Italic"

# Y-coordinate thresholds
HEADER_TOP    = 50.0   # unit label words appear above this y
FOOTER_BOTTOM = 670.0  # copyright footer words appear below this y

# Pages that mark the start of true back matter
_BACK_MATTER_PAGE_RE = re.compile(
    r"^(?:Index of Musical Examples|Subject Index|"  
    r"Appendix\s+(?:I{1,3}|IV|V|VI|1|2|3|4|5)|"  
    r"Roman Numerals and Registers|"  
    r"Systems of Register Designation)"  
    r"(?:\s+\d+)?$"   # optional trailing page number
)

# Cengage copyright boilerplate (appears in footer and on certain pages)
_COPYRIGHT_RE = re.compile(
    r"Copyright\s+20\d\d\s+Cengage|Editorial\s+review\s+has\s+deemed|"
    r"Cengage\s+Learning\s+reserves|suppressed\s+content\s+does\s+not",
    re.IGNORECASE,
)

# Pattern: body page first line = "BOOK_PAGE  Unit N  Title..."
_BODY_HDR_RE = re.compile(
    r"^(\d+)\s+Unit\s+(\d+)\s+(.+)$"
)
# Pattern: numbered section heading = "N. Title text" or "N-N. Title text"
_SEC_HDG_RE = re.compile(r"^\d+(?:\-\d+)?\.\s+\S")

# Example label on a FloraStd@9.5 caption line
_EX_LABEL_RE = re.compile(r"^(\d+-\d+[a-z]?)$")

# Catalog patterns
_CATALOG_RE = re.compile(
    r"\b(?:BWV|WoO|Hob\.|D\.|Z\.|K\.|Op\.|op\.|RV)\s*[\d,/\.]+",
    re.IGNORECASE,
)

# Notation characters
_NOTATION_RE = re.compile(
    r"[\u0300-\u036f\u2190-\u21ff\u2200-\u22ff"
    r"\u2600-\u27ff\U0001D100-\U0001D1FF]"
)

# Roman numeral regex
_ROMAN_RE = re.compile(
    r"\b(?:I{1,3}|IV|VI{0,3}|VII|ii{0,3}|iv|vi{0,2}|vii)"
    r"[°ø○\u00b0]?(?:[6-9]|add\d)?\b"
)

_FIG_BASS_RE = re.compile(
    r"\b(?:figured bass|thoroughbass|basso continuo)\b"
    r"|\b(?:[6-9]/[3-8]|6/4|6/3|7/5/3)\b",
    re.IGNORECASE,
)

_EXAMPLE_NUM_RE = re.compile(r"\bExample\s+(\d+-\d+[a-z]?)\b", re.IGNORECASE)


def base_font(fontname: str) -> str:
    return fontname.split("+")[-1] if "+" in fontname else fontname


def _notation_ratio(text: str) -> float:
    if not text: return 0.0
    return len(_NOTATION_RE.findall(text)) / max(len(text), 1)


def _is_copyright_line(line: str) -> bool:
    return bool(_COPYRIGHT_RE.search(line))


def _parse_page_header(words: list[dict]) -> dict:
    """
    Extract unit info from the top-region words of a body page.
    Returns {"unit": "N", "title": "...", "book_page": N} or {}.
    """
    header_words = sorted(
        [w for w in words if w["top"] < HEADER_TOP],
        key=lambda x: x["x0"]
    )
    if not header_words:
        return {}
    text = " ".join(w["text"] for w in header_words).strip()
    m = _BODY_HDR_RE.match(text)
    if m:
        return {
            "book_page": int(m.group(1)),
            "unit":      m.group(2),
            "title":     m.group(3).strip(),
        }
    return {}


def _extract_captions(words: list[dict]) -> list[tuple[str, str]]:
    """
    Extract all (label, caption_text) pairs from FloraStd@9.5 words.
    Returns list of (example_label, full_caption_text).
    """
    # Gather FloraStd@9.5 words in body region
    flora_words = [
        w for w in words
        if "Flora" in base_font(w.get("fontname", ""))
        and 9.0 <= w.get("size", 0) <= 10.0
        and HEADER_TOP < w["top"] < FOOTER_BOTTOM
    ]
    if not flora_words:
        return []

    # Group by y-line (within 3pt)
    lines_by_y: dict[int, list[dict]] = {}
    for w in flora_words:
        y = round(w["top"] / 3) * 3
        lines_by_y.setdefault(y, []).append(w)

    results: list[tuple[str, str]] = []
    for y in sorted(lines_by_y):
        line_words = sorted(lines_by_y[y], key=lambda x: x["x0"])
        line_text  = " ".join(w["text"] for w in line_words).strip()
        # First token is the label if it matches "N-Na" pattern
        tokens = line_text.split(None, 1)
        if len(tokens) >= 1 and re.match(r"^\d+-\d+[a-z]?$", tokens[0]):
            label   = tokens[0]
            caption = tokens[1].strip() if len(tokens) > 1 else ""
            if caption:
                results.append((label, caption))
    return results


def _find_section_headings(words: list[dict]) -> list[tuple[float, str]]:
    """
    Find inline section headings (FloraStd-Bold@10 in body region).
    Returns list of (y_position, heading_text).
    These are numbered section headings like "5. Major Scales; …"
    """
    # Collect bold body-region words at 10pt
    bold10 = [
        w for w in words
        if base_font(w.get("fontname", "")) == _FLORA_BOLD
        and 9.5 <= w.get("size", 0) <= 10.5
        and HEADER_TOP < w["top"] < FOOTER_BOTTOM
    ]
    if not bold10:
        return []

    # Group into lines
    lines_by_y: dict[int, list[dict]] = {}
    for w in bold10:
        y = round(w["top"] / 2) * 2
        lines_by_y.setdefault(y, []).append(w)

    headings: list[tuple[float, str]] = []
    for y_key in sorted(lines_by_y):
        line_words = sorted(lines_by_y[y_key], key=lambda x: x["x0"])
        text = " ".join(w["text"] for w in line_words).strip()
        # Must look like a numbered section heading: starts with "N." or "N-N."
        if _SEC_HDG_RE.match(text) and len(text.split()) >= 2:
            # Real y position (average of words on this line)
            real_y = sum(w["top"] for w in line_words) / len(line_words)
            headings.append((real_y, text))

    return headings


def _find_section_markers(words: list[dict]) -> list[tuple[float, str]]:
    """
    Detect FloraStd-Bold@12 markers: "POINTS FOR REVIEW" and "EXERCISES".
    Returns list of (y_position, marker_text).
    """
    bold12 = [
        w for w in words
        if base_font(w.get("fontname", "")) == _FLORA_BOLD
        and 11.5 <= w.get("size", 0) <= 12.5
        and w["top"] < FOOTER_BOTTOM
    ]
    if not bold12:
        return []

    lines_by_y: dict[int, list[dict]] = {}
    for w in bold12:
        y = round(w["top"] / 2) * 2
        lines_by_y.setdefault(y, []).append(w)

    markers: list[tuple[float, str]] = []
    for y_key in sorted(lines_by_y):
        line_words = sorted(lines_by_y[y_key], key=lambda x: x["x0"])
        text = " ".join(w["text"] for w in line_words).strip()
        if text in ("POINTS FOR REVIEW", "EXERCISES", "POINTS", "EXERCISES"):
            real_y = sum(w["top"] for w in line_words) / len(line_words)
            # Normalise
            if "POINT" in text:
                markers.append((real_y, "POINTS FOR REVIEW"))
            elif "EXERCISE" in text:
                markers.append((real_y, "EXERCISES"))

    return markers


print("✓  Cell 3 loaded")


# %% CELL 4 ─ Concept extraction ───────────────────────────────────────────────
CONCEPT_PATTERNS: dict[str, list[str]] = {
    "secondary dominant":           [r"secondary dominant", r"applied dominant",
                                     r"applied chord", r"applied V", r"V/V"],
    "secondary leading-tone chord": [r"secondary leading[- ]tone",
                                     r"applied leading[- ]tone", r"applied vii"],
    "Neapolitan chord":             [r"Neapolitan", r"Phrygian II", r"\bII\b.*minor"],
    "augmented sixth chord":        [r"augmented sixth", r"Italian sixth",
                                     r"French sixth", r"German sixth"],
    "modal mixture":                [r"modal mixture", r"mode mixture",
                                     r"borrowed chord"],
    "modulation":                   [r"\bmodulat", r"pivot chord",
                                     r"common[- ]chord modulation",
                                     r"closely related key"],
    "tonicization":                 [r"toniciz", r"transient (?:tonic|modulation)"],
    "sequence":                     [r"\bsequence\b", r"sequential",
                                     r"descending.fifth sequence",
                                     r"ascending.*5th.*sequence"],
    "suspension":                   [r"\bsuspension\b", r"retardation",
                                     r"\b7-6\b", r"\b4-3\b", r"\b9-8\b",
                                     r"\b2-3\b"],
    "nonharmonic tone":             [r"nonchord tone", r"non[- ]chord tone",
                                     r"nonharmonic", r"passing tone",
                                     r"neighbor(?:ing)? tone", r"anticipation",
                                     r"appoggiatura", r"escape tone",
                                     r"pedal (?:point|tone)",
                                     r"incomplete neighbor"],
    "cadence":                      [r"\bcadence\b", r"\bcadential\b",
                                     r"authentic cadence", r"half cadence",
                                     r"deceptive cadence", r"plagal cadence",
                                     r"Phrygian cadence", r"\bPAC\b",
                                     r"\bIAC\b", r"\bHC\b"],
    "phrase":                       [r"\bphrase\b", r"\bperiod\b",
                                     r"antecedent", r"consequent",
                                     r"\bsentence\b", r"hypermeter",
                                     r"phrase rhythm"],
    "voice leading":                [r"voice.lead", r"part.writ",
                                     r"parallel fifth", r"parallel octave",
                                     r"voice crossing", r"\bdoubling\b",
                                     r"contrary motion", r"oblique motion"],
    "counterpoint":                 [r"counterpoint", r"contrapuntal",
                                     r"species counterpoint", r"cantus firmus"],
    "figured bass":                 [r"figured bass", r"thoroughbass",
                                     r"basso continuo"],
    "chord inversion":              [r"chord inversion", r"first inversion",
                                     r"second inversion", r"root position",
                                     r"\b6/4\b", r"6 chord",
                                     r"position of the (?:chord|bass)"],
    "six-four chord":               [r"six.four", r"cadential 6", r"\b6/4\b"],
    "triad":                        [r"\btriad\b"],
    "seventh chord":                [r"seventh chord"],
    "dominant seventh":             [r"dominant seventh", r"\bV7\b",
                                     r"dominant-seventh"],
    "leading tone":                 [r"leading.tone", r"\bLT\b"],
    "interval":                     [r"\binterval\b", r"consonan",
                                     r"dissonan", r"tritone",
                                     r"overtone series", r"half.step",
                                     r"whole.step", r"semitone",
                                     r"augmented", r"diminished",
                                     r"numerical size"],
    "Roman numeral analysis":       [r"Roman numeral"],
    "major scale":                  [r"major scale", r"major key"],
    "minor scale":                  [r"minor scale", r"minor key",
                                     r"harmonic minor", r"melodic minor"],
    "key signature":                [r"key signature", r"circle of fifths"],
    "meter":                        [r"\bmeter\b", r"time signature",
                                     r"syncopation", r"hemiola",
                                     r"simple (?:meter|time)",
                                     r"compound (?:meter|time)", r"hypermeter",
                                     r"\baccent\b", r"\bdownbeat\b",
                                     r"bar line", r"metrical"],
    "binary form":                  [r"binary form", r"rounded binary"],
    "chorale":                      [r"\bchorale\b", r"\bSATB\b",
                                     r"four.part", r"vocal range",
                                     r"voice crossing", r"\bspacing\b"],
    "ternary form":                 [r"ternary form"],
    "sonata form":                  [r"sonata form"],
    "unfolding":                    [r"\bunfolding\b"],
    "voice exchange":               [r"voice exchange"],
    "linear chromaticism":          [r"linear chromatic", r"chromatic voice exchange"],
    "cross relation":               [r"cross.relation", r"false relation"],
    "chromaticism":                 [r"chromatic(?:ism)?"],
    "text painting":                [r"text.paint", r"word.paint",
                                     r"madrigalism", r"tone.paint"],
    "harmonic rhythm":              [r"harmonic rhythm"],
    "pedal point":                  [r"pedal (?:point|tone)", r"\bpedal\b"],
}

_COMPILED = [
    (c, re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE))
    for c, pats in CONCEPT_PATTERNS.items()
]


def extract_concepts(text: str) -> list[str]:
    return sorted({c for c, pat in _COMPILED if pat.search(text)})


def get_chapter_concepts(unit: str) -> list[str]:
    return list(CHAPTER_CONCEPT_MAP.get(unit, []))


print("✓  Cell 4 loaded")


# %% CELL 5 ─ Composer & attribution extraction ────────────────────────────────
_POSS_RE  = re.compile(r"'s\s*$")
_TRAIL_RE = re.compile(r"[,:\s]+$")

_ATTR_STOP = frozenset({
    "example", "see", "the", "a", "an", "in", "this", "that",
    "these", "as", "on", "at", "both", "also", "note", "listen",
})

# Descriptive captions (no piece attribution)
_DESCRIPTIVE_RE = re.compile(
    r"^(registers|intervals|scales|modes|chords|functions|"
    r"voice|parallel|ascending|descending|stable|active|"
    r"resolving|passing|neighboring|suspensions|anticipation"
    r"|chromatic|diatonic|consonant|dissonant|inverted|"
    r"diminished|augmented|major|minor|enharmonic|"
    r"successive|applied|neighboring|passing|sequences|"
    r"[a-z])",  # starts with lowercase
    re.IGNORECASE
)


def _clean_name(raw: str) -> str:
    return _TRAIL_RE.sub("", _POSS_RE.sub("", raw.strip())).strip()


def _canon(raw: str) -> str:
    f = _fold(raw)
    return _ALIASES.get(f) or _FOLD_MAP.get(f) or raw


def _canonicalize_composer(raw: str) -> Optional[str]:
    if not raw or not raw[0].isupper(): return None
    lower = raw.lower().strip()
    canon = _INITIALS_ALIASES.get(lower)
    if not canon:
        f     = _fold(re.sub(r"([A-Za-z])\. (?=[A-Za-z]\.)", r"\1.", raw))
        canon = _ALIASES.get(f) or _FOLD_MAP.get(f)
    if not canon:
        parts = raw.split()
        if parts:
            canon = (_SURNAMES.get(_fold(parts[-1]))
                     or SURNAME_DEFAULTS.get(parts[-1]))
    return canon


def _parse_caption(label: str, caption_text: str) -> Optional[dict]:
    """
    Parse a FloraStd@9.5 example caption into an attribution dict.
    Returns None for descriptive (non-piece) captions.
    """
    ct = caption_text.strip()
    if not ct:
        return None

    # Skip obviously descriptive captions
    if ct[0].islower():
        return None
    first_word = ct.split()[0].lower().rstrip(".,")
    if first_word in _ATTR_STOP:
        return None

    # Split at first comma → composer / work
    comma = ct.find(",")
    if 0 < comma <= 45:
        composer_raw = ct[:comma].strip()
        work_raw     = ct[comma+1:].strip()
    else:
        # No comma or comma too far — might just be a composer name or descriptive
        return None

    canon = _canonicalize_composer(composer_raw)
    if not canon or canon in _CTX_ONLY:
        return None

    cat_m   = _CATALOG_RE.search(work_raw)
    catalog = re.sub(r"[,\s]+$", "", cat_m.group(0)).strip() if cat_m else ""
    work    = _CATALOG_RE.sub("", work_raw)
    work    = re.sub(r"\bmm?\.\s*[\d\u2013\u2014\-]+", "", work)
    work    = re.sub(r",\s*,", ",", work)
    work    = re.sub(r",([\"'\u201d])\s*$", r"\1", work)
    work    = re.sub(r"[,\s;.]+$", "", re.sub(r"\s{2,}", " ", work).strip()).strip()

    return {"example_no": label, "composer": canon, "work": work, "catalog": catalog}


def _vocab(text: str) -> list[str]:
    if not _REGEX: return []
    return sorted({_canon(_clean_name(m)) for m in _REGEX.findall(text)}
                  - _CTX_ONLY)


def _ner(text: str, nlp) -> list[str]:
    if not _REGEX or nlp is None: return []
    cands: set[str] = set()
    for ent in nlp(text[:4000]).ents:
        if ent.label_ != "PERSON": continue
        if not _REGEX.search(ent.text): continue
        raw = _clean_name(ent.text.strip())
        if not raw: continue
        canon = _canon(raw)
        if _fold(canon) in _FOLD_MAP or _fold(raw) in _FOLD_MAP:
            cands.add(canon)
    return sorted(cands - _CTX_ONLY)


print("✓  Cell 5 loaded")


# %% CELL 6 ─ Passage dataclass ────────────────────────────────────────────────
_PTYPE_ROLE: dict[str, str] = {
    "points_for_review": "supplementary",
    "exercise":          "application",
    "chapter_opener":    "supplementary",
    "front_matter":      "supplementary",
    "back_matter":       "supplementary",
}


def _passage_role(ptype: str, unit: str,
                   concepts: Optional[list[str]] = None) -> str:
    if ptype in _PTYPE_ROLE: return _PTYPE_ROLE[ptype]
    cc = CHAPTER_CONCEPT_MAP.get(unit, [])
    if not cc: return "central"
    hits = sum(1 for c in cc if c in set(concepts or []))
    return "central" if hits / len(cc) >= 0.25 else "supplementary"


@dataclass
class Passage:
    book_title:   str      = ""
    book_edition: str      = ""
    book_year:    int|None = None
    book_authors: list[str] = field(default_factory=list)

    chapter_number: str      = ""
    chapter_title:  str      = ""
    section_number: str      = ""
    section_title:  str      = ""
    page_start:     int|None = None
    page_stop:      int|None = None

    body_text:      str   = ""
    passage_type:   str   = "body"
    notation_ratio: float = 0.0

    concepts_passage: list[str] = field(default_factory=list)
    concepts_chapter: list[str] = field(default_factory=list)

    composers_vocab:       list[str] = field(default_factory=list)
    composers_ner:         list[str] = field(default_factory=list)
    composers_all_sources: list[str] = field(default_factory=list)

    has_example_reference: bool      = False
    example_numbers:       list[str] = field(default_factory=list)
    piece_attributions:    list[dict] = field(default_factory=list)

    passage_role: str = "central"

    word_count:              int  = 0
    sentence_count:          int  = 0
    char_count:              int  = 0
    contains_roman_numerals: bool = False
    contains_figured_bass:   bool = False

    def to_jsonl_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


print("✓  Cell 6 loaded")


# %% CELL 7 ─ Core extraction loop ─────────────────────────────────────────────

def _make_passage(
    lines: list[str], ptype: str, unit: str, unit_title: str,
    sec_num: str, sec_title: str,
    page_start: Optional[int], page_stop: Optional[int],
    piece_attrs: list[dict], nlp
) -> Optional[Passage]:
    text = " ".join(l for l in lines if l.strip())
    text = re.sub(r"\s+", " ", text).strip()
    words_list = text.split()
    if len(words_list) < MIN_WORDS_PER_PASSAGE:
        return None
    if not unit and ptype not in ("front_matter", "back_matter"):
        ptype = "front_matter"

    ex_nums  = list(dict.fromkeys(_EXAMPLE_NUM_RE.findall(text)))
    cp       = extract_concepts(text)
    comp_v   = _vocab(text)
    comp_n   = _ner(text, nlp)
    fig_comps = list(dict.fromkeys(
        a["composer"] for a in piece_attrs if a.get("composer")
    ))
    comp_all = list(dict.fromkeys(comp_v + comp_n + fig_comps))

    return Passage(
        book_title    = BOOK_TITLE,
        book_edition  = BOOK_EDITION,
        book_year     = BOOK_YEAR,
        book_authors  = list(BOOK_AUTHORS),
        chapter_number = unit,
        chapter_title  = unit_title,
        section_number = sec_num,
        section_title  = sec_title,
        page_start    = page_start,
        page_stop     = page_stop,
        body_text     = text,
        passage_type  = ptype,
        notation_ratio = round(_notation_ratio(text), 4),
        concepts_passage      = cp,
        concepts_chapter      = get_chapter_concepts(unit),
        composers_vocab       = comp_v,
        composers_ner         = comp_n,
        composers_all_sources = comp_all,
        has_example_reference = bool(ex_nums),
        example_numbers       = ex_nums,
        piece_attributions    = piece_attrs,
        passage_role  = _passage_role(ptype, unit, cp),
        word_count     = len(words_list),
        sentence_count = len(re.split(r"(?<=[.!?])\s+", text)),
        char_count     = len(text),
        contains_roman_numerals = bool(_ROMAN_RE.search(text)),
        contains_figured_bass   = bool(_FIG_BASS_RE.search(text)),
    )


def extract_passages_from_pdf(pdf_path: Path, nlp) -> list[Passage]:
    """
    Main extraction loop.

    Per-page strategy:
    1. Extract words with font/size metadata.
    2. Detect unit from header-region words (FloraStd@8, y < 50).
       Unit opener pages are identified by first text line = "UNIT".
    3. Build caption lookup from FloraStd@9.5 words → piece attributions.
    4. Detect section headings (FloraStd-Bold@10, y > 50) → flush + new section.
    5. Detect POINTS FOR REVIEW / EXERCISES (FloraStd-Bold@12) → passage type.
    6. Extract body text via extract_text(); strip page header + copyright footer.
    7. Accumulate body lines; flush at section boundaries.
    """
    all_passages: list[Passage] = []

    # ── Persistent state ─────────────────────────────────────────────────────
    cur_unit       = ""
    cur_unit_title = ""
    cur_ptype      = "front_matter"
    cur_sec_ctr    = 0
    cur_sec_title  = ""
    cur_lines:  list[str]  = []
    cur_attrs:  list[dict] = []
    cur_page_start: Optional[int] = None
    cur_page_stop:  Optional[int] = None

    def flush():
        nonlocal cur_lines, cur_attrs, cur_page_start, cur_page_stop
        sec_num = f"{cur_unit}.{cur_sec_ctr}" if cur_unit and cur_sec_ctr else ""
        p = _make_passage(
            cur_lines, cur_ptype, cur_unit, cur_unit_title,
            sec_num, cur_sec_title,
            cur_page_start, cur_page_stop,
            list(cur_attrs), nlp,
        )
        if p: all_passages.append(p)
        cur_lines.clear(); cur_attrs.clear()
        cur_page_start = cur_page_stop = None

    log.info("Opening PDF: %s", pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        log.info("PDF: %d pages.", total)

    # Process in batches to control memory
    for batch_start in tqdm(range(0, total, 20), desc="Extracting", unit="batch"):
        batch_end = min(batch_start + 20, total)
        with pdfplumber.open(pdf_path) as pdf:
            for idx in range(batch_start, batch_end):
                pdf_page_num = idx + 1

                page  = pdf.pages[idx]
                words = page.extract_words(extra_attrs=["fontname", "size"])
                text  = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                raw_lines = [l.strip() for l in text.split("\n") if l.strip()]

                # ── Back-matter gate: page number OR first-line content ────────
                _first_line = raw_lines[0] if raw_lines else ""
                _content_is_back = bool(_BACK_MATTER_PAGE_RE.match(_first_line))

                if pdf_page_num >= BACK_MATTER_PDF_PAGE or _content_is_back:
                    if cur_ptype != "back_matter":
                        flush()
                        cur_ptype      = "back_matter"
                        cur_unit       = ""
                        cur_unit_title = ""
                    for line in raw_lines:
                        if not _is_copyright_line(line):
                            cur_lines.append(line)
                    cur_page_start = cur_page_start or pdf_page_num
                    continue

                # ── Unit opener? ─────────────────────────────────────────────
                if raw_lines and raw_lines[0] == "UNIT":
                    flush()
                    # Next lines: unit number, then title words
                    unit_num = raw_lines[1].strip() if len(raw_lines) > 1 else ""
                    title_parts = []
                    for l in raw_lines[2:]:
                        if re.match(r"^\d+-\d+", l) or _is_copyright_line(l):
                            break
                        if l in ("Listen to the audio of all the unit's examples.",
                                 "Practice the workbook's exercises, available in PDF."):
                            break
                        title_parts.append(l)
                    unit_title = " ".join(title_parts).strip()
                    if unit_num.isdigit():
                        cur_unit       = unit_num
                        cur_unit_title = UNIT_TITLES.get(unit_num, unit_title)
                        cur_sec_ctr    = 0
                        cur_sec_title  = ""
                        cur_ptype      = "chapter_opener"
                    # Opener content goes into chapter_opener passage
                    body_lines = raw_lines[2:]
                    for l in body_lines:
                        if _is_copyright_line(l): continue
                        cur_lines.append(l)
                    if cur_page_start is None: cur_page_start = pdf_page_num
                    cur_page_stop = pdf_page_num
                    continue

                # ── Parse page header to update unit tracking ─────────────────
                hdr = _parse_page_header(words)
                if hdr.get("unit"):
                    new_unit = hdr["unit"]
                    if new_unit != cur_unit:
                        flush()
                        cur_unit       = new_unit
                        cur_unit_title = UNIT_TITLES.get(new_unit,
                                          hdr.get("title", ""))
                        cur_sec_ctr    = 0
                        cur_sec_title  = ""
                        cur_ptype      = "body"
                    elif cur_ptype == "chapter_opener":
                        flush()
                        cur_ptype = "body"

                book_page = hdr.get("book_page")

                # ── Captions → attributions ───────────────────────────────────
                page_captions = _extract_captions(words)
                page_attrs: list[dict] = []
                seen_attr_keys: set[str] = set()
                for label, cap_text in page_captions:
                    attr = _parse_caption(label, cap_text)
                    if attr:
                        key = f"{attr['composer']}|{attr['work'][:30]}"
                        if key not in seen_attr_keys:
                            seen_attr_keys.add(key)
                            page_attrs.append(attr)

                # ── Section markers: POINTS FOR REVIEW / EXERCISES ────────────
                sec_markers = _find_section_markers(words)
                # Use the FIRST marker on the page to determine the split
                # (if multiple, process in y-order below)

                # ── Section headings (numbered paragraph headings) ─────────────
                sec_headings = _find_section_headings(words)

                # ── Build y-sorted event list ─────────────────────────────────
                # We process events in top-to-bottom order:
                # EVENTS: section_heading(y, text), section_marker(y, type)
                events: list[tuple[float, str, str]] = []
                for y, text in sec_markers:
                    events.append((y, "MARKER", text))
                for y, text in sec_headings:
                    events.append((y, "HEADING", text))
                events.sort(key=lambda x: x[0])

                # ── Process body text ──────────────────────────────────────────
                # Strip first line (page header) and copyright footer lines.
                body_lines: list[str] = []
                skip_first = bool(hdr)  # first line is "PAGE Unit N Title"
                for i2, line in enumerate(raw_lines):
                    if i2 == 0 and skip_first:
                        continue
                    if _is_copyright_line(line):
                        continue
                    # Also skip "Listen to audio..." prompts
                    if line.startswith(("Listen to the audio",
                                        "Practice the workbook")):
                        continue
                    body_lines.append(line)

                # Integrate events with body_lines accumulation
                # Because we can't perfectly correlate y-positions with line indices
                # (pdfplumber extract_text() doesn't preserve y), we use a simple
                # approach: flush when we see the marker/heading TEXT in body_lines.
                event_idx = 0
                for line in body_lines:
                    # Check if this line is a section marker
                    if line in ("POINTS FOR REVIEW", "EXERCISES"):
                        flush()
                        cur_ptype = ("points_for_review"
                                     if "POINT" in line else "exercise")
                        cur_page_start = book_page
                        cur_attrs.extend(page_attrs)
                        page_attrs = []
                        continue

                    # Check if this line matches a numbered section heading
                    if _SEC_HDG_RE.match(line) and cur_ptype == "body":
                        flush()
                        cur_sec_ctr  += 1
                        # Extract heading up to body prose start.
                        heading_m = re.match(r"^\d+(?:\-\d+)?\.\s+(.+)$", line)
                        heading_full = heading_m.group(1).strip() if heading_m else line
                        end_m = re.search(r"\.(?=\s+[A-Z][a-z])", heading_full)
                        if end_m:
                            cur_sec_title = heading_full[:end_m.start()].strip()
                        else:
                            # Hard cap: take up to first sentence break
                            first_period = heading_full.find(". ")
                            cap = first_period if 0 < first_period < 80 else 80
                            cur_sec_title = heading_full[:cap].strip()
                        cur_page_start = book_page

                    cur_lines.append(line)
                    if cur_page_start is None: cur_page_start = book_page
                    cur_page_stop = book_page

                # Merge page attributions into cur_attrs
                for attr in page_attrs:
                    key = f"{attr['composer']}|{attr['work'][:30]}"
                    if not any(f"{a['composer']}|{a['work'][:30]}" == key
                               for a in cur_attrs):
                        cur_attrs.append(attr)

    flush()
    log.info("Extraction complete: %d passages.", len(all_passages))
    return all_passages


print("✓  Cell 7 loaded")


# %% CELL 8 ─ Run & write JSONL ────────────────────────────────────────────────
def main() -> list[Passage]:
    global _ALIASES, _FOLD_MAP, _REGEX, _SURNAMES, _CTX_ONLY

    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_PDF_PATH
    if not pdf_path.exists():
        cand = Path("/mnt/user-data/uploads") / pdf_path.name
        if cand.exists(): pdf_path = cand
        else: log.error("PDF not found: %s", pdf_path); return []

    alias_path = COMPOSER_ALIASES_PATH
    if not alias_path.exists():
        c = Path("/mnt/user-data/outputs") / alias_path.name
        if c.exists(): alias_path = c

    csv_path = COMPOSER_CSV_PATH
    if not csv_path.exists():
        c = Path("/mnt/user-data/outputs") / csv_path.name
        if c.exists(): csv_path = c

    _CTX_ONLY = set(CONTEXT_ONLY_NAMES)
    _ALIASES  = load_aliases(alias_path)
    csv_names = load_composers(csv_path)
    _FOLD_MAP, _REGEX, _SURNAMES = build_composer_index(
        csv_names, _ALIASES, SURNAME_DEFAULTS
    )

    log.info("Loading spaCy model: %s", SPACY_MODEL)
    try:
        nlp = spacy.load(SPACY_MODEL)
        nlp.select_pipes(enable=["tok2vec", "ner"])
    except Exception as e:
        log.warning("spaCy load failed (%s) — NER disabled.", e)
        nlp = None

    return extract_passages_from_pdf(pdf_path, nlp)


passages = main()
print(f"\n✓  Extraction complete — {len(passages)} passages")

OUTPUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_JSONL_PATH.open("w", encoding="utf-8") as fh:
    for p in passages:
        fh.write(p.to_jsonl_line() + "\n")
log.info("Wrote %d records → %s", len(passages), OUTPUT_JSONL_PATH)


# %% CELL 9 ─ Diagnostic report ────────────────────────────────────────────────
from collections import Counter

def print_summary(passages: list[Passage]) -> None:
    if not passages:
        print("  ⚠  No passages extracted."); return
    sep = "═" * 64
    print(f"\n{sep}")
    print(f"  Harmony & Voice Leading 5th ed. (Aldwell et al.) — summary")
    print(sep)
    print(f"  Total passages : {len(passages)}")

    pt = Counter(p.passage_type for p in passages)
    print(f"\n  Passage types:")
    for t, n in pt.most_common():
        print(f"    {t:<22} {n:>5,}")

    pr = Counter(p.passage_role for p in passages)
    print(f"\n  Passage roles:")
    for r, n in pr.most_common():
        print(f"    {r:<15} {n:>5,}  ({n/len(passages)*100:.1f}%)")

    un = Counter(p.chapter_number for p in passages if p.chapter_number)
    covered = sorted(un.keys(), key=lambda x: int(x) if x.isdigit() else 99)
    print(f"\n  Units covered: {covered}")
    for u in covered:
        title = UNIT_TITLES.get(u, "")[:38]
        n     = un[u]
        rn    = sum(1 for p in passages if p.chapter_number==u and p.contains_roman_numerals)
        print(f"    Unit {u:<3}  {n:>3}p  {title:<38}  rn={rn}")

    comp_counts = Counter(c for p in passages for c in p.composers_all_sources)
    print(f"\n  Top 20 composers (unique: {len(comp_counts)}):")
    for name, n in comp_counts.most_common(20):
        print(f"    {name:<42} {n:>4}x")

    attrs = [a for p in passages for a in p.piece_attributions]
    body  = [p for p in passages if p.passage_type == "body"]
    w_conc = sum(1 for p in body if p.concepts_passage)
    wcs    = [p.word_count for p in passages]

    print(f"\n  Total piece attributions        : {len(attrs)}")
    print(f"  Body passages with ≥1 concept   : {w_conc}/{len(body)} "
          f"({100*w_conc/max(len(body),1):.0f}%)")
    print(f"  Word count: min={min(wcs)}  max={max(wcs)}  mean={sum(wcs)/len(wcs):.0f}")
    print(f"\n  JSONL → {OUTPUT_JSONL_PATH}")
    print(f"{sep}\n")

print_summary(passages)
