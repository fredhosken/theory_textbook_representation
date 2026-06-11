"""
extract_burstein.py
═══════════════════════════════════════════════════════════════════════════════
Burstein & Straus — Concise Introduction to Tonal Harmony, 3rd ed. (2025)
→ JSONL Extraction Pipeline

Input: path to the Markdown export of the EPUB (burstein_concise_harmony_3ed.md)

Markdown structure
──────────────────
  <!-- spine: N -->  marks each EPUB spine item (377 total)
  # chapter N        chapter number heading
  # SECTION TITLE    section heading (ALL CAPS H1) within chapter
  # POINTS FOR REVIEW  end-of-chapter review list (supplementary)
  # TEST YOURSELF      end-of-chapter exercises (application)
  # A Closer Look      extended coverage sidebar (supplementary)
  ## Glossary          inline glossary block inside body/section spines
  > More information   alt-text describing a musical score image — STRIP
  N.N[a-d]             bare example-number label on a line — STRIP
  ---                  HR spine separator

  Back matter (block 372+): Test Yourself Answers, Glossary, Credits,
    Index of Music Examples, Index of Terms and Concepts → back_matter

48 chapters: 0 (Notation of Pitch and Rhythm) through 47 (Form)

Piece attribution sources
──────────────────────────
  1. Credits block — 11 copyrighted works with format:
       "**N.N: "Work" by Composer.**" or "**N.N: Composer, "Work"**"
     → chapter number extracted from N prefix (e.g. "40.15" → chapter 40)
  2. Index of Music Examples — comprehensive composer/work list with book-page
     numbers; used to enrich composer vocabulary but not directly linked
     to specific passages (page-to-chapter mapping is approximate)

Passage types
─────────────
  body             Primary instructional prose (section blocks)
  points_for_review  POINTS FOR REVIEW end-of-chapter list (supplementary)
  test_yourself    TEST YOURSELF exercises (application)
  closer_look      A Closer Look extended sidebar (supplementary)
  glossary         ## Glossary sections within body blocks (supplementary)
  chapter_opener   # chapter N opener block
  front_matter     Contents, Preface, etc.
  back_matter      Test Yourself Answers, Glossary, Credits, Indexes

CELL 0 │ Imports
CELL 1 │ ★ USER CONFIG ★
CELL 2 │ Composer vocabulary
CELL 3 │ Markdown parsing utilities
CELL 4 │ Concept extraction & 48-chapter map
CELL 5 │ Attribution extraction from Credits + Index
CELL 6 │ Passage dataclass
CELL 7 │ Core extraction loop
CELL 8 │ Run & write JSONL
CELL 9 │ Diagnostic report
"""

# %% CELL 0 — Imports ─────────────────────────────────────────────────────────
from __future__ import annotations
import csv, json, logging, re, sys, unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

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


# %% CELL 1 — ★ USER CONFIG ★ ─────────────────────────────────────────────────
INPUT_MD_PATH         = Path("burstein_concise_harmony_3ed.md")
COMPOSER_CSV_PATH     = Path("unique_composers.csv")
COMPOSER_ALIASES_PATH = Path("composer_aliases.csv")
OUTPUT_JSONL_PATH     = Path("burstein_straus_3ed.jsonl")

BOOK_TITLE   = "Concise Introduction to Tonal Harmony"
BOOK_EDITION = "3"
BOOK_YEAR    = 2025
BOOK_AUTHORS = ["L. Poundie Burstein", "Joseph N. Straus"]

SPACY_MODEL           = _DETECTED_SPACY or "en_core_web_sm"
MIN_WORDS_PER_PASSAGE = 30

# Spine-block index (0-based) at which back matter begins.
# Block 372 = "# Test Yourself Answers" — first back-matter block.
BACK_MATTER_BLOCK = 372

CONTEXT_ONLY_NAMES: set[str] = {
    "Chicago", "Journey", "Queen", "Canon", "Coda", "variant",
    "Heinrich Schenker",
}

SURNAME_DEFAULTS: dict[str, str] = {
    "Bach":         "Johann Sebastian Bach",
    "Mozart":       "Wolfgang Amadeus Mozart",
    "Beethoven":    "Ludwig van Beethoven",
    "Schubert":     "Franz Schubert",
    "Brahms":       "Johannes Brahms",
    "Chopin":       "Frédéric Chopin",
    "Schumann":     "Robert Schumann",
    "Handel":       "George Frideric Handel",
    "Haydn":        "Franz Joseph Haydn",
    "Wagner":       "Richard Wagner",
    "Mendelssohn":  "Felix Mendelssohn",
    "Debussy":      "Claude Debussy",
    "Bartók":       "Béla Bartók",
    "Schoenberg":   "Arnold Schoenberg",
    "Webern":       "Anton Webern",
    "Berg":         "Alban Berg",
    "Stravinsky":   "Igor Stravinsky",
    "Crawford":     "Ruth Crawford Seeger",
    "Seeger":       "Ruth Crawford Seeger",
    "Adès":         "Thomas Adès",
    "Crumb":        "George Crumb",
    "Lutyens":      "Elisabeth Lutyens",
    "Dun":          "Tan Dun",
    "Schnittke":    "Alfred Schnittke",
}

_INITIALS_ALIASES: dict[str, str] = {
    "j. s. bach":     "Johann Sebastian Bach",
    "j.s. bach":      "Johann Sebastian Bach",
    "w. a. mozart":   "Wolfgang Amadeus Mozart",
    "w.a. mozart":    "Wolfgang Amadeus Mozart",
    "g. f. handel":   "George Frideric Handel",
    "g.f. handel":    "George Frideric Handel",
    "c. p. e. bach":  "Carl Philipp Emanuel Bach",
    "j. m. nunes garcia": "José Maurício Nunes Garcia",
    "r. nathaniel dett":  "Robert Nathaniel Dett",
}

CHAPTER_TITLES: dict[str, str] = {
    "0":  "Notation of Pitch and Rhythm",
    "1":  "Scales",
    "2":  "Intervals",
    "3":  "Triads and Seventh Chords",
    "4":  "Four-Part Harmony",
    "5":  "Voice Leading",
    "6":  "Harmonic Progression",
    "7":  "Melodic Elaboration",
    "8":  "Species Counterpoint",
    "9":  "I and V",
    "10": "The Dominant Seventh Chord: V7",
    "11": "I6 and V6",
    "12": "V65 and V42",
    "13": "V43 and vii°6",
    "14": "Approaching the Dominant: IV, ii6, and ii65",
    "15": "Embellishing V: Cadential 64",
    "16": "Leading to the Tonic: IV",
    "17": "The Leading-Tone Seventh Chord: vii°7 and viiø7",
    "18": "Approaching V: IV6, ii, ii7, and IV7",
    "19": "Multiple Functions: VI",
    "20": "Voice Leading with Embellishing Tones",
    "21": "III and VII",
    "22": "Sequences",
    "23": "Other 64 Chords",
    "24": "Other Embellishing Chords",
    "25": "Applied Dominants of V",
    "26": "Other Applied Chords",
    "27": "Modulation to the Dominant Key",
    "28": "Modulation to Closely Related Keys",
    "29": "Modal Mixture",
    "30": "♭II6: The Neapolitan Sixth",
    "31": "Augmented Sixth Chords",
    "32": "Other Chromatically Altered Chords",
    "33": "Chromatic Sequences",
    "34": "Chromatic Modulation",
    "35": "Sentences and Other Phrase Types",
    "36": "Periods and Other Phrase Pairs",
    "37": "Binary Form",
    "38": "Ternary and Rondo Forms",
    "39": "Sonata Form",
    "40": "Collections and Scales I: Diatonic and Pentatonic",
    "41": "Collections and Scales II: Octatonic, Hexatonic, and Whole-Tone",
    "42": "Triadic Post-Tonality",
    "43": "Intervals (Post-Tonal)",
    "44": "Pitch-Class Sets: Trichords",
    "45": "Inversional Symmetry",
    "46": "Twelve-Tone Serialism",
    "47": "Form (Post-Tonal)",
}

CHAPTER_CONCEPT_MAP: dict[str, list[str]] = {
    "0":  ["pitch", "meter", "rhythm"],
    "1":  ["major scale", "minor scale", "key signature"],
    "2":  ["interval"],
    "3":  ["triad", "seventh chord", "chord inversion", "figured bass",
           "Roman numeral analysis"],
    "4":  ["chorale", "voice leading"],
    "5":  ["voice leading"],
    "6":  ["harmonic progression", "harmonic rhythm"],
    "7":  ["nonharmonic tone", "passing tone", "neighboring tone"],
    "8":  ["counterpoint"],
    "9":  ["cadence", "tonic", "dominant"],
    "10": ["dominant seventh", "voice leading"],
    "11": ["chord inversion"],
    "12": ["dominant seventh", "chord inversion"],
    "13": ["leading-tone triad", "chord inversion"],
    "14": ["subdominant", "supertonic"],
    "15": ["six-four chord"],
    "16": ["subdominant"],
    "17": ["leading-tone seventh chord"],
    "18": ["subdominant", "supertonic", "seventh chord"],
    "19": ["submediant", "deceptive cadence"],
    "20": ["suspension", "nonharmonic tone"],
    "21": ["mediant"],
    "22": ["sequence"],
    "23": ["six-four chord"],
    "24": ["nonharmonic tone"],
    "25": ["secondary dominant", "tonicization"],
    "26": ["secondary dominant", "tonicization"],
    "27": ["modulation"],
    "28": ["modulation"],
    "29": ["modal mixture"],
    "30": ["Neapolitan chord"],
    "31": ["augmented sixth chord"],
    "32": ["chromaticism"],
    "33": ["sequence", "chromaticism"],
    "34": ["modulation", "chromaticism"],
    "35": ["phrase", "sentence"],
    "36": ["phrase", "period"],
    "37": ["binary form"],
    "38": ["ternary form"],
    "39": ["sonata form"],
    "40": ["pentatonic scale", "world music", "mode"],
    "41": ["octatonic scale", "whole-tone scale"],
    "42": ["chromaticism"],
    "43": ["interval", "pitch class"],
    "44": ["pitch class", "set theory"],
    "45": ["set theory"],
    "46": ["twelve-tone"],
    "47": ["sonata form", "binary form", "ternary form"],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
print("✓  Cell 1 loaded")


# %% CELL 2 — Composer vocabulary ─────────────────────────────────────────────
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
    _GEN_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "2nd", "3rd", "4th"}
    for canonical in ordered:
        parts = canonical.split()
        if parts:
            last = parts[-1].rstrip(".")
            # Skip generational suffixes as surname keys; use prior word
            if _fold(last) in _GEN_SUFFIXES and len(parts) >= 2:
                sk = _fold(parts[-2].rstrip("."))
            else:
                sk = _fold(last)
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


# %% CELL 3 — Markdown parsing utilities ──────────────────────────────────────

_SPINE_SEP       = re.compile(r"\n<!-- spine: \d+ -->\n")
_CHAPTER_H1      = re.compile(r"^# chapter (\d+)$", re.MULTILINE | re.IGNORECASE)
_H1              = re.compile(r"^# (.+)$")
_H2              = re.compile(r"^## (.+)$")
_BOLD_ITALIC     = re.compile(r"\*{1,3}([^*]+)\*{1,3}")
_MORE_INFO_BLOCK = re.compile(r"> More information\n.*?(?=\n\n|\Z)", re.DOTALL)
_BARE_EX_LABEL   = re.compile(r"^\d+\.\d+[a-d]?(?:–\d+[a-d]?)?$")
_EX_REF          = re.compile(r"\bExample\s+(\d+[.\-–]\d+[a-d]?)\b", re.IGNORECASE)

# Attribution anchored by EXPAND N.NCLOSE N.N markers.
# Examples appear as: body_text + "N.N Composer, Work" + newline + EXPAND N.N CLOSE N.N
# (the attribution runs directly onto the end of a sentence without extra whitespace)
_EXPAND_ATTR_RE  = re.compile(
    r"(\d+\.\d+[a-z]?)\s+"          # example number
    r"([A-Z\u00c0-\u024f\u0100-\u024f][^\n]+?)"  # attribution (uppercase start)
    r"(?:\n|\nEXPAND)"               # ends at newline before EXPAND
)
_EXPAND_ANY_RE   = re.compile(r"EXPAND\s+(\d+\.\d+[a-z]?)CLOSE\s+\1")
_CATALOG_RE      = re.compile(
    r"\b(?:BWV|WoO|Hob\.|D\.|Z\.|K\.|Op\.|op\.|RV)\s*[\d,/\.]+",
    re.IGNORECASE,
)
_NOTATION_RE     = re.compile(
    r"[\u0300-\u036f\u2190-\u21ff\u2200-\u22ff"
    r"\u2600-\u27ff\U0001D100-\U0001D1FF]"
)
_ROMAN_RE        = re.compile(
    r"\b(?:I{1,3}|IV|VI{0,3}|VII|ii{0,3}|iv|vi{0,2}|vii)"
    r"[°ø\u00b0]?(?:[6-9]|add\d)?\b"
)
_FIG_BASS_RE     = re.compile(
    r"\b(?:figured bass|thoroughbass|basso continuo)\b"
    r"|\b(?:[6-9]/[3-8]|6/4|6/3|7/5/3)\b",
    re.IGNORECASE,
)

# Map H1 text → passage_type for special section blocks
_H1_PTYPE: dict[str, str] = {
    "POINTS FOR REVIEW":  "points_for_review",
    "TEST YOURSELF":      "test_yourself",
    "A Closer Look":      "closer_look",
}

# Back matter block first-line markers
_BACK_MATTER_H1 = frozenset({
    "Test Yourself Answers",
    "Glossary",
    "Credits",
    "Index of Music Examples",
    "Index of Terms and Concepts",
})


def strip_markdown(text: str) -> str:
    """Remove inline markdown formatting for clean NLP text."""
    text = _BOLD_ITALIC.sub(r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^\*+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^-\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def clean_body_text(block: str) -> str:
    """
    Clean a spine block's body text for NLP:
    1. Remove '> More information' alt-text blocks (score descriptions).
    2. Remove bare example-number labels (lines like '1.5a').
    3. Strip markdown formatting.
    4. Collapse whitespace.
    """
    # Strip More information blocks
    text = _MORE_INFO_BLOCK.sub("", block)
    # Remove bare example labels
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped == "---":
            continue
        if _BARE_EX_LABEL.match(stripped):
            continue
        # Skip lines that are only an example caption label like "0.1"
        # followed by a brief description (these precede > More information)
        lines.append(stripped)
    text = " ".join(lines)
    text = strip_markdown(text)
    return re.sub(r"\s{2,}", " ", text).strip()


def split_at_glossary(block: str) -> tuple[str, str]:
    """
    Split a spine block at its '## Glossary' section.
    Returns (body_text, glossary_text).
    """
    idx = block.find("\n## Glossary\n")
    if idx < 0:
        return block, ""
    return block[:idx], block[idx+1:]


def _notation_ratio(text: str) -> float:
    if not text: return 0.0
    return len(_NOTATION_RE.findall(text)) / max(len(text), 1)


print("✓  Cell 3 loaded")


# %% CELL 4 — Concept extraction ──────────────────────────────────────────────
CONCEPT_PATTERNS: dict[str, list[str]] = {
    "secondary dominant":           [r"secondary dominant", r"applied dominant",
                                     r"applied chord", r"V/V", r"applied V"],
    "secondary leading-tone chord": [r"secondary leading[- ]tone",
                                     r"applied leading[- ]tone"],
    "Neapolitan chord":             [r"Neapolitan", r"♭II", r"bII"],
    "augmented sixth chord":        [r"augmented sixth", r"Italian sixth",
                                     r"French sixth", r"German sixth"],
    "modal mixture":                [r"modal mixture", r"mode mixture",
                                     r"borrowed chord"],
    "modulation":                   [r"\bmodulat", r"pivot chord",
                                     r"common[- ]chord modulation",
                                     r"closely related key", r"key area"],
    "tonicization":                 [r"toniciz"],
    "sequence":                     [r"\bsequence\b", r"sequential",
                                     r"descending.fifth", r"ascending.fifth",
                                     r"5–6 sequence", r"chromatic sequence"],
    "suspension":                   [r"\bsuspension\b", r"retardation",
                                     r"\b7-6\b", r"\b4-3\b", r"\b9-8\b"],
    "nonharmonic tone":             [r"nonchord tone", r"non[- ]chord tone",
                                     r"nonharmonic", r"passing tone",
                                     r"neighbor(?:ing)? tone", r"anticipation",
                                     r"appoggiatura", r"escape tone",
                                     r"pedal (?:point|tone)"],
    "cadence":                      [r"\bcadence\b", r"\bcadential\b",
                                     r"authentic cadence", r"half cadence",
                                     r"deceptive cadence", r"plagal cadence",
                                     r"Phrygian cadence"],
    "phrase":                       [r"\bphrase\b", r"\bperiod\b",
                                     r"antecedent", r"consequent",
                                     r"\bsentence\b", r"hypermeter"],
    "motive":                       [r"\bmotive\b", r"\bmotif\b", r"motivic"],
    "voice leading":                [r"voice.lead", r"part.writ",
                                     r"parallel fifth", r"parallel octave",
                                     r"\bdoubling\b", r"voice crossing"],
    "harmonic progression":         [r"harmonic progression",
                                     r"harmonic rhythm", r"circle of fifths"],
    "counterpoint":                 [r"counterpoint", r"contrapuntal",
                                     r"species counterpoint", r"cantus firmus"],
    "figured bass":                 [r"figured bass", r"thoroughbass"],
    "chord inversion":              [r"chord inversion", r"first inversion",
                                     r"second inversion", r"root position",
                                     r"6/4"],
    "six-four chord":               [r"six.four", r"cadential 6", r"6/4"],
    "triad":                        [r"\btriad\b"],
    "seventh chord":                [r"seventh chord"],
    "dominant seventh":             [r"dominant seventh", r"\bV7\b"],
    "leading tone":                 [r"leading.tone"],
    "leading-tone triad":           [r"leading.tone triad", r"vii"],
    "leading-tone seventh chord":   [r"leading.tone seventh", r"vii.*7",
                                     r"half[- ]diminished"],
    "submediant":                   [r"\bsubmediant\b", r"\bVI\b"],
    "mediant":                      [r"\bmediant\b", r"\bIII\b"],
    "subdominant":                  [r"\bsubdominant\b", r"\bIV\b"],
    "supertonic":                   [r"\bsupertonic\b", r"\bii\b"],
    "deceptive cadence":            [r"deceptive cadence", r"deceptive progression"],
    "interval":                     [r"\binterval\b", r"consonan",
                                     r"dissonan", r"tritone",
                                     r"half.step", r"whole.step"],
    "Roman numeral analysis":       [r"Roman numeral"],
    "major scale":                  [r"major scale", r"major key"],
    "minor scale":                  [r"minor scale", r"minor key",
                                     r"harmonic minor", r"melodic minor"],
    "key signature":                [r"key signature", r"circle of fifths"],
    "meter":                        [r"\bmeter\b", r"time signature",
                                     r"syncopation", r"hemiola"],
    "chorale":                      [r"\bchorale\b", r"\bSATB\b",
                                     r"four.part", r"vocal range"],
    "binary form":                  [r"binary form", r"rounded binary"],
    "ternary form":                 [r"ternary form"],
    "sonata form":                  [r"sonata form", r"exposition",
                                     r"development", r"recapitulation"],
    "rondo form":                   [r"rondo"],
    "chromaticism":                 [r"chromatic(?:ism)?"],
    "pentatonic scale":             [r"pentatonic"],
    "octatonic scale":              [r"octatonic"],
    "whole-tone scale":             [r"whole.tone"],
    "set theory":                   [r"set theory", r"pitch.class set",
                                     r"normal form", r"prime form"],
    "pitch class":                  [r"pitch.class", r"\bpc\s"],
    "twelve-tone":                  [r"twelve.tone", r"tone row",
                                     r"serial(?:ism)?"],
    "world music":                  [r"pentatonic", r"modal", r"non.Western",
                                     r"folk"],
    "mode":                         [r"\bmode\b", r"\bmodal\b",
                                     r"Dorian", r"Phrygian", r"Lydian",
                                     r"Mixolydian", r"Aeolian"],
    "tonic":                        [r"\btonic\b", r"\bI\b"],
    "dominant":                     [r"\bdominant\b", r"\bV\b"],
    "pitch":                        [r"\bpitch\b", r"staff", r"\bclef\b"],
    "rhythm":                       [r"\brhythm\b", r"\bbeat\b",
                                     r"note value"],
}

_COMPILED = [
    (c, re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE))
    for c, pats in CONCEPT_PATTERNS.items()
]


def extract_concepts(text: str) -> list[str]:
    return sorted({c for c, pat in _COMPILED if pat.search(text)})


def get_chapter_concepts(ch: str) -> list[str]:
    return list(CHAPTER_CONCEPT_MAP.get(ch, []))


print("✓  Cell 4 loaded")


# %% CELL 5 — Attribution extraction ──────────────────────────────────────────

_POSS_RE  = re.compile(r"'s\s*$")
_TRAIL_RE = re.compile(r"[,:\s]+$")


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


def _clean_work(raw: str) -> str:
    """Strip catalog numbers, mm. references, arranger credits, balance quotes."""
    work = _CATALOG_RE.sub("", raw)
    work = re.sub(r"\bop\.\s*\d+(?:,\s*no\.\s*\d+)*", "", work, flags=re.I)
    work = re.sub(r"\bmm?\.\s*[\d\u2013\u2014\-]+", "", work)
    # Strip arranger credits: "(arr. Name)", "(trans. Name)", "(ed. Name)"
    work = re.sub(r"\s*\([^)]*\b(?:arr|trans|ed)\b[^)]*\)", "", work, flags=re.I)
    work = re.sub(r",\s*,", ",", work)
    work = re.sub(r',(["\u201d])\s*$', r'\1', work)
    return re.sub(r"[,\s;.]+$", "", re.sub(r"\s{2,}", " ", work).strip()).strip()


def parse_all_attributions(full_text: str) -> dict[str, list[dict]]:
    """
    Extract all piece attributions from the full markdown text using the
    EXPAND N.NCLOSE N.N anchor markers.

    Each attributed example appears as:
        body_prose[N.N Composer, Work]\nEXPAND N.NCLOSE N.N
    where the attribution runs directly onto the end of the preceding sentence.

    Returns {chapter_num: [attribution_dict, ...]} — deduped within chapter.
    """
    result: dict[str, list[dict]] = defaultdict(list)
    # Dedup within chapter only — the same work CAN appear across multiple chapters
    seen_per_chapter: dict[str, set[str]] = defaultdict(set)

    for m in _EXPAND_ATTR_RE.finditer(full_text):
        ex_label  = m.group(1)
        attr_text = m.group(2).strip()
        ch_num    = ex_label.split(".")[0]

        # Confirm this attribution is followed by EXPAND N.NCLOSE N.N
        expand_pos = full_text.find(f"EXPAND {ex_label}CLOSE {ex_label}", m.end())
        if expand_pos < 0 or expand_pos - m.end() > 2000:
            continue

        # Parse "Composer, Work"
        comma = attr_text.find(",")
        if 0 < comma <= 50:
            composer_raw = attr_text[:comma].strip()
            work_raw     = attr_text[comma+1:].strip()
        else:
            continue  # no comma → likely not a piece attribution

        # Skip descriptive captions
        if not composer_raw or not composer_raw[0].isupper():
            continue
        first_word = composer_raw.split()[0].lower().rstrip(".,")
        if first_word in {"note", "see", "the", "a", "an", "in", "this",
                          "figure", "example", "diagram", "text", "shown"}:
            continue

        # Fall back to raw name if not in vocabulary — this book prominently
        # features historically underrepresented composers who may not yet be
        # in unique_composers.csv; we capture them regardless.
        canon = _canonicalize_composer(composer_raw) or composer_raw.strip()
        if not canon or canon in _CTX_ONLY:
            continue

        cat_m   = _CATALOG_RE.search(work_raw)
        catalog = re.sub(r"[,\s]+$", "", cat_m.group(0)).strip() if cat_m else ""
        work    = _clean_work(work_raw)

        # Deduplicate within chapter (same work can appear across different chapters)
        key = f"{canon}|{work[:30]}"
        if key not in seen_per_chapter[ch_num]:
            seen_per_chapter[ch_num].add(key)
            result[ch_num].append({
                "example_no": ex_label,
                "composer":   canon,
                "work":       work,
                "catalog":    catalog,
            })

    total = sum(len(v) for v in result.values())
    log.info("EXPAND/CLOSE attribution scan: %d attributions across %d chapters.",
             total, len(result))
    return dict(result)


def block_attributions(block_raw: str, chapter_attrs: list[dict]) -> list[dict]:
    """
    Filter chapter attribution list to only those examples whose
    EXPAND N.NCLOSE N.N marker appears within this spine block.
    Scopes piece_attributions to the passage that contains the example.
    """
    if not chapter_attrs:
        return []
    present: set[str] = set()
    for m in _EXPAND_ANY_RE.finditer(block_raw):
        present.add(m.group(1))
    return [a for a in chapter_attrs if a["example_no"] in present]


def parse_index_composers(index_text: str) -> list[str]:
    """
    Extract all composer names from the Index of Music Examples.
    The index uses "Last, First" format — flip to "First Last" before
    canonicalization.  Returns a flat list of canonical names.
    """
    composers: list[str] = []
    seen: set[str] = set()
    name_re = re.compile(r"^-\s+(.+)$", re.MULTILINE)

    for m in name_re.finditer(index_text):
        raw = m.group(1).strip()
        if not raw:
            continue
        # Skip anonymous/hymn entries (start with quote or bracket)
        if raw[0] in ('"', '\u201c', '[', '('):
            continue
        # Skip entries that contain page numbers (work lines, not name lines)
        if re.search(r"\b\d{2,}\b", raw):
            continue
        # Flip "Last, First" → "First Last"
        if "," in raw:
            parts = raw.split(",", 1)
            last, first = parts[0].strip(), parts[1].strip()
            if not last or not last[0].isupper():
                continue
            raw = f"{first} {last}".strip()
        # Use the name directly; canonicalize if possible, else keep raw
        if not raw or not raw[0].isupper():
            continue
        if len(raw.split()) < 1:
            continue
        canon = _canonicalize_composer(raw) or raw
        if canon and canon not in _CTX_ONLY and canon not in seen:
            seen.add(canon)
            composers.append(canon)
    return composers


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


# %% CELL 6 — Passage dataclass ───────────────────────────────────────────────
_PTYPE_ROLE: dict[str, str] = {
    "points_for_review": "supplementary",
    "test_yourself":     "application",
    "closer_look":       "supplementary",
    "glossary":          "supplementary",
    "chapter_opener":    "supplementary",
    "front_matter":      "supplementary",
    "back_matter":       "supplementary",
}


def _passage_role(ptype: str, ch: str,
                   concepts: Optional[list[str]] = None) -> str:
    if ptype in _PTYPE_ROLE: return _PTYPE_ROLE[ptype]
    cc = CHAPTER_CONCEPT_MAP.get(ch, [])
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


# %% CELL 7 — Core extraction loop ────────────────────────────────────────────

def _make_passage(
    text: str, ptype: str, ch: str, ch_title: str,
    sec_num: str, sec_title: str,
    spine_num: Optional[int], piece_attrs: list[dict], nlp
) -> Optional[Passage]:
    text = re.sub(r"\s+", " ", text).strip()
    words_list = text.split()
    if len(words_list) < MIN_WORDS_PER_PASSAGE:
        return None
    if not ch and ptype not in ("front_matter", "back_matter"):
        ptype = "front_matter"

    ex_nums  = list(dict.fromkeys(_EX_REF.findall(text)))
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
        chapter_number = ch,
        chapter_title  = ch_title,
        section_number = sec_num,
        section_title  = sec_title,
        page_start    = spine_num,
        page_stop     = spine_num,
        body_text     = text,
        passage_type  = ptype,
        notation_ratio = round(_notation_ratio(text), 4),
        concepts_passage      = cp,
        concepts_chapter      = get_chapter_concepts(ch),
        composers_vocab       = comp_v,
        composers_ner         = comp_n,
        composers_all_sources = comp_all,
        has_example_reference = bool(ex_nums),
        example_numbers       = ex_nums,
        piece_attributions    = piece_attrs,
        passage_role          = _passage_role(ptype, ch, cp),
        word_count     = len(words_list),
        sentence_count = len(re.split(r"(?<=[.!?])\s+", text)),
        char_count     = len(text),
        contains_roman_numerals = bool(_ROMAN_RE.search(text)),
        contains_figured_bass   = bool(_FIG_BASS_RE.search(text)),
    )


def _classify_block(block: str) -> tuple[str, str]:
    """
    Classify a spine block by its leading H1 heading.
    Returns (passage_type, section_title).
    """
    stripped = block.strip()
    m = _H1.match(stripped.split("\n")[0])
    if not m:
        return "body", ""

    h1 = m.group(1).strip()

    # Chapter opener
    if re.match(r"^chapter\s+\d+$", h1, re.IGNORECASE):
        return "chapter_opener", h1

    # Named special types
    if h1 in _H1_PTYPE:
        return _H1_PTYPE[h1], h1

    # Back matter
    if h1 in _BACK_MATTER_H1:
        return "back_matter", h1

    # Part dividers ("part one", "part two")
    if re.match(r"^part\s+(one|two|three|four|five|six|seven|eight|\d+)$", h1, re.I):
        return "front_matter", h1

    # ALL-CAPS body section headings
    if h1 == h1.upper() and len(h1) > 2:
        return "body", h1

    # Mixed-case H1 that isn't a chapter → body section
    return "body", h1


def extract_from_markdown(md_path: Path, nlp) -> list[Passage]:
    """
    Main extraction: splits on <!-- spine: N --> comments, classifies each
    block, strips noise (> More information, bare example labels), and
    builds passages.  Credits are parsed first and stored per-chapter for
    use as piece_attributions.
    """
    text = md_path.read_text(encoding="utf-8", errors="replace")
    log.info("MD file: %d chars, %d lines", len(text), text.count("\n"))

    raw_blocks = _SPINE_SEP.split(text)
    spine_nums  = [int(m) for m in re.findall(r"<!-- spine: (\d+) -->", text)]
    blocks_with_spine: list[tuple[str, Optional[int]]] = [(raw_blocks[0], None)]
    for i, blk in enumerate(raw_blocks[1:]):
        spine = spine_nums[i] if i < len(spine_nums) else None
        blocks_with_spine.append((blk, spine))

    # ── Pre-pass: parse all attributions and Index composers ────────────────
    # parse_all_attributions does a single full-document pass finding all
    # EXPAND-marked piece attributions (187 expected). Returns per-chapter dicts.
    all_chapter_attrs: dict[str, dict[str, dict]] = parse_all_attributions(text)

    index_block = next((b for b, _ in blocks_with_spine
                        if b.strip().startswith("# Index of Music Examples")), None)
    index_composers: list[str] = []
    if index_block:
        index_composers = parse_index_composers(index_block)
        log.info("Index of Music Examples: %d composer names.", len(index_composers))

    # ── Main extraction ───────────────────────────────────────────────────────
    all_passages: list[Passage] = []
    cur_ch        = ""
    cur_ch_title  = ""
    cur_sec_ctr   = 0

    for blk_idx, (block, spine_num) in enumerate(
        tqdm(blocks_with_spine, desc="Parsing blocks", unit="spine")
    ):
        # ── Back matter ───────────────────────────────────────────────────────
        if blk_idx >= BACK_MATTER_BLOCK:
            continue

        stripped = block.strip()
        if not stripped:
            continue

        # ── Chapter heading detection ─────────────────────────────────────────
        ch_m = _CHAPTER_H1.search(stripped)
        if ch_m:
            cur_ch       = ch_m.group(1)
            cur_ch_title = CHAPTER_TITLES.get(cur_ch, "")
            cur_sec_ctr  = 0

        # ── Classify block ────────────────────────────────────────────────────
        ptype, sec_title = _classify_block(block)

        # Chapter opener: extract title from second H1 in the block
        if ptype == "chapter_opener":
            lines = stripped.split("\n")
            for line in lines[1:]:
                l = line.strip()
                if l and l.startswith("# ") and not re.match(r"^# chapter", l, re.I):
                    cur_ch_title = l[2:].strip()
                    if cur_ch:
                        CHAPTER_TITLES[cur_ch] = cur_ch_title
                    break

        # ── Per-block attribution resolution ──────────────────────────────────
        ch_attr_list  = all_chapter_attrs.get(cur_ch, [])
        block_attrs   = block_attributions(block, ch_attr_list)

        # ── Handle ## Glossary split within body/section blocks ───────────────
        if "## Glossary" in block and ptype == "body":
            body_part, glossary_part = split_at_glossary(block)
            body_text = clean_body_text(body_part)
            sec_num   = f"{cur_ch}.{cur_sec_ctr}" if cur_ch else ""
            p = _make_passage(body_text, "body", cur_ch, cur_ch_title,
                               sec_num, sec_title, spine_num, block_attrs, nlp)
            if p: all_passages.append(p)
            gloss_text = clean_body_text(glossary_part)
            g = _make_passage(gloss_text, "glossary", cur_ch, cur_ch_title,
                               sec_num, "Glossary", spine_num, [], nlp)
            if g: all_passages.append(g)
            cur_sec_ctr += 1
            continue

        # ── Normal passage ────────────────────────────────────────────────────
        if ptype == "front_matter" and not cur_ch:
            continue

        body_text = clean_body_text(block)
        # Body/closer_look passages get block-scoped attributions.
        # POINTS FOR REVIEW and TEST YOURSELF don't cite musical examples.
        attrs = block_attrs if ptype in ("body", "closer_look", "chapter_opener") else []
        sec_num = f"{cur_ch}.{cur_sec_ctr}" if cur_ch else ""

        p = _make_passage(body_text, ptype, cur_ch, cur_ch_title,
                           sec_num, sec_title, spine_num, attrs, nlp)
        if p: all_passages.append(p)
        if ptype == "body":
            cur_sec_ctr += 1

    # ── Enrich composer regex vocabulary with Index composers ───────────────
    # The Index gives us names not in the composer CSV (newer/diverse composers).
    # Add them to the fold_map and regex so vocab matching can find them in body text.
    if index_composers and _FOLD_MAP is not None:
        # Canonicalize index names through initials_aliases first, so
        # "W. A. Mozart" → "Wolfgang Amadeus Mozart" rather than duplicating.
        resolved = []
        for n in index_composers:
            canon = _INITIALS_ALIASES.get(_fold(n)) or n
            resolved.append(canon)
        new_names = [n for n in resolved if _fold(n) not in _FOLD_MAP]
        if new_names:
            extra_rex = set(new_names)
            escaped = sorted((re.escape(n) for n in extra_rex), key=len, reverse=True)
            global _REGEX
            existing_pats = _REGEX.pattern if _REGEX else None
            new_pat = r"\b(" + "|".join(escaped) + r")\b"
            if existing_pats:
                _REGEX = re.compile(f"(?:{existing_pats})|(?:{new_pat})", re.IGNORECASE)
            else:
                _REGEX = re.compile(new_pat, re.IGNORECASE)
            for name in new_names:
                _FOLD_MAP[_fold(name)] = name
            log.info("Index enrichment: added %d new composer forms to vocabulary.", len(new_names))

    log.info("Extraction complete: %d passages.", len(all_passages))
    return all_passages


print("✓  Cell 7 loaded")


# %% CELL 8 — Run & write JSONL ───────────────────────────────────────────────
def main() -> list[Passage]:
    global _ALIASES, _FOLD_MAP, _REGEX, _SURNAMES, _CTX_ONLY

    md_path = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_MD_PATH
    if not md_path.exists():
        cand = Path("/mnt/user-data/uploads") / md_path.name
        if cand.exists(): md_path = cand
        else: log.error("MD file not found: %s", md_path); return []

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

    return extract_from_markdown(md_path, nlp)


passages = main()
print(f"\n✓  Extraction complete — {len(passages)} passages")

OUTPUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_JSONL_PATH.open("w", encoding="utf-8") as fh:
    for p in passages:
        fh.write(p.to_jsonl_line() + "\n")
log.info("Wrote %d records → %s", len(passages), OUTPUT_JSONL_PATH)


# %% CELL 9 — Diagnostic report ───────────────────────────────────────────────
from collections import Counter

def print_summary(passages: list[Passage]) -> None:
    if not passages:
        print("  ⚠  No passages extracted."); return
    sep = "═" * 64
    print(f"\n{sep}")
    print(f"  Concise Introduction to Tonal Harmony 3rd ed. — summary")
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

    chs = Counter(p.chapter_number for p in passages if p.chapter_number)
    covered  = sorted(chs.keys(), key=lambda x: int(x) if x.isdigit() else 99)
    expected = [str(i) for i in range(48)]
    missing  = [c for c in expected if c not in chs]
    print(f"\n  Chapters: {len(covered)}/48 covered  missing={missing}")
    for k in covered:
        t   = CHAPTER_TITLES.get(k, "")[:35]
        n   = chs[k]
        rn  = sum(1 for p in passages if p.chapter_number==k and p.contains_roman_numerals)
        print(f"    Ch{k:>3}  {n:>3}p  {t:<35}  rn={rn}")

    comp_counts = Counter(c for p in passages for c in p.composers_all_sources)
    print(f"\n  Top 20 composers (unique: {len(comp_counts)}):")
    for name, n in comp_counts.most_common(20):
        print(f"    {name:<42} {n:>4}x")

    attrs  = [a for p in passages for a in p.piece_attributions]
    body   = [p for p in passages if p.passage_type == "body"]
    w_conc = sum(1 for p in body if p.concepts_passage)
    wcs    = [p.word_count for p in passages]

    print(f"\n  Total piece attributions        : {len(attrs)}")
    print(f"  Body passages with ≥1 concept   : {w_conc}/{len(body)} ({100*w_conc/max(len(body),1):.0f}%)")
    print(f"  Word count: min={min(wcs)}  max={max(wcs)}  mean={sum(wcs)/len(wcs):.0f}")
    print(f"\n  JSONL → {OUTPUT_JSONL_PATH}")
    print(f"{sep}\n")


print_summary(passages)
