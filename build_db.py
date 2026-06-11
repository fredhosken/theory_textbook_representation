"""
build_db.py
══════════════════════════════════════════════════════════════════════════════
Assemble a single SQLite database for the music-theory textbook representation
project, so that composer–content relations and cross-textbook patterns can be
surfaced with SQL instead of re-running the GPU NLP pipeline.

This is an ADDITIVE tool. It reads files the pipeline already produces and
writes one new file (textrep.db). It does not modify the analysis server or the
R script, and nothing downstream depends on it. Re-run it any time the inputs
change; it rebuilds the DB from scratch (idempotent).

Inputs (all optional except --data-dir + --bio):
    data/*.jsonl                       passage-level corpus (one file per book)
    bio_data_processed.csv             per-composer demographics + geography
    concepts_music_theory.csv          concept → regex patterns
    composer_aliases.csv               variant → canonical composer names
    results/                           server CSV outputs (01_*.csv … 15_*.csv)
                                       imported verbatim as analysis_* tables

Output:
    textrep.db   (SQLite; open with `sqlite3 textrep.db` or DBI::dbConnect)

Schema (core relational tables)
    textbooks(book_id, title, edition, year, authors, open_access)
    composers(composer_id, canonical_name, fold_key, sex, bipoc, race,
              country, continent, latitude, longitude, born, died, dominant_flag)
    composer_aliases(variant, canonical, composer_id)
    passages(passage_id, book_id, chapter_number, chapter_title, section_number,
             section_title, page_start, page_end, body_text, passage_type,
             passage_role, word_count, notation_ratio, framing_label,
             framing_score)
    passage_composers(id, passage_id, composer_id, detection_source)
    concepts(concept_id, name, patterns, auto_generated)
    passage_concepts(passage_id, concept_id, source)
    analysis_<name>   one table per results/*.csv (verbatim import)

Usage
─────
    python build_db.py
    python build_db.py --data-dir ./data --bio bio_data_processed.csv \
        --concepts concepts_music_theory.csv --aliases composer_aliases.csv \
        --results ./results --db textrep.db
"""
from __future__ import annotations
import argparse, csv, json, re, sqlite3, sys, unicodedata
from pathlib import Path

# "The Boys" + the wider canonical core — used to set dominant_flag.
DOMINANT_SURNAMES = {
    "bach", "handel", "haydn", "mozart", "beethoven", "schubert", "chopin",
    "schumann", "brahms", "mendelssohn", "wagner", "liszt", "tchaikovsky",
    "vivaldi", "corelli", "scarlatti", "telemann", "purcell", "monteverdi",
    "palestrina", "dvorak", "verdi", "rossini", "berlioz", "debussy", "ravel",
    "stravinsky", "bartok", "prokofiev", "shostakovich", "schoenberg", "webern",
    "berg", "hindemith",
}

# Maps the server's internal book keys (used in DATA_FILES and all results/*.csv
# outputs) to the JSONL file stems (used as book_id in passages/textbooks).
# Without this, any JOIN between passages and analysis_* tables on book_id
# silently returns zero rows.
#
# Keys must mirror DATA_FILES in analysis_server.py exactly; update this dict
# whenever DATA_FILES changes. Unmapped keys are reported as warnings at build
# time rather than failing silently.
BOOK_ID_MAP: dict[str, str] = {
    "aldwell_2019":       "aldwell_schachter_5ed",
    "benward_2021":       "benward_saker_v1_10ed",
    "burnstein_2025":     "burnstein_straus_3ed",
    "clendinning_2026":   "clendinning_marvin_5ed",
    "gotham_2023":        "gotham_omt",
    "hutchinson_2025":    "hutchinson_mt21c_2025",
    "kostka_2024":        "kostka_almen_9ed",
    "laitz_2023":         "laitz_5ed",
    "mount_2020":         "mount_fff_2020",
    "roig_francoli_2020": "roig_francoli_3ed",
}

# Reverse map for populating textbooks.server_id
_JSONL_TO_SERVER: dict[str, str] = {v: k for k, v in BOOK_ID_MAP.items()}


def fold(s) -> str:
    if s is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower().strip()


def surname(name: str) -> str:
    n = re.sub(r"\(.*?\)", "", name).strip()
    return fold(n.split()[-1]) if n.split() else ""


# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS passage_concepts;
DROP TABLE IF EXISTS passage_composers;
DROP TABLE IF EXISTS book_id_map;
DROP TABLE IF EXISTS passages;
DROP TABLE IF EXISTS composer_aliases;
DROP TABLE IF EXISTS concepts;
DROP TABLE IF EXISTS composers;
DROP TABLE IF EXISTS textbooks;

CREATE TABLE textbooks (
    book_id      TEXT PRIMARY KEY,   -- JSONL file stem (canonical throughout DB)
    server_id    TEXT,               -- server DATA_FILES key (used in analysis CSVs)
    title        TEXT,
    edition      INTEGER,
    year         INTEGER,
    authors      TEXT,
    open_access  INTEGER DEFAULT 0
);

-- Convenience lookup between the two book-ID namespaces.
-- book_id = JSONL stem; server_id = server DATA_FILES key.
CREATE TABLE book_id_map (
    server_id TEXT PRIMARY KEY,
    jsonl_id  TEXT NOT NULL REFERENCES textbooks(book_id)
);

CREATE TABLE composers (
    composer_id    INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    fold_key       TEXT UNIQUE,
    sex            TEXT,
    bipoc          TEXT,
    race           TEXT,
    country        TEXT,
    continent      TEXT,
    latitude       REAL,
    longitude      REAL,
    born           INTEGER,
    died           INTEGER,
    dominant_flag  INTEGER
);

CREATE TABLE composer_aliases (
    variant     TEXT,
    canonical   TEXT,
    composer_id INTEGER REFERENCES composers(composer_id)
);

CREATE TABLE concepts (
    concept_id     INTEGER PRIMARY KEY,
    name           TEXT UNIQUE,
    patterns       TEXT,
    auto_generated INTEGER DEFAULT 0
);

CREATE TABLE passages (
    passage_id     INTEGER PRIMARY KEY,
    book_id        TEXT REFERENCES textbooks(book_id),
    chapter_number INTEGER,
    chapter_title  TEXT,
    section_number TEXT,
    section_title  TEXT,
    page_start     INTEGER,
    page_end       INTEGER,
    body_text      TEXT,
    passage_type   TEXT,
    passage_role   TEXT,
    word_count     INTEGER,
    notation_ratio REAL,
    framing_label  TEXT,
    framing_score  REAL
);

CREATE TABLE passage_composers (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    passage_id       INTEGER REFERENCES passages(passage_id),
    composer_id      INTEGER REFERENCES composers(composer_id),
    composer_name    TEXT,
    detection_source TEXT
);

CREATE TABLE passage_concepts (
    passage_id INTEGER REFERENCES passages(passage_id),
    concept_id INTEGER REFERENCES concepts(concept_id),
    source     TEXT
);

CREATE INDEX idx_pc_passage  ON passage_composers(passage_id);
CREATE INDEX idx_pc_composer ON passage_composers(composer_id);
CREATE INDEX idx_pcon_passage ON passage_concepts(passage_id);
CREATE INDEX idx_pcon_concept ON passage_concepts(concept_id);
CREATE INDEX idx_pass_book    ON passages(book_id);
"""


def _to_int(v):
    try:
        return int(float(str(v).strip()))
    except (ValueError, TypeError):
        return None


def _to_real(v):
    try:
        return float(str(v).strip())
    except (ValueError, TypeError):
        return None


def _to_str(v) -> str | None:
    """Coerce any value (including lists) to a plain string for SQLite."""
    if v is None:
        return None
    if isinstance(v, list):
        return "; ".join(str(x) for x in v if x is not None)
    s = str(v).strip()
    return s if s else None


def load_aliases(path: Path) -> dict[str, str]:
    out = {}
    if path and path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                var, canon = (row.get("variant") or "").strip(), (row.get("canonical") or "").strip()
                if var and canon:
                    out[fold(var)] = canon
    return out


def build_composers(con, bio_path: Path, aliases: dict[str, str]):
    cur = con.cursor()
    fold_to_id: dict[str, int] = {}
    next_id = 1
    if bio_path and bio_path.exists():
        with bio_path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                name = (row.get("composer") or "").strip()
                if not name:
                    continue
                fk = fold(aliases.get(fold(name), name))
                if fk in fold_to_id:
                    continue
                cur.execute("""
                    INSERT INTO composers
                      (composer_id, canonical_name, fold_key, sex, bipoc, race,
                       country, continent, latitude, longitude, born, died,
                       dominant_flag)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    next_id, aliases.get(fk, name), fk,
                    row.get("sex"), row.get("bipoc"), row.get("race"),
                    row.get("country"), row.get("continent"),
                    _to_real(row.get("latitude")), _to_real(row.get("longitude")),
                    _to_int(row.get("born")), _to_int(row.get("died")),
                    1 if surname(name) in DOMINANT_SURNAMES else 0,
                ))
                fold_to_id[fk] = next_id
                next_id += 1
    # alias rows
    for var_fold, canon in aliases.items():
        cid = fold_to_id.get(fold(canon))
        cur.execute("INSERT INTO composer_aliases (variant, canonical, composer_id) VALUES (?,?,?)",
                    (var_fold, canon, cid))
    con.commit()
    print(f"  composers: {len(fold_to_id)}")
    return fold_to_id


def build_concepts(con, concepts_path: Path):
    cur = con.cursor()
    name_to_id: dict[str, int] = {}
    if not (concepts_path and concepts_path.exists()):
        return name_to_id
    with concepts_path.open(encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            name = (row.get("concept") or "").strip()
            if not name:
                continue
            cur.execute("INSERT OR IGNORE INTO concepts (concept_id, name, patterns, auto_generated) VALUES (?,?,?,?)",
                        (i, name, row.get("patterns", ""), _to_int(row.get("auto_generated")) or 0))
            name_to_id[name.lower()] = i
    con.commit()
    print(f"  concepts: {len(name_to_id)}")
    return name_to_id


def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return default


def resolve_composer(con, fold_to_id, aliases, name: str) -> int | None:
    fk = fold(aliases.get(fold(name), name))
    if fk in fold_to_id:
        return fold_to_id[fk]
    return None


def build_passages(con, data_dir: Path, fold_to_id, name_to_concept, aliases):
    cur = con.cursor()
    pid = 1
    books_seen = {}
    n_links_comp = n_links_con = 0

    jsonl_files = sorted(data_dir.glob("*.jsonl")) if data_dir.exists() else []
    for jf in jsonl_files:
        book_key = jf.stem
        with jf.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # textbook row (once per file, from first record's metadata)
                if book_key not in books_seen:
                    book_id = _first(rec, "book_id", default=book_key)
                    cur.execute("""INSERT OR IGNORE INTO textbooks
                        (book_id, server_id, title, edition, year, authors, open_access)
                        VALUES (?,?,?,?,?,?,?)""", (
                        book_id,
                        _JSONL_TO_SERVER.get(book_key),   # None if not in map
                        _to_str(_first(rec, "book_title", "title")),
                        _to_int(_first(rec, "book_edition", "edition")),
                        _to_int(_first(rec, "book_year", "year")),
                        _to_str(_first(rec, "book_authors", "authors")),
                        0,
                    ))
                    books_seen[book_key] = book_id
                book_id = books_seen[book_key]

                cur.execute("""INSERT INTO passages
                    (passage_id, book_id, chapter_number, chapter_title,
                     section_number, section_title, page_start, page_end,
                     body_text, passage_type, passage_role, word_count,
                     notation_ratio, framing_label, framing_score)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    pid, book_id,
                    _to_int(_first(rec, "chapter_number")),
                    _first(rec, "chapter_title"),
                    str(_first(rec, "section_number", default="")) or None,
                    _first(rec, "section_title"),
                    _to_int(_first(rec, "page_start")),
                    _to_int(_first(rec, "page_stop", "page_end")),
                    _to_str(_first(rec, "body_text")),
                    _first(rec, "passage_type"),
                    _first(rec, "passage_role"),
                    _to_int(_first(rec, "word_count")),
                    _to_real(_first(rec, "notation_ratio")),
                    _first(rec, "framing_label", "framing"),
                    _to_real(_first(rec, "framing_score")),
                ))

                # composer links (union of NER + vocab + any 'composers_all')
                comp_sources = {
                    "ner":   _first(rec, "composers_ner", default=[]),
                    "vocab": _first(rec, "composers_vocab", "composers_named", default=[]),
                    "all":   _first(rec, "composers_all_sources", default=[]),
                }
                linked = set()
                for src, names in comp_sources.items():
                    if isinstance(names, str):
                        names = [names]
                    for nm in (names or []):
                        nm = str(nm).strip()
                        if not nm or (nm, src) in linked:
                            continue
                        cid = resolve_composer(con, fold_to_id, aliases, nm)
                        cur.execute("""INSERT INTO passage_composers
                            (passage_id, composer_id, composer_name, detection_source)
                            VALUES (?,?,?,?)""", (pid, cid, nm, src))
                        linked.add((nm, src))
                        n_links_comp += 1

                # concept links (passage-level + chapter-level)
                for field, src in (("concepts_passage", "passage"),
                                   ("concepts_chapter", "chapter")):
                    vals = _first(rec, field, default=[])
                    if isinstance(vals, str):
                        vals = [vals]
                    for c in (vals or []):
                        cid = name_to_concept.get(str(c).strip().lower())
                        if cid:
                            cur.execute("""INSERT INTO passage_concepts
                                (passage_id, concept_id, source) VALUES (?,?,?)""",
                                (pid, cid, src))
                            n_links_con += 1
                pid += 1
        con.commit()
    print(f"  textbooks: {len(books_seen)}")
    print(f"  passages: {pid-1}")
    print(f"  passage→composer links: {n_links_comp}")
    print(f"  passage→concept links: {n_links_con}")


def _translate_book_ids(row: list, book_id_cols: set[int]) -> list:
    """Replace server book keys with JSONL stems in any book_id column."""
    out = list(row)
    for i in book_id_cols:
        if i < len(out) and out[i] and out[i] in BOOK_ID_MAP:
            out[i] = BOOK_ID_MAP[out[i]]
    return out


def import_results(con, results_dir: Path):
    """Import results/*.csv as analysis_<name> tables, normalising book_id
    columns from server keys (e.g. clendinning_2026) to JSONL stems
    (clendinning_marvin_5ed) so joins to passages/textbooks work correctly."""
    if not (results_dir and results_dir.exists()):
        print("  results/: not found — skipping analysis_* tables")
        return
    cur = con.cursor()
    n = 0
    unmapped: set[str] = set()
    for csvf in sorted(results_dir.glob("*.csv")):
        tbl = "analysis_" + re.sub(r"[^0-9a-zA-Z]+", "_", csvf.stem).strip("_").lower()
        with csvf.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                continue
            cols = [re.sub(r"[^0-9a-zA-Z]+", "_", h).strip("_").lower() or f"col{i}"
                    for i, h in enumerate(header)]
            # Identify columns that carry book identifiers
            book_id_cols = {i for i, c in enumerate(cols) if c in ("book_id", "book")}
            cur.execute(f"DROP TABLE IF EXISTS {tbl}")
            cur.execute(f'CREATE TABLE {tbl} ({", ".join(c + " TEXT" for c in cols)})')
            ph = ",".join("?" * len(cols))
            for raw_row in reader:
                raw_row = (raw_row + [None] * len(cols))[:len(cols)]
                # Detect values that look like server keys but aren't in the map
                for i in book_id_cols:
                    v = raw_row[i]
                    if v and v not in BOOK_ID_MAP and v not in _JSONL_TO_SERVER:
                        unmapped.add(v)
                row = _translate_book_ids(raw_row, book_id_cols)
                cur.execute(f"INSERT INTO {tbl} VALUES ({ph})", row)
        n += 1
    con.commit()
    if unmapped:
        print(f"  WARNING: {len(unmapped)} unrecognised book_id value(s) in analysis CSVs")
        print(f"    Add to BOOK_ID_MAP: {sorted(unmapped)}")
    print(f"  analysis_* tables: {n} (book_id columns normalised to JSONL stems)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--bio",      type=Path, default=Path("bio_data_processed.csv"))
    ap.add_argument("--concepts", type=Path, default=Path("concepts_music_theory.csv"))
    ap.add_argument("--aliases",  type=Path, default=Path("composer_aliases.csv"))
    ap.add_argument("--results",  type=Path, default=Path("results"))
    ap.add_argument("--db",       type=Path, default=Path("textrep.db"))
    a = ap.parse_args()

    if a.db.exists():
        a.db.unlink()
    con = sqlite3.connect(a.db)
    con.executescript(SCHEMA)

    print(f"Building {a.db} ...")
    aliases = load_aliases(a.aliases)
    print(f"  aliases: {len(aliases)}")
    fold_to_id = build_composers(con, a.bio, aliases)
    name_to_concept = build_concepts(con, a.concepts)
    build_passages(con, a.data_dir, fold_to_id, name_to_concept, aliases)

    # Populate book_id_map from BOOK_ID_MAP for any textbook now in the DB
    cur = con.cursor()
    for server_id, jsonl_id in BOOK_ID_MAP.items():
        exists = cur.execute(
            "SELECT 1 FROM textbooks WHERE book_id = ?", (jsonl_id,)
        ).fetchone()
        if exists:
            cur.execute(
                "INSERT OR IGNORE INTO book_id_map (server_id, jsonl_id) VALUES (?,?)",
                (server_id, jsonl_id),
            )
    con.commit()

    import_results(con, a.results)

    # Quick sanity counts
    cur = con.cursor()
    for tbl in ("textbooks", "book_id_map", "composers", "concepts", "passages",
                "passage_composers", "passage_concepts"):
        n = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"    {tbl:20s} {n:>8d} rows")
    unresolved = cur.execute(
        "SELECT COUNT(*) FROM passage_composers WHERE composer_id IS NULL").fetchone()[0]
    total_links = cur.execute("SELECT COUNT(*) FROM passage_composers").fetchone()[0]
    if total_links:
        print(f"    unresolved composer links: {unresolved}/{total_links} "
              f"({100*unresolved/total_links:.1f}%) — add these names to "
              f"composer_aliases.csv to improve coverage")
    con.close()
    print(f"Done → {a.db}")


if __name__ == "__main__":
    main()
