# Demographic Diversity and Framing in Music Theory Textbooks — Analysis Pipeline

This repository accompanies the article *Counted But Not Centered: Demographic Diversity and Framing in Post-2020 Music Theory Textbooks* and
contains the complete computational pipeline behind it: a demographic audit of
composer representation across ten college music-theory textbooks (2019–2026)
and an NLP framing analysis of how those composers are pedagogically deployed.

## What is and is not included

The textbooks analyzed in the article are copyrighted works. **No textbook
text is distributed in this repository in any form.** Specifically excluded
are the passage-level JSONL corpus, the assembled SQLite database
(`textrep.db`), and the per-mention results tables that carry verbatim context
windows. The `.gitignore` enforces these exclusions.

What *is* included: all analysis code, the independently curated reference
datasets (composer demographics and the composer–piece index, which contain
facts about the books but no expression from them), and a fully synthetic
example corpus that exercises the pipeline end to end. With these materials
and independently obtained copies of the textbooks, the study is reproducible.

## Repository layout

```
analysis_server.py                  NLP analysis (TF-IDF, BGE embeddings, LLM framing classifier)
build_db.py                         Assembles textrep.db from corpus + results
framing_analysis.py                 Statistical analysis (chi-square, Fisher, BH-FDR)
prepare_bio_csv.py                  Builds the composer demographic table (output already in data/)
Textbook_Representation_Analysis.R  All figures (demographic audit + statistical-support plots)
requirements.txt                    Python dependencies
data/
  bio_data_processed.csv            Per-composer demographics and geography (~925 composers)
  composers_pieces_index.xlsx       Composer–piece index across 20 textbook editions
  composer_aliases.csv              Variant -> canonical composer-name mappings (106 rows)
  unique_composers.csv              Composer vocabulary for the extraction scripts (945 names)
  concepts_music_theory.csv         Concept -> regex-pattern vocabulary (248 concepts)
examples/dummy_corpus/              Synthetic two-book corpus + synthetic results (see below)
examples/extraction/                Two worked extraction examples (PDF and EPUB-Markdown)
LICENSE                             MIT (code)
LICENSE-DATA.md                     CC BY 4.0 (data)
```

## Pipeline overview

The pipeline runs in five stages. Each stage's outputs feed the next.

1. **Extraction.** Each textbook is converted to a passage-level JSONL
   file, one record per passage, carrying structural metadata, body text,
   detected composer mentions, and concept tags. Extraction is necessarily
   bespoke per book; `examples/extraction/` contains two worked examples
   illustrating the two main strategies (see below).
2. **`analysis_server.py`** reads the JSONL corpus and produces the
   `results/` CSV tables: composer-mention framing classifications
   (lexicon backbone, upgraded by BAAI/bge-large-en-v1.5 embeddings and a
   Llama-3.1-8B-Instruct zero-shot pass), representation metrics, and the
   `r_data/` exports for the R script.
3. **`build_db.py`** assembles everything — corpus, demographics, concept
   vocabulary, aliases, and the results CSVs — into one SQLite database,
   normalizing the two book-ID namespaces via `BOOK_ID_MAP`.
4. **`framing_analysis.py`** queries the database and produces the
   statistical results reported in the article (framing distributions,
   between-book variation, structural placement by gender and BIPOC status,
   integration rates, and concept associations with FDR correction).
5. **`Textbook_Representation_Analysis.R`** produces every figure: the
   demographic audit (Part 1, from `data/composers_pieces_index.xlsx`) and
   the NLP and statistical-support plots (Part 2, from `r_data/` and
   `textrep.db`).

## Setup

Python 3.10 or later:

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Model access: the embedding model (`BAAI/bge-large-en-v1.5`) downloads
automatically from the Hugging Face Hub on first run.
`meta-llama/Llama-3.1-8B-Instruct` is a gated model — request access on its
Hugging Face page and authenticate with `huggingface-cli login` before running
the server. A CUDA GPU with roughly 18 GB or more of available VRAM is
recommended for the fp16 classifier; on smaller cards set
`LLM_CLASSIFIER_LOAD_4BIT = True` in `analysis_server.py` and install
`bitsandbytes`.

R (4.x) packages, for the figure script: tidyverse, readxl, janitor, DBI,
RSQLite, igraph, ggraph, vegan, scales, maps, geosphere, ggrepel, and
optionally writexl, sf, rnaturalearth, rnaturalearthdata.

## Minimal working example

`examples/dummy_corpus/` contains a synthetic two-book corpus (28 invented
passages mentioning real composers; no published text), a miniature alias and
concept vocabulary, and synthetic results tables. It exists so the database
assembly and statistical stages can be run and inspected without any
copyrighted material:

```bash
python build_db.py \
    --data-dir examples/dummy_corpus/data \
    --bio      data/bio_data_processed.csv \
    --concepts examples/dummy_corpus/concepts_music_theory.csv \
    --aliases  examples/dummy_corpus/composer_aliases.csv \
    --results  examples/dummy_corpus/results \
    --db       example.db

python framing_analysis.py example.db
```

The build prints a warning that the two synthetic book IDs are not in
`BOOK_ID_MAP`; that is expected (the map covers the ten real textbooks) and
demonstrates how unmapped IDs are surfaced rather than silently mistranslated.
The numbers `framing_analysis.py` prints for the dummy corpus are, of course,
meaningless — the example verifies mechanics, not findings.

## Reproducing the full study

1. Obtain the ten textbooks independently (the corpus is listed in the
   article and in `BOOK_LABELS` in `framing_analysis.py`).
2. Convert each book to a JSONL file matching the schema below, named with
   the stems in `DATA_FILES` (`analysis_server.py`) and placed in `data/`.
   Extraction is necessarily bespoke per book — page layouts, running
   headers, and section conventions differ — so no single general extractor
   exists. Instead, `examples/extraction/` provides one worked example per
   source format: `extract_aldwell5.py` parses a PDF with pdfplumber, using
   the book's font taxonomy (face and size) to separate body prose, section
   headings, example captions, and special sections; `extract_burstein.py`
   parses a Markdown export of an EPUB, using its heading and spine-block
   structure for the same purpose. Both scripts document their layout
   constants in the module docstring, write the JSONL schema below, and end
   with a diagnostic report (chapter coverage, composer-mention counts,
   attribution rates) for verifying a new extraction. They expect
   `unique_composers.csv` and `composer_aliases.csv` in the working
   directory for the composer-vocabulary pass (copy them from `data/`, or
   point the `*_PATH` constants in CELL 1 at `data/`). Adapting one of them to a
   new book means rewriting the layout constants and parsing rules in CELLs
   3–5 while keeping CELLs 6–9 (passage dataclass, JSONL writer,
   diagnostics) intact.
3. Run the server, then assemble and analyze:

```bash
python analysis_server.py --data-dir ./data --output-dir ./results \
    --bio-data data/bio_data_processed.csv \
    --concept-csv data/concepts_music_theory.csv
python build_db.py --data-dir ./data \
    --bio data/bio_data_processed.csv \
    --concepts data/concepts_music_theory.csv \
    --aliases data/composer_aliases.csv
python framing_analysis.py
Rscript Textbook_Representation_Analysis.R
```

The R script's geographic sections (Section 11) additionally read a
composer-to-country lookup (`composers_countries.csv`), which is not included
in this release; all other sections run without it from
`data/composers_pieces_index.xlsx`, the `r_data/` exports, and `textrep.db`.

### JSONL schema

One JSON object per line, per passage. Fields consumed by the pipeline
(see `examples/dummy_corpus/data/` for working records):

`book_id`, `book_title`, `book_edition`, `book_year`, `book_authors` —
book metadata (read from the first record of each file);
`chapter_number`, `chapter_title`, `section_number`, `section_title`,
`page_start`, `page_stop` — structural position;
`body_text` — passage text; `passage_type` (prose / example / exercise),
`passage_role` (central / supplementary / application), `word_count`,
`notation_ratio` — passage characterization;
`composers_ner`, `composers_vocab` — detected composer mentions by source;
`concepts_passage`, `concepts_chapter` — concept tags at two granularities.

## Licenses

Code is released under the MIT License (`LICENSE`). The curated datasets in
`data/` and the synthetic examples are released under CC BY 4.0
(`LICENSE-DATA.md`).

## Citation

Please cite the accompanying article; see `CITATION.cff`.
