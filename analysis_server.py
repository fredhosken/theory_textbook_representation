#!/usr/bin/env python3
"""
analysis_server.py
──────────────────
NLP representation and framing analysis of the music-theory textbook corpus.
Reads the passage-level JSONL corpus and writes the results/ CSV tables,
figures, and r_data/ exports consumed by build_db.py, framing_analysis.py,
and the R visualization script.

A CUDA GPU is strongly recommended: the LLM framing classifier loads
Llama-3.1-8B-Instruct in fp16 (~16 GB VRAM); 4-bit quantisation is available
for smaller cards (see LLM_CLASSIFIER_LOAD_4BIT).

Usage
─────
  python analysis_server.py [OPTIONS]

  --data-dir DIR           Directory containing JSONL textbook files
  --output-dir DIR         Where to write results (default: ./results)
  --bio-data PATH          Path to biographical CSV (optional)
  --concept-csv PATH       Path to concepts_music_theory.csv (optional)
  --excel-data PATH        Path to theory_diversity_full_export.xlsx (optional)
  --bert-upgrade-threshold FLOAT  BERT cosine similarity cutoff (default: 0.50)
  --zero-shot-threshold FLOAT     LLM classifier confidence cutoff (default: 0.70)
  --no-bert                Disable BERT encoder pass
  --no-zero-shot           Disable LLM framing classifier pass
  --tfidf-keywords         Enable TF-IDF keyword extraction (off by default)
  --debug                  Fast run — sample N passages per book
  --sample-n N             Passages per file in debug mode (default: 80)
"""

import argparse
import re
import json
import difflib
import warnings
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import Counter, OrderedDict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DEFAULT CONFIGURATION  (override via CLI arguments)                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Book IDs mapped to filenames relative to --data-dir
DATA_FILES = {
    "aldwell_2019":    "aldwell_schachter_5ed.jsonl",
    "benward_2021":    "benward_saker_v1_10ed.jsonl",
    "burnstein_2025":   "burnstein_straus_3ed.jsonl",
    "clendinning_2026":  "clendinning_marvin_5ed.jsonl",
    "gotham_2023":     "gotham_omt.jsonl",
    "hutchinson_2025": "hutchinson_mt21c_2025.jsonl",
    "kostka_2024":     "kostka_almen_9ed.jsonl",
    "laitz_2023":      "laitz_5ed.jsonl",
    "mount_2020":      "mount_fff_2020.jsonl",
    "roig_francoli_2020": "roig_francoli_3ed.jsonl",
}

# Maps Excel 'short_label' values to book_ids used in DATA_FILES above.
EXCEL_BOOK_MAP = {
    "Aldwell, 5th":  "aldwell_2019",
    "Benward, 10th": "benward_2021",
    "Burstein, 5th": "burnstein_2025",
    "Clendinning, 5th": "clendinning_2026",
    "Gotham":        "gotham_2023",
    "Hutchinson":    "hutchinson_2025",
    "Kostka, 9th":   "kostka_2024",
    "Laitz, 5th":    "laitz_2023",
    "Mount":         "mount_2020",
    "Roig Francoli, 3rd": "roig_francoli_2020",
}

# ── BERT support layer ────────────────────────────────────────────────────────
# BAAI/bge-large-en-v1.5 is the right tool for this task.
# Context windows are 30–80 tokens (±2 sentences), well within BGE-large's
# strength.
BERT_MODEL             = "BAAI/bge-large-en-v1.5"
BERT_DEVICE            = "auto"        # "cuda" | "cpu" | "auto"
BERT_OVERRIDES_NEUTRAL = True
BERT_UPGRADE_THRESHOLD = 0.50          # adjustable via --bert-upgrade-threshold
BERT_BATCH_SIZE        = 256           # suitable for a 24 GB GPU; reduce on smaller cards

# ── LLM framing classifier ───────────────────────────────────────────────────
# Llama-3.1-8B-Instruct in fp16 uses ~16 GB VRAM; BGE-large adds ~1.3 GB.
# fp16 is preferred over 4-bit NF4: it avoids the bitsandbytes dependency, and
# quantisation introduces a small but measurable accuracy regression on
# classification tasks.  Set LLM_CLASSIFIER_LOAD_4BIT = True on GPUs with less
# than ~18 GB of available VRAM.  vLLM is used when available (parallelises
# generation across CPU threads); falls back to HuggingFace transformers.

LLM_CLASSIFIER_MODEL    = "meta-llama/Llama-3.1-8B-Instruct"
LLM_CLASSIFIER_DEVICE   = "auto"
LLM_CLASSIFIER_LOAD_4BIT = False      # fp16 by default; True = NF4 4-bit (~5 GB)
LLM_BATCH_SIZE           = 8          # prompt+generation calls; larger uses more VRAM
LLM_MAX_NEW_TOKENS       = 8          # single-token label + brief confidence
LLM_CLASSIFIER_THRESHOLD = 0.70       # adjustable via --zero-shot-threshold
# Backward-compat alias so existing --zero-shot-threshold flag still works:
ZERO_SHOT_MODEL     = LLM_CLASSIFIER_MODEL
ZERO_SHOT_THRESHOLD = LLM_CLASSIFIER_THRESHOLD
ZERO_SHOT_BATCH     = LLM_BATCH_SIZE

# ── Analysis parameters ───────────────────────────────────────────────────────
N_CLUSTERS      = 8
COOC_THRESHOLD  = 2
TFIDF_MAX_FEATS = 1000
# TF-IDF keyword extraction (fig09 / 08_tfidf_keywords.csv) is disabled by
# default.  The log-odds analysis consistently surfaces piece titles, instrument
# names, and corpus-skew artifacts rather than genuine framing rhetoric because
# the ~300-document context-window corpus is too small for reliable term-level
# discrimination.
# Re-enable with --tfidf-keywords if needed for exploratory inspection.
TFIDF_KEYWORDS_ENABLED = False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  COMPOSER METADATA  (demographics + framing lexicons)                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ASSUMPTION DOCUMENTATION:
# "Dominant"     = Western European Classical canon (Ewell 2020, Citron 1993).
# "Marginalized" = Women composers, Black/BIPOC artists, non-Western traditions,
#                  LGBTQ+-identified artists, popular music in classical curricula.
# These are analytical categories, not value judgements.
# When BIO_DATA_PATH is set, the 'bipoc' and 'sex' columns from raw_df.csv
# directly supply the marginalized flag, replacing the hand-coded values below.

COMPOSER_METADATA = {
    # ── Baroque ───────────────────────────────────────────────────────────────
    "Johann Sebastian Bach":    {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"baroque",    "dominant":True, "marginalized":False,"margin_reason":None},
    "George Frideric Handel":   {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"baroque",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Antonio Vivaldi":          {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"baroque",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Henry Purcell":            {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"baroque",    "dominant":True, "marginalized":False,"margin_reason":None},
    "François Couperin":        {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"baroque",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Orlando de Lassus":        {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"renaissance","dominant":True, "marginalized":False,"margin_reason":None},
    "Johann Joseph Fux":        {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"baroque",    "dominant":True, "marginalized":False,"margin_reason":None},
    # ── Classical / Romantic ──────────────────────────────────────────────────
    "Wolfgang Amadeus Mozart":  {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"classical", "dominant":True, "marginalized":False,"margin_reason":None},
    "Ludwig van Beethoven":     {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"classical", "dominant":True, "marginalized":False,"margin_reason":None},
    "Franz Joseph Haydn":       {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"classical", "dominant":True, "marginalized":False,"margin_reason":None},
    "Franz Schubert":           {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Robert Schumann":          {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Johannes Brahms":          {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Frédéric Chopin":          {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Richard Wagner":           {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Pyotr Ilyich Tchaikovsky": {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Alexander Borodin":        {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Sergei Rachmaninov":       {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Richard Strauss":          {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Claude Debussy":           {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"modern",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Maurice Ravel":            {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"modern",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Gustav Holst":             {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"modern",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Georges Bizet":            {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Giacomo Puccini":          {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Giuseppe Verdi":           {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Gioachino Rossini":        {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Giuseppe Giordani":        {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"classical", "dominant":True, "marginalized":False,"margin_reason":None},
    "Friedrich Kulhau":         {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"classical", "dominant":True, "marginalized":False,"margin_reason":None},
    "Alexander Gretchaninoff":  {"gender":"male", "ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    # ── 20th-C / Modernism / Minimalism ──────────────────────────────────────
    "Arnold Schoenberg":        {"gender":"male", "ethnicity":"White European","tradition":"contemporary_classical","era":"modern",      "dominant":True, "marginalized":False,"margin_reason":None},
    "Alban Berg":               {"gender":"male", "ethnicity":"White European","tradition":"contemporary_classical","era":"modern",      "dominant":True, "marginalized":False,"margin_reason":None},
    "Anton Webern":             {"gender":"male", "ethnicity":"White European","tradition":"contemporary_classical","era":"modern",      "dominant":True, "marginalized":False,"margin_reason":None},
    "Igor Stravinsky":          {"gender":"male", "ethnicity":"White European","tradition":"contemporary_classical","era":"modern",      "dominant":True, "marginalized":False,"margin_reason":None},
    "Béla Bartók":              {"gender":"male", "ethnicity":"White European","tradition":"contemporary_classical","era":"modern",      "dominant":True, "marginalized":False,"margin_reason":None},
    "György Ligeti":            {"gender":"male", "ethnicity":"White European","tradition":"contemporary_classical","era":"modern",      "dominant":True, "marginalized":False,"margin_reason":None},
    "Elliott Carter":           {"gender":"male", "ethnicity":"White American","tradition":"contemporary_classical","era":"modern",      "dominant":True, "marginalized":False,"margin_reason":None},
    "Aaron Copland":            {"gender":"male", "ethnicity":"White American","tradition":"contemporary_classical","era":"modern",      "dominant":True, "marginalized":False,"margin_reason":None},
    "Samuel Barber":            {"gender":"male", "ethnicity":"White American","tradition":"contemporary_classical","era":"modern",      "dominant":True, "marginalized":False,"margin_reason":None},
    "Philip Glass":             {"gender":"male", "ethnicity":"White American","tradition":"contemporary_classical","era":"contemporary","dominant":True, "marginalized":False,"margin_reason":None},
    "Steve Reich":              {"gender":"male", "ethnicity":"White American","tradition":"contemporary_classical","era":"contemporary","dominant":True, "marginalized":False,"margin_reason":None},
    "La Monte Young":           {"gender":"male", "ethnicity":"White American","tradition":"contemporary_classical","era":"contemporary","dominant":True, "marginalized":False,"margin_reason":None},
    "Terry Riley":              {"gender":"male", "ethnicity":"White American","tradition":"contemporary_classical","era":"contemporary","dominant":True, "marginalized":False,"margin_reason":None},
    # ── Film Music ────────────────────────────────────────────────────────────
    "John Williams":            {"gender":"male", "ethnicity":"White American","tradition":"film","era":"contemporary","dominant":True, "marginalized":False,"margin_reason":None},
    "Henry Mancini":            {"gender":"male", "ethnicity":"White American","tradition":"film","era":"contemporary","dominant":True, "marginalized":False,"margin_reason":None},
    "Michael Giacchino":        {"gender":"male", "ethnicity":"White American","tradition":"film","era":"contemporary","dominant":True, "marginalized":False,"margin_reason":None},
    "Alan Menken":              {"gender":"male", "ethnicity":"White American","tradition":"film","era":"contemporary","dominant":True, "marginalized":False,"margin_reason":None},
    # ── Women Composers ───────────────────────────────────────────────────────
    "Clara Schumann":           {"gender":"female","ethnicity":"White European","tradition":"classical_canon","era":"romantic",   "dominant":False,"marginalized":True,"margin_reason":"Woman composer in 19th-c patriarchal context"},
    "Zenobia Powell Perry":     {"gender":"female","ethnicity":"Black American","tradition":"contemporary_classical","era":"modern","dominant":False,"marginalized":True,"margin_reason":"Black woman composer; doubly marginalized"},
    "Tui St. George Tucker":    {"gender":"female","ethnicity":"White American","tradition":"contemporary_classical","era":"modern","dominant":False,"marginalized":True,"margin_reason":"Woman composer; historically overlooked"},
    "Joyce Solomon":            {"gender":"female","ethnicity":"unknown",       "tradition":"contemporary_classical","era":"modern","dominant":False,"marginalized":True,"margin_reason":"Woman composer; historically overlooked"},
    "Annie Harrison":           {"gender":"female","ethnicity":"White European","tradition":"classical_canon","era":"romantic",   "dominant":False,"marginalized":True,"margin_reason":"Woman composer; underrepresented"},
    "Maria Wolowska Szymanowska":{"gender":"female","ethnicity":"White European","tradition":"classical_canon","era":"romantic","dominant":False,"marginalized":True,"margin_reason":"Woman composer; Polish, historically overlooked in Western canon"},
    "Josephine Lang":           {"gender":"female","ethnicity":"White European","tradition":"classical_canon","era":"romantic","dominant":False,"marginalized":True,"margin_reason":"Woman composer; historically overlooked in Western canon"},
    # ── Joseph Bologne ────────────────────────────────────────────────────────
    "Joseph Bologne":           {"gender":"male","ethnicity":"Black European","tradition":"classical_canon","era":"classical","dominant":False,"marginalized":True,"margin_reason":"Afro-Caribbean in 18th-c European classical world"},
    "Joseph Bologne Chevalier de Saint-Georges": {"gender":"male","ethnicity":"Black European","tradition":"classical_canon","era":"classical","dominant":False,"marginalized":True,"margin_reason":"Afro-Caribbean in 18th-c European classical world (alias)"},
    # ── Jazz ──────────────────────────────────────────────────────────────────
    "Duke Ellington":           {"gender":"male",  "ethnicity":"Black American","tradition":"jazz","era":"modern",      "dominant":False,"marginalized":True,"margin_reason":"Black American; jazz peripheral in classical pedagogy"},
    "Charlie Parker":           {"gender":"male",  "ethnicity":"Black American","tradition":"jazz","era":"modern",      "dominant":False,"marginalized":True,"margin_reason":"Black American jazz artist"},
    "Miles Davis":              {"gender":"male",  "ethnicity":"Black American","tradition":"jazz","era":"modern",      "dominant":False,"marginalized":True,"margin_reason":"Black American jazz artist"},
    "Thelonious Monk":          {"gender":"male",  "ethnicity":"Black American","tradition":"jazz","era":"modern",      "dominant":False,"marginalized":True,"margin_reason":"Black American jazz artist"},
    "John Coltrane":            {"gender":"male",  "ethnicity":"Black American","tradition":"jazz","era":"modern",      "dominant":False,"marginalized":True,"margin_reason":"Black American jazz artist"},
    "Count Basie":              {"gender":"male",  "ethnicity":"Black American","tradition":"jazz","era":"modern",      "dominant":False,"marginalized":True,"margin_reason":"Black American jazz artist"},
    "Billy Taylor":             {"gender":"male",  "ethnicity":"Black American","tradition":"jazz","era":"modern",      "dominant":False,"marginalized":True,"margin_reason":"Black American jazz pianist"},
    "Norah Jones":              {"gender":"female","ethnicity":"Mixed (Indian-American)","tradition":"jazz","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Woman of mixed heritage; jazz/pop crossover"},
    "Bart Howard":              {"gender":"male",  "ethnicity":"White American","tradition":"jazz","era":"modern",      "dominant":False,"marginalized":True,"margin_reason":"LGBTQ+ cabaret/jazz songwriter"},
    "Michael Bublé":            {"gender":"male",  "ethnicity":"White Canadian","tradition":"jazz","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    # ── Broadway / Musical Theatre ────────────────────────────────────────────
    "George Gershwin":          {"gender":"male","ethnicity":"White American","tradition":"musical_theatre","era":"modern",      "dominant":False,"marginalized":False,"margin_reason":None},
    "Cole Porter":              {"gender":"male","ethnicity":"White American","tradition":"musical_theatre","era":"modern",      "dominant":False,"marginalized":True, "margin_reason":"LGBTQ+ composer; Broadway peripheral in classical theory"},
    "Irving Berlin":            {"gender":"male","ethnicity":"White American","tradition":"musical_theatre","era":"modern",      "dominant":False,"marginalized":False,"margin_reason":None},
    "Jerome Kern":              {"gender":"male","ethnicity":"White American","tradition":"musical_theatre","era":"modern",      "dominant":False,"marginalized":False,"margin_reason":None},
    "Richard Rodgers":          {"gender":"male","ethnicity":"White American","tradition":"musical_theatre","era":"modern",      "dominant":False,"marginalized":False,"margin_reason":None},
    "Lorenz Hart":              {"gender":"male","ethnicity":"White American","tradition":"musical_theatre","era":"modern",      "dominant":False,"marginalized":True, "margin_reason":"LGBTQ+ lyricist"},
    "Frank Loesser":            {"gender":"male","ethnicity":"White American","tradition":"musical_theatre","era":"modern",      "dominant":False,"marginalized":False,"margin_reason":None},
    "John Kander":              {"gender":"male","ethnicity":"White American","tradition":"musical_theatre","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Hoagy Carmichael":         {"gender":"male","ethnicity":"White American","tradition":"musical_theatre","era":"modern",      "dominant":False,"marginalized":False,"margin_reason":None},
    "Trevor Nunn":              {"gender":"male","ethnicity":"White British", "tradition":"musical_theatre","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    # ── R&B / Soul / Funk / Motown ────────────────────────────────────────────
    "Stevie Wonder":            {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American; R&B peripheral in classical theory"},
    "Marvin Gaye":              {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American soul artist"},
    "Otis Redding":             {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American soul artist"},
    "Michael Jackson":          {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American pop artist"},
    "Bill Withers":             {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American soul artist"},
    "Rick James":               {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American funk artist"},
    "Lamont Dozier":            {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American Motown songwriter"},
    "Brian Holland":            {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American Motown songwriter"},
    "Ronald White":             {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American Motown songwriter"},
    "Maurice White":            {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American (Earth, Wind & Fire)"},
    "Al McKay":                 {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American (Earth, Wind & Fire)"},
    "Barry Gordy":              {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American Motown founder"},
    "Reggie Calloway":          {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American R&B artist"},
    "Frankie Lymon":            {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"modern",      "dominant":False,"marginalized":True,"margin_reason":"Black American doo-wop artist"},
    "Lionel Richie":            {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American pop/R&B artist"},
    "Alicia Keys":              {"gender":"female","ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American woman artist"},
    "Beyoncé":                  {"gender":"female","ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American woman artist"},
    "The Supremes":             {"gender":"group", "ethnicity":"Black American","tradition":"r_and_b_soul","era":"modern",      "dominant":False,"marginalized":True,"margin_reason":"Black American women's group (Motown)"},
    "CeeLo Green":              {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American R&B/hip-hop artist"},
    "John Stephens":            {"gender":"male",  "ethnicity":"Black American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American (John Legend)"},
    "Shawn Carter":             {"gender":"male",  "ethnicity":"Black American","tradition":"hip_hop",     "era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American hip-hop (Jay-Z)"},
    "Bo Diddley":               {"gender":"male",  "ethnicity":"Black American","tradition":"rock_pop",    "era":"modern",      "dominant":False,"marginalized":True,"margin_reason":"Black American rock'n'roll pioneer"},
    "Chuck Berry":              {"gender":"male",  "ethnicity":"Black American","tradition":"rock_pop",    "era":"modern",      "dominant":False,"marginalized":True,"margin_reason":"Black American rock'n'roll pioneer"},
    "Bob Marley":               {"gender":"male",  "ethnicity":"Black Jamaican","tradition":"reggae",      "era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black Jamaican; non-Western tradition"},
    "Ramón Luis a.k.a. Daddy Yankee Ayala Rodríguez": {"gender":"male","ethnicity":"Hispanic/Puerto Rican","tradition":"latin","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Latin American; reggaeton peripheral in classical theory"},
    # ── Women Pop / Rock ──────────────────────────────────────────────────────
    "Adele":                    {"gender":"female","ethnicity":"White British", "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Woman artist in popular music"},
    "Sara Bareilles":           {"gender":"female","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Woman artist in popular music"},
    "Cyndi Lauper":             {"gender":"female","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Woman artist in popular music"},
    "Fiona Apple":              {"gender":"female","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Woman artist in popular music"},
    "Kesha":                    {"gender":"female","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Woman artist in popular music"},
    "Christina Perri":          {"gender":"female","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Woman artist in popular music"},
    "Ariana Grande":            {"gender":"female","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Woman artist in popular music"},
    "Meghan Trainor":           {"gender":"female","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Woman artist in popular music"},
    "Dolly Parton":             {"gender":"female","ethnicity":"White American","tradition":"country", "era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Woman artist; country peripheral in classical theory"},
    "Irene Cara":               {"gender":"female","ethnicity":"Hispanic/Black American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Woman of color artist"},
    "Allee Willis":             {"gender":"female","ethnicity":"White American","tradition":"r_and_b_soul","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Woman songwriter (LGBTQ+)"},
    # ── Rock / Pop (mainstream) ───────────────────────────────────────────────
    "Paul McCartney":           {"gender":"male","ethnicity":"White British", "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "John Lennon":              {"gender":"male","ethnicity":"White British", "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "The Beatles":              {"gender":"group","ethnicity":"White British","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Ed Sheeran":               {"gender":"male","ethnicity":"White British", "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Bruce Springsteen":        {"gender":"male","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Billy Joel":               {"gender":"male","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Brian Wilson":             {"gender":"male","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Jason Mraz":               {"gender":"male","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "James Arthur":             {"gender":"male","ethnicity":"White British", "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Shawn Mendes":             {"gender":"male","ethnicity":"White Canadian","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Jon Bon Jovi":             {"gender":"male","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Justin Bieber":            {"gender":"male","ethnicity":"White Canadian","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Hall and Oates":           {"gender":"group","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Robert Lamm":              {"gender":"male","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Max Martin":               {"gender":"male","ethnicity":"White Swedish", "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Lukasz Gottwald":          {"gender":"male","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Michael Masser":           {"gender":"male","ethnicity":"White American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Toby Gad":                 {"gender":"male","ethnicity":"unknown",       "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Calvin Harris":            {"gender":"male","ethnicity":"White Scottish","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "George Michael":           {"gender":"male","ethnicity":"White British", "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"LGBTQ+ artist"},
    "Elton John":               {"gender":"male","ethnicity":"White British", "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"LGBTQ+ artist"},
    "Freddie Mercury":          {"gender":"male","ethnicity":"South Asian (Parsi)","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"LGBTQ+ artist of Indian-Parsi heritage"},
    "Ian Gillan":               {"gender":"male","ethnicity":"White British", "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Ritchie Blackmore":        {"gender":"male","ethnicity":"White British", "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Ian Paice":                {"gender":"male","ethnicity":"White British", "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Jon Lord":                 {"gender":"male","ethnicity":"White British", "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Roger Glover":             {"gender":"male","ethnicity":"White British", "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Bruno Mars":               {"gender":"male","ethnicity":"Mixed (Filipino/Puerto Rican/Jewish)","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Artist of mixed Pacific Islander/Latino heritage"},
    # ── Folk / Ragtime / Blues / Experimental ────────────────────────────────
    "William Christopher Handy":{"gender":"male","ethnicity":"Black American","tradition":"blues","era":"modern","dominant":False,"marginalized":True,"margin_reason":"Black American blues pioneer"},
    "Charles Leslie Johnson":   {"gender":"male","ethnicity":"White American","tradition":"ragtime","era":"modern","dominant":False,"marginalized":False,"margin_reason":None},
    "Jaromír Vejvoda":          {"gender":"male","ethnicity":"White European","tradition":"folk_popular","era":"modern","dominant":False,"marginalized":False,"margin_reason":None},
    "Samuel Francis Smith":     {"gender":"male","ethnicity":"White American","tradition":"folk_popular","era":"romantic","dominant":False,"marginalized":False,"margin_reason":None},
    "Too Many Zooz":            {"gender":"group","ethnicity":"mixed","tradition":"contemporary_experimental","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Street-performance tradition; non-classical"},
    "Jan Ladislav Dussek":      {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"classical","dominant":True,"marginalized":False,"margin_reason":None},
    # ── Bands ─────────────────────────────────────────────────────────────────
    "U2":           {"gender":"group","ethnicity":"White Irish",           "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Beach Boys":   {"gender":"group","ethnicity":"White American",        "tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},
    "Fleetwood Mac":{"gender":"group","ethnicity":"White British/American","tradition":"rock_pop","era":"contemporary","dominant":False,"marginalized":False,"margin_reason":None},

    # ══════════════════════════════════════════════════════════════════════════
    # Additional composers appearing in the corpus
    # ══════════════════════════════════════════════════════════════════════════

    # ── Western Canon — Baroque ───────────────────────────────────────────────
    "Arcangelo Corelli":        {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"baroque",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Domenico Scarlatti":       {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"baroque",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Alessandro Scarlatti":     {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"baroque",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Georg Philipp Telemann":   {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"baroque",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Jean-Philippe Rameau":     {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"baroque",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Claudio Monteverdi":       {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"baroque",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Johann Pachelbel":         {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"baroque",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Dieterich Buxtehude":      {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"baroque",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Giovanni Battista Pergolesi":{"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"baroque",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Heinrich Schütz":          {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"baroque",    "dominant":True, "marginalized":False,"margin_reason":None},

    # ── Western Canon — Renaissance ───────────────────────────────────────────
    "Josquin des Prez":         {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"renaissance","dominant":True, "marginalized":False,"margin_reason":None},
    "Giovanni Pierluigi da Palestrina":{"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"renaissance","dominant":True,"marginalized":False,"margin_reason":None},
    "Thomas Tallis":            {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"renaissance","dominant":True, "marginalized":False,"margin_reason":None},
    "William Byrd":             {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"renaissance","dominant":True, "marginalized":False,"margin_reason":None},
    "Orlando de Lassus":        {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"renaissance","dominant":True, "marginalized":False,"margin_reason":None},
    "Thomas Morley":            {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"renaissance","dominant":True, "marginalized":False,"margin_reason":None},

    # ── Western Canon — Classical / Early Romantic ────────────────────────────
    "Johann Christian Bach":    {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"classical", "dominant":True, "marginalized":False,"margin_reason":None},
    "Carl Philipp Emanuel Bach":{"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"classical", "dominant":True, "marginalized":False,"margin_reason":None},
    "Felix Mendelssohn":        {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Franz Liszt":              {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Hector Berlioz":           {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Gustav Mahler":            {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Edvard Grieg":             {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "César Franck":             {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Nikolai Rimsky-Korsakov":  {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Modest Petrovich Mussorgsky":{"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"romantic","dominant":True, "marginalized":False,"margin_reason":None},
    "Camille Saint-Saëns":      {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Gabriel Fauré":            {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Anton Bruckner":           {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Hugo Wolf":                {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Antonín Dvořák":           {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Bedřich Smetana":          {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Bedrich Smetana":          {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":True, "marginalized":False,"margin_reason":None},
    "Sergei Prokofiev":         {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"modern",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Dmitri Shostakovich":      {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"modern",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Jean Sibelius":            {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"modern",    "dominant":True, "marginalized":False,"margin_reason":None},
    "Aleksandr Scriabin":       {"gender":"male","ethnicity":"White European","tradition":"classical_canon","era":"modern",    "dominant":True, "marginalized":False,"margin_reason":None},

    # ── Contemporary Classical ────────────────────────────────────────────────
    "Olivier Messiaen":         {"gender":"male","ethnicity":"White European","tradition":"contemporary_classical","era":"modern",       "dominant":True,"marginalized":False,"margin_reason":None},
    "Pierre Boulez":            {"gender":"male","ethnicity":"White European","tradition":"contemporary_classical","era":"modern",       "dominant":True,"marginalized":False,"margin_reason":None},
    "John Cage":                {"gender":"male","ethnicity":"White American","tradition":"contemporary_classical","era":"modern",       "dominant":True,"marginalized":False,"margin_reason":None},
    "Paul Hindemith":           {"gender":"male","ethnicity":"White European","tradition":"contemporary_classical","era":"modern",       "dominant":True,"marginalized":False,"margin_reason":None},
    "Darius Milhaud":           {"gender":"male","ethnicity":"White European","tradition":"contemporary_classical","era":"modern",       "dominant":True,"marginalized":False,"margin_reason":None},
    "Charles Ives":             {"gender":"male","ethnicity":"White American","tradition":"contemporary_classical","era":"modern",       "dominant":True,"marginalized":False,"margin_reason":None},
    "Milton Babbitt":           {"gender":"male","ethnicity":"White American","tradition":"contemporary_classical","era":"modern",       "dominant":True,"marginalized":False,"margin_reason":None},
    "George Crumb":             {"gender":"male","ethnicity":"White American","tradition":"contemporary_classical","era":"contemporary","dominant":True,"marginalized":False,"margin_reason":None},
    "Luciano Berio":            {"gender":"male","ethnicity":"White European","tradition":"contemporary_classical","era":"modern",       "dominant":True,"marginalized":False,"margin_reason":None},
    "Karlheinz Stockhausen":    {"gender":"male","ethnicity":"White European","tradition":"contemporary_classical","era":"modern",       "dominant":True,"marginalized":False,"margin_reason":None},
    "Krzysztof Penderecki":     {"gender":"male","ethnicity":"White European","tradition":"contemporary_classical","era":"contemporary","dominant":True,"marginalized":False,"margin_reason":None},
    "Morton Feldman":           {"gender":"male","ethnicity":"White American","tradition":"contemporary_classical","era":"modern",       "dominant":True,"marginalized":False,"margin_reason":None},
    "Henry Cowell":             {"gender":"male","ethnicity":"White American","tradition":"contemporary_classical","era":"modern",       "dominant":True,"marginalized":False,"margin_reason":None},

    # ── Marginalized: Women Composers ────────────────────────────────────────
    "Fanny Mendelssohn Hensel": {"gender":"female","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":False,"marginalized":True,"margin_reason":"Woman composer; historically overlooked in favour of brother Felix"},
    "Louise Farrenc":           {"gender":"female","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":False,"marginalized":True,"margin_reason":"Woman composer; underrepresented in Western canon"},
    "Louise Reichardt":         {"gender":"female","ethnicity":"White European","tradition":"classical_canon","era":"classical", "dominant":False,"marginalized":True,"margin_reason":"Woman composer; historically overlooked"},
    "Amy Beach":                {"gender":"female","ethnicity":"White American","tradition":"classical_canon","era":"romantic",  "dominant":False,"marginalized":True,"margin_reason":"Woman composer; first major American woman symphonist"},
    "Lili Boulanger":           {"gender":"female","ethnicity":"White European","tradition":"classical_canon","era":"modern",    "dominant":False,"marginalized":True,"margin_reason":"Woman composer; died young, historically underrepresented"},
    "Tania León":               {"gender":"female","ethnicity":"Hispanic/Cuban","tradition":"contemporary_classical","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Latina woman composer; doubly marginalized"},
    "Ruth Crawford Seeger":     {"gender":"female","ethnicity":"White American","tradition":"contemporary_classical","era":"modern",    "dominant":False,"marginalized":True,"margin_reason":"Woman composer; pioneer of American ultramodernism"},
    "Élisabeth Jacquet de la Guerre":{"gender":"female","ethnicity":"White European","tradition":"classical_canon","era":"baroque","dominant":False,"marginalized":True,"margin_reason":"Woman composer; Baroque era, historically overlooked"},
    "Margaret Bonds":           {"gender":"female","ethnicity":"Black American","tradition":"classical_canon","era":"modern",    "dominant":False,"marginalized":True,"margin_reason":"Black American woman composer; doubly marginalized"},
    "Augusta Browne":           {"gender":"female","ethnicity":"White American","tradition":"classical_canon","era":"romantic",  "dominant":False,"marginalized":True,"margin_reason":"Woman composer; 19th-century America, historically overlooked"},
    "Josephine Frances Hummell":{"gender":"female","ethnicity":"White European","tradition":"classical_canon","era":"romantic",  "dominant":False,"marginalized":True,"margin_reason":"Woman composer; historically overlooked"},
    "Josephine Frances L. Hummell":{"gender":"female","ethnicity":"White European","tradition":"classical_canon","era":"romantic","dominant":False,"marginalized":True,"margin_reason":"Woman composer; historically overlooked"},
    "Barbara Strozzi":          {"gender":"female","ethnicity":"White European","tradition":"classical_canon","era":"baroque",   "dominant":False,"marginalized":True,"margin_reason":"Woman composer; Baroque Venice, historically overlooked"},
    "Marianna Martines":        {"gender":"female","ethnicity":"White European","tradition":"classical_canon","era":"classical", "dominant":False,"marginalized":True,"margin_reason":"Woman composer; Classical era, historically overlooked"},
    "Sofia Gubaidulina":        {"gender":"female","ethnicity":"mixed (Tatar-Russian)","tradition":"contemporary_classical","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Woman composer; Soviet-era marginalization + ethnic minority"},
    "Chen Yi":                  {"gender":"female","ethnicity":"Asian (Chinese-American)","tradition":"contemporary_classical","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Chinese-American woman composer; doubly marginalized"},

    # ── Marginalized: Black / BIPOC Classical Composers ───────────────────────
    "Florence Price":           {"gender":"female","ethnicity":"Black American","tradition":"classical_canon","era":"modern",    "dominant":False,"marginalized":True,"margin_reason":"Black American woman composer; doubly marginalized"},
    "Scott Joplin":             {"gender":"male",  "ethnicity":"Black American","tradition":"ragtime","era":"modern",           "dominant":False,"marginalized":True,"margin_reason":"Black American ragtime pioneer"},
    "Samuel Coleridge-Taylor":  {"gender":"male",  "ethnicity":"Black British", "tradition":"classical_canon","era":"romantic", "dominant":False,"marginalized":True,"margin_reason":"Black British composer; mixed-race heritage, historically underrepresented"},
    "Hale Smith":               {"gender":"male",  "ethnicity":"Black American","tradition":"contemporary_classical","era":"modern","dominant":False,"marginalized":True,"margin_reason":"Black American composer"},
    "Ulysses Kay":              {"gender":"male",  "ethnicity":"Black American","tradition":"contemporary_classical","era":"modern","dominant":False,"marginalized":True,"margin_reason":"Black American composer"},
    "Adolphus Hailstork":       {"gender":"male",  "ethnicity":"Black American","tradition":"contemporary_classical","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American composer"},
    "R. Nathaniel Dett":        {"gender":"male",  "ethnicity":"Black American","tradition":"classical_canon","era":"modern",    "dominant":False,"marginalized":True,"margin_reason":"Black Canadian-American composer; historically underrepresented"},
    "Julius Eastman":           {"gender":"male",  "ethnicity":"Black American","tradition":"contemporary_classical","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American gay composer; doubly marginalized"},
    "William Grant Still":      {"gender":"male",  "ethnicity":"Black American","tradition":"classical_canon","era":"modern",    "dominant":False,"marginalized":True,"margin_reason":"Black American; first Black American to conduct major US orchestra"},
    "Dorothy Rudd Moore":       {"gender":"female","ethnicity":"Black American","tradition":"contemporary_classical","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American woman composer; doubly marginalized"},
    "Undine Smith Moore":       {"gender":"female","ethnicity":"Black American","tradition":"classical_canon","era":"modern",    "dominant":False,"marginalized":True,"margin_reason":"Black American woman composer; doubly marginalized"},
    "Nkeiru Okoye":             {"gender":"female","ethnicity":"Black American","tradition":"contemporary_classical","era":"contemporary","dominant":False,"marginalized":True,"margin_reason":"Black American woman composer; doubly marginalized"},
}

COMPOSER_ALIASES = {
    # Possessives (Unicode curly + straight apostrophe)
    "Debussy\u2019s":"Claude Debussy",  "Debussy's":"Claude Debussy",
    "Bach\u2019s":   "Johann Sebastian Bach",  "Bach's":"Johann Sebastian Bach",
    "Beethoven\u2019s":"Ludwig van Beethoven", "Beethoven's":"Ludwig van Beethoven",
    "Mozart\u2019s": "Wolfgang Amadeus Mozart","Mozart's": "Wolfgang Amadeus Mozart",
    "Schubert\u2019s":"Franz Schubert",         "Schubert's":"Franz Schubert",
    "Brahms\u2019s": "Johannes Brahms",         "Brahms's": "Johannes Brahms",
    "Chopin\u2019s": "Frédéric Chopin",         "Chopin's": "Frédéric Chopin",
    "Handel\u2019s": "George Frideric Handel",  "Handel's": "George Frideric Handel",
    "Ravel\u2019s":  "Maurice Ravel",           "Ravel's":  "Maurice Ravel",
    "Vivaldi\u2019s":"Antonio Vivaldi",          "Vivaldi's":"Antonio Vivaldi",
    "Wagner\u2019s": "Richard Wagner",          "Wagner's": "Richard Wagner",
    # Last-name-only NER short forms
    "Debussy":   "Claude Debussy",
    "Ravel":     "Maurice Ravel",
    "Brahms":    "Johannes Brahms",
    "Schubert":  "Franz Schubert",
    "Schumann":  "Robert Schumann",
    "Handel":    "George Frideric Handel",
    "Vivaldi":   "Antonio Vivaldi",
    "Haydn":     "Franz Joseph Haydn",
    "Chopin":    "Frédéric Chopin",
    "Wagner":    "Richard Wagner",
    "Verdi":     "Giuseppe Verdi",
    "Puccini":   "Giacomo Puccini",
    "Rossini":   "Gioachino Rossini",
    "Bartók":    "Béla Bartók",
    "Ligeti":    "György Ligeti",
    "Webern":    "Anton Webern",
    "Schoenberg":"Arnold Schoenberg",
    "Stravinsky":"Igor Stravinsky",
    "Ellington": "Duke Ellington",
    "Parker":    "Charlie Parker",
    "Monk":      "Thelonious Monk",
    "Coltrane":  "John Coltrane",
    "Davis":     "Miles Davis",
    "Gershwin":  "George Gershwin",
    "Copland":   "Aaron Copland",
    "Barber":    "Samuel Barber",
    "Glass":     "Philip Glass",
    "Reich":     "Steve Reich",
    "Holst":     "Gustav Holst",
    "Bizet":     "Georges Bizet",
    "Borodin":   "Alexander Borodin",
    "Rachmaninov": "Sergei Rachmaninov",
    "Rachmaninoff":"Sergei Rachmaninov",
    "Tchaikovsky":"Pyotr Ilyich Tchaikovsky",
    # Standard abbreviations
    "W. A. Mozart":  "Wolfgang Amadeus Mozart",
    "W.A. Mozart":   "Wolfgang Amadeus Mozart",
    "J.S. Bach":     "Johann Sebastian Bach",
    "J. S. Bach":    "Johann Sebastian Bach",
    "Bach":          "Johann Sebastian Bach",
    "Mozart":        "Wolfgang Amadeus Mozart",
    "Beethoven":     "Ludwig van Beethoven",
    # Biographical aliases
    "Beyoncé Knowles":"Beyoncé",
    "Joseph Bologne Chevalier de Saint-Georges":"Joseph Bologne",
    # piece_attributions bleed (short composer strings from that field)
    "Debussy, Préludes, Book II":"Claude Debussy",
    "Debussy, Danses":           "Claude Debussy",
    "Debussy":                   "Claude Debussy",

    # ── Full-name accent-variant aliases ──────────────────────────────────────
    # The canonical names in JSONL files come from unique_composers.csv, which
    # stores unaccented forms.  These aliases bridge the gap so that
    # COMPOSER_METADATA lookups succeed for composers whose metadata key uses
    # the accented Unicode form.
    "Frederic Chopin":            "Frédéric Chopin",
    "Frederic Francois Chopin":   "Frédéric Chopin",
    "Bela Bartok":                "Béla Bartók",
    "Gyorgy Ligeti":              "György Ligeti",
    "Francois Couperin":          "François Couperin",
    "Antonin Dvorak":             "Antonín Dvořák",
    "Antonín Dvořák":             "Antonín Dvořák",
    "Leos Janacek":               "Leoš Janáček",
    "Zoltan Kodaly":              "Zoltán Kodály",
    "Georg Friedrich Handel":     "George Frideric Handel",
    "George Frederic Handel":     "George Frideric Handel",
    "Georg Frideric Handel":      "George Frideric Handel",
    "Pyotr Tchaikovsky":          "Pyotr Ilyich Tchaikovsky",
    "Peter Tchaikovsky":          "Pyotr Ilyich Tchaikovsky",
    "Piotr Tchaikovsky":          "Pyotr Ilyich Tchaikovsky",
    "Sergei Rachmaninoff":        "Sergei Rachmaninov",
    "Serge Rachmaninov":          "Sergei Rachmaninov",
    "Gioacchino Rossini":         "Gioachino Rossini",
    "Arcangelo Corelli":          "Arcangelo Corelli",
    "Carl Philipp Emanuel Bach":  "Carl Philipp Emanuel Bach",
    "C.P.E. Bach":                "Carl Philipp Emanuel Bach",
    "C. P. E. Bach":              "Carl Philipp Emanuel Bach",
    "Dmitri Shostakovich":        "Dmitri Shostakovich",
    "Dmitry Shostakovich":        "Dmitri Shostakovich",
    "Heitor Villa Lobos":         "Heitor Villa-Lobos",
    "Cesar Franck":               "César Franck",
    "Camille Saint-Saens":        "Camille Saint-Saëns",
    "Tania Leon":                 "Tania León",
    "Jean-Philippe Rameau":       "Jean-Philippe Rameau",
    "Jean Philippe Rameau":       "Jean-Philippe Rameau",
    "Elisabeth Jacquet de la Guerre": "Élisabeth Jacquet de la Guerre",
    "Helene de Montgeroult":      "Hélène de Montgeroult",
    "Helene Montgeroult":         "Hélène de Montgeroult",
    "Hélène Montgeroult":         "Hélène de Montgeroult",
    "Maria Teresa Agnesi":        "Maria Teresa d'Agnesi",
    "Modest Mussorgsky":          "Modest Petrovich Mussorgsky",
    "Modest Moussorgsky":         "Modest Petrovich Mussorgsky",

    # ── Name-form / partial-name variants ─────────────────────────────────────
    # Surname-only shortcuts (NER often extracts bare surname)
    "Corelli":                    "Arcangelo Corelli",
    "D. Scarlatti":               "Domenico Scarlatti",
    "Scarlatti":                  "Domenico Scarlatti",
    "Telemann":                   "Georg Philipp Telemann",
    "Pergolesi":                  "Giovanni Battista Pergolesi",
    "Rameau":                     "Jean-Philippe Rameau",
    "Monteverdi":                 "Claudio Monteverdi",
    "Palestrina":                 "Giovanni Pierluigi da Palestrina",
    "Josquin":                    "Josquin des Prez",
    "Josquin des Pres":           "Josquin des Prez",
    "Josquin Desprez":            "Josquin des Prez",
    "Grieg":                      "Edvard Grieg",
    "Mahler":                     "Gustav Mahler",
    "Liszt":                      "Franz Liszt",
    "Berlioz":                    "Hector Berlioz",
    "Messiaen":                   "Olivier Messiaen",
    "Boulez":                     "Pierre Boulez",
    "Hindemith":                  "Paul Hindemith",
    "Milhaud":                    "Darius Milhaud",
    "Cage":                       "John Cage",
    "Stockhausen":                "Karlheinz Stockhausen",
    "Penderecki":                 "Krzysztof Penderecki",
    "Joplin":                     "Scott Joplin",
    "Prokofiev":                  "Sergei Prokofiev",
    "Shostakovich":               "Dmitri Shostakovich",
    "Mussorgsky":                 "Modest Petrovich Mussorgsky",

    # Partial/alternate first names and married/birth name variants
    "Joseph Haydn":               "Franz Joseph Haydn",
    "Sebastian Bach":             "Johann Sebastian Bach",
    "Johann C. Bach":             "Johann Christian Bach",
    "J. C. Bach":                 "Johann Christian Bach",
    "Felix Mendelssohn Bartholdy":"Felix Mendelssohn",
    "Fanny Hensel":               "Fanny Mendelssohn Hensel",
    "Fanny Mendelssohn":          "Fanny Mendelssohn Hensel",
    "Clara Wieck":                "Clara Schumann",
    "Clara Wieck Schumann":       "Clara Schumann",
    "Florence B. Price":          "Florence Price",
    "Ruth Crawford":              "Ruth Crawford Seeger",
    "Nathaniel Dett":             "R. Nathaniel Dett",
    "R. N. Dett":                 "R. Nathaniel Dett",
    "William Grant Still":        "William Grant Still",  # guard

    # ── Non-breaking / narrow-no-break space variants (Unicode U+00A0, U+202F) ─
    # These appear when PDF extraction preserves special spaces; after newline
    # stripping the names normalise to their standard form below.
    "Ben Kynard":                 "Ben Kynard",
    "Bernie Miller":              "Bernie Miller",
    "Bedrich Smetana":            "Bedřich Smetana",
    "Josephine Frances Hummell":  "Josephine Frances Hummell",
}

# Standardise on "Clara Schumann" (the most common corpus form): collapse any
# "Clara Wieck Schumann" entry onto it at runtime so the aliases above redirect
# both variants to the same record.
_cm = COMPOSER_METADATA.pop("Clara Wieck Schumann", None)
if _cm and "Clara Schumann" not in COMPOSER_METADATA:
    COMPOSER_METADATA["Clara Schumann"] = _cm

TRADITION_LABELS = {
    "classical_canon":          "Classical Canon",
    "contemporary_classical":   "Contemporary Classical",
    "jazz":                     "Jazz",
    "r_and_b_soul":             "R&B / Soul / Funk",
    "rock_pop":                 "Rock / Pop",
    "musical_theatre":          "Musical Theatre",
    "film":                     "Film Music",
    "hip_hop":                  "Hip-Hop",
    "reggae":                   "Reggae",
    "latin":                    "Latin",
    "country":                  "Country",
    "ragtime":                  "Ragtime",
    "blues":                    "Blues",
    "folk_popular":             "Folk / Popular",
    "contemporary_experimental":"Contemporary Experimental",
    "non_western":              "Non-Western (inferred)",
    "unknown":                  "Unknown",
}

PALETTE = {
    "dominant":      "#4878CF",
    "marginalized":  "#E87B5A",
    "unclassified":  "#B0B8C4",  # grey — composer not yet in knowledge base
    "neutral":       "#8EA8C3",
    "female":        "#D95F02",
    "male":          "#1B9E77",
    "group":         "#7570B3",
}


def _composer_color(dominant: bool, marginalized: bool) -> str:
    """
    Three-tier colour selection.

    dominant=True             → PALETTE["dominant"]    (blue)
    dominant=False,
      marginalized=True       → PALETTE["marginalized"] (orange)
    dominant=False,
      marginalized=False      → PALETTE["unclassified"] (grey)

    The old binary  `dominant / else`  coding painted composers not yet in
    COMPOSER_METADATA (e.g. "Frederic Chopin" failing an accent lookup) in
    the marginalized colour — a false positive that has now been fixed.
    """
    if dominant:
        return PALETTE["dominant"]
    if marginalized:
        return PALETTE["marginalized"]
    return PALETTE["unclassified"]

FRAMING_LEXICONS = {
    "normative": [
        (r"\bstandard\b",1.5),(r"\bfundamental\b",1.5),(r"\bcommon practice\b",2.0),
        (r"\btraditional(ly)?\b",1.0),(r"\bcanonical\b",2.0),(r"\bessential\b",1.5),
        (r"\btypical(ly)?\b",1.0),(r"\bprimarily\b",0.8),(r"\bconventional(ly)?\b",1.2),
        (r"\bestablished\b",1.0),(r"\bfoundation\b",1.5),(r"\bprincip(al|le)\b",1.0),
        (r"\bdefault\b",1.0),(r"\bordinary\b",1.0),(r"\bregular(ly)?\b",0.8),
        (r"\bnorm(al|ative)?\b",1.0),(r"\bcore\b",1.2),(r"\buniversal\b",1.5),
        (r"\bmain\b",0.7),(r"\bclassical\b",0.5),
    ],
    "additive": [
        (r"\balso\b",1.0),(r"\bin addition\b",1.5),(r"\badditionally\b",1.5),
        (r"\bfurthermore\b",1.0),(r"\bas well\b",1.0),(r"\bmoreover\b",1.0),
        (r"\banother example\b",1.8),(r"\bother example\b",1.5),(r"\bsimilarly\b",0.8),
        (r"\blikewise\b",0.8),(r"\btoo\b",0.5),(r"\balongside\b",1.2),
        (r"\bwe (can also|may also|also find)\b",1.5),(r"\bcan be found\b",0.7),
        (r"\bis (also|another)\b",1.2),(r"\binclude[sd]?\b",0.5),(r"\bbeyond\b",0.7),
    ],
    "exceptional": [
        (r"\bunusual\b",2.0),(r"\brare(ly)?\b",1.8),(r"\bexotic\b",2.5),
        (r"\bunique\b",1.5),(r"\bspecial\b",1.0),(r"\bunlike\b",1.5),
        # 'different' omitted — ubiquitous in analytical prose; generates false positives.
        # 'contrast'/'in contrast' downweighted — analytical vocabulary, not a marker
        # of exceptional status.
        (r"\bexception(al)?\b",1.8),(r"\bin contrast\b",0.4),
        (r"\bnovel\b",1.0),(r"\bdistinctive\b",1.0),
        (r"\bunorthodox\b",2.0),(r"\batypical\b",2.0),(r"\birregular\b",1.5),
        (r"\bpeculiar\b",2.0),(r"\bunexpected\b",1.5),(r"\bsurpris\w+\b",1.0),
        (r"\bnotably\b",0.8),(r"\bremarkable\b",1.2),
    ],
    "corrective": [
        (r"\bhistorically (overlooked|underrepresented|excluded|marginalized|ignored)\b",3.0),
        (r"\bpreviously ignored\b",2.5),(r"\boften (neglected|ignored|overlooked)\b",2.5),
        (r"\bunderrepresented\b",2.0),(r"\bdeserves? recognition\b",2.5),
        (r"\btoo often\b",1.5),(r"\blong been ignored\b",2.5),
        (r"\brecent scholarship\b",1.5),(r"\brediscovered\b",2.0),(r"\bforgotten\b",1.5),
        (r"\bmarginalized\b",2.0),(r"\bexcluded\b",1.8),
        # 'despite' omitted — in music theory texts it typically signals harmonic
        # contrast rather than corrective scholarly framing.
        (r"\balthough often\b",1.5),(r"\bwrongly\b",1.5),(r"\bbelatedly\b",2.0),
        (r"\blong overdue\b",2.5),(r"\binclus\w+\b",1.2),(r"\bdiversity\b",1.5),
    ],
}

FRAMING_PROTOTYPES = {
    "normative": [
        "Common practice harmony is the foundation of tonal music theory.",
        "The standard approach to voice leading requires stepwise motion.",
        "This is the conventional way to resolve a dominant seventh chord.",
        "Most textbooks treat this progression as fundamental to tonal syntax.",
        "The typical harmonic rhythm in classical music is one chord per bar.",
    ],
    "additive": [
        "In addition to classical examples, we also find this pattern in jazz.",
        "Another example of this structure appears in popular music as well.",
        "We can also observe similar techniques in the music of Stevie Wonder.",
        "Furthermore, this chord progression is found in many R&B songs.",
        "Similarly, Broadway composers employed this device alongside classical composers.",
    ],
    "exceptional": [
        "This unusual harmonic choice is rare in tonal music.",
        "Unlike most composers of her era, she employed exotic modal scales.",
        "This is an exceptional case that does not follow standard practice.",
        "The piece features irregular rhythms that are atypical for the period.",
        "This represents a unique departure from conventional harmonic syntax.",
    ],
    "corrective": [
        "Historically overlooked by music theory curricula, this tradition deserves recognition.",
        "Despite being marginalized for decades, her work is now receiving scholarly attention.",
        "Recent scholarship has begun to correct the long exclusion of non-Western traditions.",
        "Too often forgotten, this composer made fundamental contributions to the form.",
        "This music was previously ignored by theorists trained in the Western canon.",
    ],
    "neutral": [
        "The chord resolves to the tonic in measure four.",
        "Bach uses a descending sequence in the second phrase.",
        "The melody begins on the third scale degree.",
        "This example illustrates the use of a passing tone.",
        "The following excerpt is from the first movement.",
    ],
}

GENRE_KEYWORD_MAP = {
    "jazz":         [r"\bjazz\b",r"\bbebop\b",r"\bswing\b",r"\bbig band\b"],
    "blues":        [r"\bblues\b",r"\b12.bar blues\b",r"\b12 bar blues\b"],
    "rock":         [r"\brock(?: ?'?n'? ?roll)?\b",r"\brock music\b"],
    "pop":          [r"\bpop music\b",r"\bpopular music\b",r"\bpop song\b"],
    "r_and_b":      [r"\br&b\b",r"\brhythm and blues\b",r"\bsoul music\b",r"\bmotown\b",r"\bfunk\b"],
    "reggae":       [r"\breggae\b",r"\bska\b"],
    "hip_hop":      [r"\bhip.hop\b",r"\brap\b"],
    "latin":        [r"\blatin music\b",r"\bsalsa\b",r"\bregga?eton\b",r"\bclave\b"],
    "country":      [r"\bcountry music\b",r"\bbluegrass\b"],
    "musical_theatre":[r"\bbroadway\b",r"\bmusical theatre\b",r"\bmusical theater\b",r"\bshowtune\b"],
    "classical":    [r"\bclassical music\b",r"\bcommon practice\b",r"\bbaroque\b"],
    "contemporary_classical":[r"\bserialism\b",r"\btwelve.tone\b",r"\bminimalism\b",r"\bpost.tonal\b",r"\bimpressioni[sz]m\b"],
    "film":         [r"\bfilm music\b",r"\bfilm score\b",r"\bsoundtrack\b"],
}

ROLE_WEIGHTS = {"central":1.00,"application":0.60,"supplementary":0.25}

TFIDF_ALPHA = 0.30


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ARGUMENT PARSING                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_args():
    p = argparse.ArgumentParser(
        description="Music Theory Representation & Framing Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",   default="./data",
                   help="Directory containing JSONL textbook files")
    p.add_argument("--output-dir", default="./results",
                   help="Directory for all outputs (CSVs + figures)")
    p.add_argument("--bio-data",   default=None,
                   help="Path to biographical CSV (optional)")
    p.add_argument("--concept-csv", default=None,
                   help="Path to concepts_music_theory.csv (optional)")
    p.add_argument("--excel-data", default=None,
                   help="Path to theory_diversity_full_export.xlsx (optional)")
    p.add_argument("--bert-upgrade-threshold", type=float, default=BERT_UPGRADE_THRESHOLD,
                   help="BERT cosine similarity threshold for framing upgrades")
    p.add_argument("--zero-shot-threshold", type=float, default=ZERO_SHOT_THRESHOLD,
                   help="Zero-shot confidence threshold for framing upgrades")
    p.add_argument("--no-bert",      action="store_true",
                   help="Disable BERT encoder pass entirely")
    p.add_argument("--no-zero-shot", action="store_true",
                   help="Disable LLM framing classifier pass entirely")
    p.add_argument("--tfidf-keywords", action="store_true",
                   help="Enable TF-IDF keyword extraction (fig09/08_tfidf_keywords.csv). "
                        "Disabled by default — on small corpora the log-odds analysis "
                        "surfaces corpus-skew artifacts rather than framing rhetoric.")
    p.add_argument("--debug",    action="store_true",
                   help="Fast run: sample SAMPLE_N passages per book")
    p.add_argument("--sample-n", type=int, default=80,
                   help="Passages per book in debug mode")
    return p.parse_args()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  BIOGRAPHICAL DATA                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _try_read_csv(path: str) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"Could not read {path}")


def _norm_for_match(name: str) -> str:
    import unicodedata
    if not isinstance(name, str): return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_ = nfkd.encode("ascii","ignore").decode("ascii")
    return re.sub(r"[^a-z\s]","", ascii_.lower()).strip()


def _fuzzy_match(query: str, candidates: list) -> str | None:
    q = _norm_for_match(query)
    norms = [_norm_for_match(c) for c in candidates]
    if q in norms: return candidates[norms.index(q)]
    hits = difflib.get_close_matches(q, norms, n=1, cutoff=0.82)
    if hits: return candidates[norms.index(hits[0])]
    return None


def _safe_year(value) -> int | None:
    """
    Convert a birth/death year field to int, handling:
      - Plain integers or numeric strings: "1685" → 1685
      - Circa / approximate notation: "c. 1600", "ca. 1750", "~1720" → first 4-digit run
      - Ranges: "1685-1750" → 1685 (first year)
      - Pandas NA / None / float NaN → None
      - Any other non-parseable string → None
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if not s or s.lower() in ("na", "nan", "none", "unknown", "?"):
        return None
    match = re.search(r'\b(\d{4})\b', s)
    if match:
        return int(match.group(1))
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _era_from_dates(born, died) -> str:
    b = _safe_year(born)
    if b is None:
        return "unknown"
    if b < 1600: return "renaissance"
    if b < 1720: return "baroque"
    if b < 1810: return "classical"
    if b < 1900: return "romantic"
    if b < 1945: return "modern"
    return "contemporary"


def load_biographical_data(path: str | None, resolve_fn) -> dict:
    """
    Load raw_df.csv → composer metadata dict compatible with COMPOSER_METADATA.

    Columns used (case-insensitive):
      composer_clean  canonical name (preferred) or 'composer'
      race            e.g. "White", "Black", "Hispanic", "Unknown"
      sex / gender    "M", "F", "Unknown"
      bipoc           "Y" / "N"  ← direct marginalization flag
      country         birth country
      continent       birth continent
      born / died     birth/death years

    Returns {} if path is None or file not found.
    """
    if path is None: return {}
    p = Path(path)
    if not p.exists():
        print(f"[load_bio] File not found: {p}"); return {}
    bio_raw = _try_read_csv(str(p))
    bio_raw.columns = [c.lower().strip() for c in bio_raw.columns]
    name_col = "composer_clean" if "composer_clean" in bio_raw.columns else "composer"
    bio_raw[name_col] = bio_raw[name_col].astype(str).str.strip()
    bio = bio_raw.drop_duplicates(subset=[name_col]).copy()
    print(f"[load_bio] {len(bio):,} unique composers from {p.name}")
    meta_keys = list(COMPOSER_METADATA.keys())
    result = {}
    fuzzy_aliases: dict[str, str] = {}   # names that fuzzy-matched → register as aliases
    for _, row in bio.iterrows():
        raw = str(row[name_col]).strip()
        if not raw or raw.lower() in ("nan","anon","anonymous","unknown"): continue
        canon = resolve_fn(raw)
        if canon not in COMPOSER_METADATA:
            fm = _fuzzy_match(canon, meta_keys)
            if fm:
                # Fuzzy match found an existing entry.  Register canon→fm as an alias
                # so that extract_entities can look up the JSONL's name directly later.
                # Without this, the bio row silently updates the fm entry but leaves
                # canon as a lookup dead-end (e.g. "Clara Schumann" → absorbed into
                # "Clara Wieck Schumann" but never added → _get_meta("Clara Schumann")
                # returns 'unknown').
                fuzzy_aliases[canon] = fm
                canon = fm
        race_raw   = str(row.get("race","Unknown")).strip()
        gender_raw = str(row.get("sex", row.get("gender","Unknown"))).strip().upper()
        bipoc_raw  = str(row.get("bipoc","N")).strip().upper()
        country    = str(row.get("country","Unknown")).strip()
        continent  = str(row.get("continent","Unknown")).strip()
        born       = row.get("born",  None)
        died       = row.get("died",  None)
        gender_map = {"M":"male","F":"female","MALE":"male","FEMALE":"female",
                      "UNKNOWN":"unknown","NA":"unknown","NAN":"unknown"}
        gender         = gender_map.get(gender_raw, "unknown")
        is_bipoc        = (bipoc_raw == "Y")
        is_female       = (gender == "female")
        is_marginalized = is_bipoc or is_female
        reasons = []
        if is_bipoc:   reasons.append(f"{race_raw} artist")
        if is_female:  reasons.append("woman composer")
        ethnicity_base = race_raw.lower()
        eth_map = {
            "white":    "White European" if continent=="Europe" else "White American",
            "black":    "Black American" if continent in ("North America","Unknown") else f"Black ({country})",
            "hispanic": f"Hispanic ({country})",
            "asian":    f"Asian ({country})",
        }
        ethnicity = eth_map.get(ethnicity_base, race_raw)
        ex = COMPOSER_METADATA.get(canon, {})
        result[canon] = {
            "gender":        gender,
            "ethnicity":     ethnicity,
            "tradition":     ex.get("tradition","classical_canon"),
            "era":           _era_from_dates(born, died),
            "dominant":      ex.get("dominant", False),
            "marginalized":  is_marginalized,
            "margin_reason": "; ".join(reasons) if reasons else None,
            "birth_year":    _safe_year(born),
            "death_year":    _safe_year(died),
            "birth_country": country,
            "continent":     continent,
            "is_bipoc":      is_bipoc,
            "race":          race_raw,
        }
    if fuzzy_aliases:
        print(f"[load_bio] {len(fuzzy_aliases)} fuzzy-matched names registered as aliases:")
        for src, tgt in sorted(fuzzy_aliases.items()):
            COMPOSER_ALIASES[src] = tgt
            print(f"  {src!r:45s} → {tgt!r}")
    return result


def merge_biographical_data(bio_lookup: dict) -> dict:
    n_upd, n_add = 0, 0
    for canon, entry in bio_lookup.items():
        if canon in COMPOSER_METADATA:
            entry["tradition"] = COMPOSER_METADATA[canon].get("tradition", entry["tradition"])
            entry["dominant"]  = COMPOSER_METADATA[canon].get("dominant",  entry["dominant"])
            COMPOSER_METADATA[canon].update(entry)
            n_upd += 1
        else:
            COMPOSER_METADATA[canon] = entry
            n_add += 1
    print(f"[merge_bio] Updated {n_upd} + added {n_add} entries. "
          f"Total: {len(COMPOSER_METADATA)}")
    return COMPOSER_METADATA


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONCEPT CSV                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_concept_patterns(path: str | None) -> dict:
    """
    Load concepts_music_theory.csv and compile each pattern.

    Accepts both tab-separated and comma-separated files; the separator is
    detected automatically. Returns {concept_name: compiled re.Pattern}.
    Patterns compiled with re.IGNORECASE | re.UNICODE.
    Returns {} if path is None.

    Two roles in the pipeline:
      (a) Stopword enrichment — concept-name bigrams are suppressed in TF-IDF
          so they don't appear as false "distinctive" keywords.
      (b) Concept-presence feature layer — binary passage × concept matrix
          built from regex patterns, capturing notation variants (V/V, It6, N6)
          that word tokenisation misses.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[load_concept_patterns] File not found: {p}")
        return {}

    df_c = None
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        for sep in ["\t", ","]:
            try:
                candidate = pd.read_csv(str(p), sep=sep, encoding=enc)
                candidate.columns = [c.lower().strip() for c in candidate.columns]
                if "concept" in candidate.columns and "patterns" in candidate.columns:
                    df_c = candidate
                    print(f"[load_concept_patterns] Read with sep={repr(sep)}, enc={enc}")
                    break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        if df_c is not None:
            break

    if df_c is None:
        try:
            df_c = pd.read_csv(str(p), sep=None, engine="python")
            df_c.columns = [c.lower().strip() for c in df_c.columns]
            print(f"[load_concept_patterns] Read with auto-detected separator")
        except Exception as e:
            print(f"[load_concept_patterns] Could not read CSV: {e}")
            return {}

    if "concept" not in df_c.columns or "patterns" not in df_c.columns:
        print(f"[load_concept_patterns] Expected columns 'concept' and 'patterns'. "
              f"Found: {list(df_c.columns)}")
        return {}

    out, errors = {}, []
    for _, row in df_c.iterrows():
        concept = str(row["concept"]).strip()
        raw_pat = str(row["patterns"]).strip()
        if not concept or concept.lower() == "nan": continue
        if raw_pat.lower() == "nan":                continue
        try:
            out[concept] = re.compile(raw_pat, re.IGNORECASE | re.UNICODE)
        except re.error as e:
            errors.append(f"  '{concept}': {e}")
    if errors:
        print(f"[load_concept_patterns] {len(errors)} pattern compile error(s):")
        for err in errors: print(err)
    print(f"[load_concept_patterns] {len(out)} patterns loaded from {p.name}")
    return out


def _detect_concepts_for_passage(args):
    """Worker function for parallel concept detection."""
    text, concepts, patterns_serialised = args
    results = []
    for j, concept in enumerate(concepts):
        pat = re.compile(patterns_serialised[j], re.IGNORECASE | re.UNICODE)
        results.append(1.0 if pat.search(text) else 0.0)
    return results


def detect_concepts_in_passages(df: pd.DataFrame, concept_patterns: dict,
                                  n_workers: int = 1) -> tuple:
    """
    Scan body_text against every concept regex pattern.

    Returns
    -------
    df : pd.DataFrame
        Input df with two extra columns:
          concepts_detected   — list of concept names matched in each passage
          n_concepts_detected — int count
    concept_df : pd.DataFrame
        Binary float32 matrix (n_passages × n_concepts).
    """
    if not concept_patterns:
        print("[detect_concepts_in_passages] No patterns loaded — skipping.")
        df = df.copy()
        df["concepts_detected"]   = [[] for _ in range(len(df))]
        df["n_concepts_detected"] = 0
        return df, pd.DataFrame(index=df.index)

    texts    = df["body_text"].fillna("").tolist()
    concepts = list(concept_patterns.keys())
    matrix   = np.zeros((len(texts), len(concepts)), dtype=np.float32)

    for j, concept in enumerate(concepts):
        pat = concept_patterns[concept]
        for i, text in enumerate(texts):
            if pat.search(text):
                matrix[i, j] = 1.0

    concept_df = pd.DataFrame(matrix, index=df.index, columns=concepts)

    detected_lists = [
        [concepts[j] for j in range(len(concepts)) if matrix[i, j] > 0]
        for i in range(len(texts))
    ]
    df = df.copy()
    df["concepts_detected"]   = detected_lists
    df["n_concepts_detected"] = [len(d) for d in detected_lists]

    print(f"[detect_concepts_in_passages] {len(concepts)} concepts × {len(texts)} passages")
    print("  Hit counts:")
    for c, n in sorted({c: int(concept_df[c].sum()) for c in concepts}.items(),
                       key=lambda x: -x[1]):
        print(f"    {c:40s}  {int(n):3d}")
    return df, concept_df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EXTERNAL DIVERSITY DATA                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_external_diversity_data(path: str | None, book_map: dict) -> dict:
    """
    Load theory_diversity_full_export.xlsx and return a dict of DataFrames,
    each keyed by sheet name, with a 'book_id' column added.

    Sheets used: diversity_summary, geographic_summary, token_metrics,
                 composer_frequency, edition_deltas, top_bipoc, top_female.

    Returns {} if path is None or file not found.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[load_external_diversity] File not found: {p}"); return {}

    sheets = ["diversity_summary", "geographic_summary", "token_metrics",
              "composer_frequency", "edition_deltas", "top_bipoc", "top_female"]
    out = {}
    try:
        xl = pd.ExcelFile(str(p))
        for sheet in sheets:
            if sheet in xl.sheet_names:
                df_s = xl.parse(sheet)
                label_col = "short_label" if "short_label" in df_s.columns else None
                if label_col:
                    df_s["book_id"] = df_s[label_col].map(book_map)
                out[sheet] = df_s
        print(f"[load_external_diversity] Loaded {len(out)} sheets from {p.name}")
        matched = sum(out["diversity_summary"]["book_id"].notna()) \
                  if "diversity_summary" in out else 0
        print(f"  Books matched to NLP corpus: {matched}/{len(book_map)}")
    except Exception as e:
        print(f"[load_external_diversity] Error reading file: {e}")
        return {}
    return out


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  NAME NORMALISATION                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Junk composer token deny-list ─────────────────────────────────────────────
# Tokens that the NER or vocab extractor sometimes returns as "composer" names
# but are actually tempo/expression marks, movement labels, piece-type names,
# abbreviations, OCR fragments, or song/piece titles.  Any name that resolves
# to one of these is silently dropped by extract_entities.
_JUNK_COMPOSER_TOKENS: frozenset = frozenset({
    # Tempo / expression marks
    "adagio","andante","allegro","allegretto","presto","vivace","largo","lento",
    "moderato","allegro assai","presto ma","adagio molto","andante cantabile",
    "andante con espressione","andante di molto","allegro grazioso",
    "allegro di molto e con brio","moderato cantabile","moderato semplice",
    "assai allegro","poco sostenuto","tempo di gavotta","tempo di menuetto",
    # Movement / section labels
    "minuet","minuetto","minuet trio","menuet","menuetto da capo","menuet i",
    "waltz","waltz op","ländler","ecossaises","polonaise","gigue","allemanda",
    "gavotte","bourrée","sarabande","courante","rondeau","gigue",
    # Genre / form tokens that NER misreads as names
    "harp sonata","violin sonata","keyboard sonata","harpsichord sonata",
    "string quintet","lieder ohne worte","sacri musicali affetti",
    "das wohltemperierte klavier","clavier", "partita",
    # Abbreviations / catalog fragments
    "bwv anh","bwv","lip","roman","anon",
    # OCR / formatting junk
    "fall apart","presto ma non troppo",
    # German song/chorale incipits misread as names
    "herr jesu christ","gott lebet noch","gott vater","gott sohn","gott fähret auf",
    "nun danket alle gott","bisher habt ihr","befiehl du deine wege",
    "dein ist allein","ach gott","ich hab mein sach gott","ach wie flüchtig",
    "ihr gestim","herzens grunde","lebens licht","gloria sei dir gesungen",
    "gute nacht","der neugierige","der müller","kleine studie",
    "wie melodien zieht es mir","fleiß bewahren","don giovanni",
    "alla turca","erlkönig","matthäuspassion","jesu","cavatine de bellini",
    "la straniera","les italiennes","poesia di metatasio",
    "responsório ii","matinas e encomendação de defuntos",
    "danksagung an den bach","herr jesu",
    # Textbook-author surnames misread as composers (Gotham OMT bibliography)
    "slonimsky","lackner","spinner","wolpe","stefan","matthias","hauer matthias",
    # Incipit fragments
    "anna magdalena bach","sacri",
    # Musical form/genre tokens
    "scherzo","barcarolle","sonatina","solfège","solfege","aria","barcarole",
    # Single-word German song incipits / chorale fragments
    "seele","seule","schau","wunden","wachet","weihnachtsoratorium",
    "originaltänze","veranderungen","veränderungen",
    # Multi-word German incipits
    "bunten rosenhecken","vous dirai-je","vom himmel","seinen lauf",
    "seid getrost","sei gepreiset","meine freude","mein leben","lieber gott",
    "ich elender mensch","ibn",
    # OCR fragments and musical annotation noise
    "triads","waltzes","meter","rhythm","supertonic","vii",
    "a. m3","a. facsimile","a. autograph","sol, la","sobbbbb",
    "ma andante","m. 23","bbo ut","bas","diatbo nic","abbb ove","theb o",
    "use##s pitches","↔ m","a.\nb",
    # Movement label fragments
    "memoriam igor stravinsky","i. tempo di gavotta","i. poco sostenuto",
    "i. grave","i. allegretto","josephine lang - traumbild",
    # Textbook-author / editor names
    "apurva ashok","john hajda","zoe wake hyde","allison brown",
    "andre\nmount","andre mount",
})


def normalize_composer_name(raw: str) -> str:
    """
    Strip possessives, piece-attribution bleed, and trailing punctuation.
    Returns empty string for tokens in _JUNK_COMPOSER_TOKENS.
    Called before alias lookup so "Debussy's" → "Debussy" → "Claude Debussy".
    """
    if not isinstance(raw, str): return ""
    name = raw.strip()
    # Strip OCR newline artifacts (e.g. "Herr Jesu\nChrist", "Andre\nMount")
    # and non-breaking / narrow no-break space variants
    name = re.sub(r'[\n\r\t\u00a0\u202f\u2009]+', ' ', name).strip()
    name = re.sub(r'\s{2,}', ' ', name)
    name = re.sub(r"[\u2018\u2019\u02bc']s?\s*$", "", name).strip()
    name = re.sub(r"'s?\s*$", "", name).strip()
    if "," in name:
        pre = name.split(",")[0].strip()
        if len(pre) >= 4: name = pre
    name = name.rstrip(".,;:")
    name = name.strip()
    # Drop junk tokens — these are tempo marks, incipits, abbreviations etc.
    if name.lower() in _JUNK_COMPOSER_TOKENS:
        return ""
    # Drop single-character tokens, tokens with OCR noise patterns, and
    # tokens that are clearly not names (contain digits, musical symbols,
    # multiple consecutive punctuation, or are all-lowercase single words)
    if len(name) < 3:
        return ""
    if re.search(r'[♯♭♮#@$%^&*+=<>|~`]', name):
        return ""
    if re.search(r'\b[a-z]{2,}\b##|use##|[a-z]{2,}bb[a-z]', name):
        return ""   # OCR ligature artifacts
    if re.match(r'^[a-z]', name) and len(name.split()) == 1:
        return ""   # single lowercase word — not a name
    return name


def resolve_composer(raw: str) -> str:
    """Normalise → check aliases → return canonical name."""
    cleaned = normalize_composer_name(raw)
    return COMPOSER_ALIASES.get(cleaned, cleaned)


def _get_meta(name: str) -> dict:
    """Return metadata for a composer; defaults for unknowns."""
    default = {"gender":"unknown","ethnicity":"unknown","tradition":"unknown",
               "era":"unknown","dominant":False,"marginalized":False,
               "margin_reason":"not in knowledge base"}
    return COMPOSER_METADATA.get(resolve_composer(name), default)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LOAD DATA                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _load_single_book(args):
    """Worker for parallel book loading."""
    book_id, path, debug, sample_n = args
    p = Path(path)
    if not p.exists():
        return book_id, []
    records = []
    count = 0
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            if debug and count >= sample_n: break
            rec = json.loads(line)
            rec["book_id"] = book_id
            records.append(rec)
            count += 1
    return book_id, records, count


def load_data(data_files: dict, data_dir: str, debug: bool = False,
              sample_n: int = 80) -> pd.DataFrame:
    """
    Load multiple JSONL textbook files into one DataFrame.

    Parameters
    ----------
    data_files : dict  {book_id: filename}
    data_dir   : str   Base directory prepended to each filename
    debug      : bool  Load only the first sample_n passages per file
    sample_n   : int   Passages per file in debug mode
    """
    base = Path(data_dir)
    args_list = [(bid, str(base / fname), debug, sample_n)
                 for bid, fname in data_files.items()]

    all_records = []
    for book_id, fname, debug_, sample_n_ in args_list:
        book_id, records, count = _load_single_book((book_id, fname, debug_, sample_n_))
        tag = f"[DEBUG {count}]" if debug_ else f"[{count} passages]"
        p = Path(fname)
        if records:
            print(f"  ✓ {book_id:30s} {tag}  ← {p.name}")
        else:
            print(f"  ⚠  {book_id:30s} not found, skipping: {p}")
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    df.insert(0, "passage_id", range(len(df)))
    for col in ["chapter_number","section_number"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in ["word_count","sentence_count","notation_ratio"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    list_cols = ["composers_all_sources","composers_ner","composers_vocab",
                 "concepts_passage","concepts_chapter","piece_attributions",
                 "audio_refs","example_numbers"]
    for col in list_cols:
        if col not in df.columns: df[col] = [[] for _ in range(len(df))]
        df[col] = df[col].apply(lambda x: x if isinstance(x,list) else [])
    role_map = {"body":"central","sidebar":"supplementary","exercise":"application"}
    if "passage_role" not in df.columns: df["passage_role"] = "central"
    df["passage_role"] = df["passage_role"].fillna("central")
    if "passage_type" in df.columns:
        mask = df["passage_role"].str.strip() == ""
        df.loc[mask,"passage_role"] = df.loc[mask,"passage_type"].map(role_map).fillna("central")
    df["chapter_position"] = 0.5
    for bid, grp in df.groupby("book_id"):
        mx = grp["chapter_number"].max()
        if pd.notna(mx) and mx > 0:
            df.loc[grp.index,"chapter_position"] = (grp["chapter_number"]/mx).round(3)
    print(f"\n[load_data] {'DEBUG' if debug else 'FULL'} — "
          f"{len(df):,} passages | {df['book_id'].nunique()} book(s)")
    print(f"  passage_role: {df['passage_role'].value_counts().to_dict()}")
    return df.reset_index(drop=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PREPROCESS TEXT + GENRE DETECTION                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Genre signals come from two independent sources:
#   1. METADATA genre (COMPOSER_METADATA['tradition']) — reflects the
#      composer's genre, not whether the passage discusses that genre.
#   2. TEXT genre (genre_mentions column) — keyword scan of body_text.
# The combination reveals four passage types:
#   a. Named AND composer cited: genre discussed AND exemplified
#   b. Named ONLY: genre mentioned but no named composer
#   c. Composer ONLY: composer cited but genre name never appears
#   d. Neither: no genre signal at all

def _clean_body(text: str) -> str:
    """Lightweight normalisation for framing classifier and sentence tokeniser."""
    if not isinstance(text, str): return ""
    text = re.sub(r'(Figure|Example|Subsection|Definition|Principle)\s+[\d.]+', ' ', text)
    text = re.sub(r'[♯♭♮𝄞𝄢𝄫𝄪]', '', text)
    text = re.sub(r'\b[IiVvbBO#°ø\+\-\d]{1,6}\b', '', text)
    # PDF OCR character-doubling artefact (benward_2008 scanned PDF): every character
    # is printed twice — "SSyynnccooppaattiioonn", "PPMM", "TTOO TTHHEE".
    def _collapse_doubled_body(m: re.Match) -> str:
        s = m.group(0)
        if len(s) < 4 or len(s) % 2 != 0:
            return s
        if all(s[i] == s[i+1] for i in range(0, len(s)-1, 2)):
            return s[::2]
        return s
    text = re.sub(r'\b[A-Za-z]{4,}\b', _collapse_doubled_body, text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _tokenize_sentences(text: str) -> list:
    """Regex sentence tokeniser (no external NLP library required)."""
    abbrev = (r"\b(Mr|Mrs|Ms|Dr|Prof|Op|No|Vol|Fig|Ex|mm|ca|bars?|"
              r"vs|e\.g|i\.e|cf|min|sec|Hz|kHz|BWV|K|p|pp|mf|mp|ff|fff)\.")
    text = re.sub(abbrev, lambda m: m.group(0).replace(".", "_DOT_"),
                  text, flags=re.IGNORECASE)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\"\(\[])', text)
    return [s.replace("_DOT_", ".").strip()
            for s in sentences if len(s.split()) > 2]


def detect_genre_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """Add genre_mentions (list) and n_genre_mentions (int) columns to passages."""
    df = df.copy()
    concept_genre = {
        "jazz harmony":"jazz", "lead-sheet notation":"jazz",
        "twelve-tone":"contemporary_classical", "serialism":"contemporary_classical",
        "set theory":"contemporary_classical", "post-tonal":"contemporary_classical",
        "church modes":"classical", "species counterpoint":"classical",
    }
    genre_lists = []
    for _, row in df.iterrows():
        text     = (row.get("body_text") or "").lower()
        concepts = row.get("concepts_passage", [])
        found    = set()
        for genre, pats in GENRE_KEYWORD_MAP.items():
            if any(re.search(p, text, re.IGNORECASE) for p in pats):
                found.add(genre)
        for c in concepts:
            g = concept_genre.get(c.lower())
            if g: found.add(g)
        genre_lists.append(sorted(found))
    df["genre_mentions"]   = genre_lists
    df["n_genre_mentions"] = df["genre_mentions"].apply(len)
    total_hits = Counter(g for gl in genre_lists for g in gl)
    print(f"[detect_genre_mentions] {(df['n_genre_mentions']>0).sum()} passages "
          f"with explicit genre mention")
    print(f"  Genre counts: {dict(total_hits.most_common(8))}")
    return df


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_text"]      = df["body_text"].apply(_clean_body)
    df["sentences"]       = df["body_text"].apply(_tokenize_sentences)
    df["clean_sentences"] = df["clean_text"].apply(_tokenize_sentences)
    df["n_sentences_tok"] = df["sentences"].apply(len)
    print(f"[preprocess_text] avg {df['n_sentences_tok'].mean():.1f} sentences/passage")
    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  EXTRACT ENTITIES                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _ctx_window(sentences: list, idx: int, window: int=2) -> str:
    start = max(0, idx - window)
    end   = min(len(sentences), idx + window + 1)
    return " ".join(sentences[start:end])


def extract_entities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build entity-level DataFrame: one row per (passage × composer mention sentence).

    Per-passage deduplication: composers_all_sources may contain both
    "Claude Debussy" and "Debussy" (NER + vocab union). OrderedDict
    collapses these to one canonical entry before creating rows.
    Final dedup on (passage_id, composer_canonical, sentence_idx) removes
    any remaining duplicates from multiple sentence matches.
    """
    rows = []
    for _, passage in df.iterrows():
        raw_composers = passage.get("composers_all_sources", [])
        if not raw_composers: continue
        sentences = passage["sentences"]

        seen: OrderedDict[str, str] = OrderedDict()
        for raw in raw_composers:
            canon = resolve_composer(raw)
            if canon and canon not in seen:
                seen[canon] = raw

        for canonical, composer_raw in seen.items():
            meta      = _get_meta(canonical)
            name_parts= [p for p in canonical.split() if len(p) >= 4] or canonical.split()
            found = False
            for s_idx, sent in enumerate(sentences):
                sl = sent.lower()
                if any(part.lower() in sl for part in name_parts):
                    rows.append({"passage_id":passage["passage_id"],
                                 "book_id":passage["book_id"],
                                 "book_title":passage.get("book_title",""),
                                 "chapter_number":passage["chapter_number"],
                                 "chapter_title":passage.get("chapter_title",""),
                                 "section_title":passage.get("section_title",""),
                                 "passage_role":passage["passage_role"],
                                 "passage_type":passage.get("passage_type",""),
                                 "chapter_position":passage["chapter_position"],
                                 "word_count":passage["word_count"],
                                 "composer_raw":composer_raw,
                                 "composer_canonical":canonical,
                                 **meta,
                                 "mention_sentence":sent,
                                 "context_window":_ctx_window(sentences, s_idx),
                                 "sentence_idx":s_idx})
                    found = True
            if not found:
                rows.append({"passage_id":passage["passage_id"],
                             "book_id":passage["book_id"],
                             "book_title":passage.get("book_title",""),
                             "chapter_number":passage["chapter_number"],
                             "chapter_title":passage.get("chapter_title",""),
                             "section_title":passage.get("section_title",""),
                             "passage_role":passage["passage_role"],
                             "passage_type":passage.get("passage_type",""),
                             "chapter_position":passage["chapter_position"],
                             "word_count":passage["word_count"],
                             "composer_raw":composer_raw,
                             "composer_canonical":canonical,
                             **meta,
                             "mention_sentence":"",
                             "context_window":(passage["body_text"] or "")[:500],
                             "sentence_idx":-1})
    edf = pd.DataFrame(rows)
    edf = edf.drop_duplicates(subset=["passage_id","composer_canonical","sentence_idx"])
    print(f"[extract_entities] {len(edf):,} rows | "
          f"{edf['composer_canonical'].nunique()} unique composers | "
          f"{edf['book_id'].nunique()} book(s)")
    return edf


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TF-IDF + RULE FRAMING CLASSIFIER  (backbone)                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _build_tfidf_stopwords(concept_patterns: dict | None = None) -> set:
    """
    Build comprehensive stopword list.

    When concept_patterns is provided, all concept-name unigrams and bigrams
    are added to the suppression set. This prevents structural music theory
    labels (e.g. 'binary form', 'modal mixture') from appearing as false
    distinctive keywords in the TF-IDF contrast analysis.
    """
    music_abbrev = {"op","opus","bwv","kwv","woo","hob","rv","sz","bb","k","kv",
                    "no","vol","fig","ex","mm","ca","bars","bk","mov","mvt",
                    "ii","iii","iv","vi","vii","viii"}
    foreign_sw   = {"für","von","der","die","das","und","ist","auf","mit","aus",
                    "bei","nach","über","unter","zum","zur","des","den","dem",
                    "ein","eine","einer","eines","einem","de","du","la","les",
                    "le","et","en","au","aux","un","une","dans","sur","par",
                    "est","qui","di","il","al","da","del","dei","col","con"}
    composer_tokens = set()
    for name in COMPOSER_METADATA:
        for part in name.lower().split():
            p_ = re.sub(r"[^a-z]","",part)
            if len(p_) >= 3: composer_tokens.add(p_)
    for alias in COMPOSER_ALIASES:
        for part in alias.lower().split():
            p_ = re.sub(r"[^a-z]","",part)
            if len(p_) >= 3: composer_tokens.add(p_)
    universal = {
        "chord","chords","note","notes","music","musical","measure","measures",
        "beat","beats","bar","bars","voice","voices","pitch","pitches",
        "interval","intervals","scale","scales","key","tone","tones",
        "melody","melodies","rhythm","rhythms","tempo","meter","time",
        "piece","song","section","phrase","phrases","cadence","cadences","pattern",
        "example","examples","excerpt","excerpts","figure","figures","following","above","below",
        "first","second","third","fourth","next","last","similar","different",
        "common","use","used","using","often","also","well","called","known",
        "found","written","played","listen","see","shown","appear","appears",
        "want","love","like","feel","got","let","need","come","know","just",
        "make","take","way","dont","doesnt","didnt","isnt","involve","wont",
        "song","songs","line","lines","passage","passages","hear","heard",
        "sound","sounds",
        "unit","units","assignment","assignments","exercise","exercises",
        "chapter","chapters","page","pages","practice","review","worksheet",
        "occur","occurs","occurred","usually","frequently","illustrates","illustrate",
        "moves","notice","identify","starts","begins","begin","shows","show",
        "asks","ask","performed","perform","playlist","university",
        "composers","upper","text","range","score","collection","collections","elements",
        "structural","often","double","track","performed",
        "iac","pac","hac","dc",
        "pdf","latex","caption","schema","schemas","shortcode","href","img","alt",
        "leadingtone","ppmm","aamm","aapp","ttoo","tthhee",
        "students","theory","row","rows",
        "mscz","docx","xlsx","html","svg","midi","xml",
        "http","www",
        "quartet","identify","notice","starts","range","collection","score",
        # Web navigation / online textbook UI artifacts (Gotham OMT, Hutchinson web sources).
        "view","order","content","features",
        # Digital/online textbook reader artifacts (Hutchinson browser-based ereader)
        "audio","online","ereader","browserbased",
        # InDesign export artifact
        "indd",
        # Generic music terms too broadly distributed to be discriminative
        "function","ascending","descending",
        # High-frequency genre/form labels that appear almost exclusively in classical passages
        "symphony","symphonic","allegro","soprano","prelude","fugue",
        # Generic analytical/academic terms that appear uniformly across both groups
        "complete","degree","moving","effect","series","listener","relationship","goal",
        # Pedagogical structural terms
        "sentence","embellishing",
        # ── Generic prose words appearing as false TF-IDF signals ──
        # Generic academic prose:
        "possibility","possibilities","especially","normally","stylistic","frequent",
        "composition","compositions","theme","themes","number","numbers",
        "important","importance","write","writing","wrote",
        "macro",                        # "macro analysis" (Benward analytical term)
        "ing",                          # OCR hyphenation fragment: "challeng-\ning"
        # General analytical vocabulary:
        "predominant","predominantly","sense","context","contexts","model","models",
        "conclusive","conclusively","original","originally","reprise",
        "downward","upward","diagram","diagrams",
        # Additional generic textbook prose:
        "particular","particularly","specific","specifically","general","generally",
        "certain","various","several","approach","approaches","however","although",
        "therefore","result","results","process","type","types","form","forms",
        "point","points","term","terms","level","levels","kind","kinds",
        "example","place","places","idea","ideas","aspect","aspects",
        "manner","means","way","ways","fact","situation","situations",
        # ── Additional generic prose words ──
        "words","word","century","centuries","position","positions",
        "study","studies","start","starts","starting","actually","really",
        "incomplete","beginning","endings","begins","ended","began",
        "later","earlier","instead","within","without","toward","towards",
        "perhaps","though","thus","since","whether","along","across",
        "table","tables","mark","marks","home","homes","distinguish",
        "notated","notation","written","showed","showing",
        "contains","contain","support","supports","technique","techniques",
        "movement","movements","main","french","romantic","dissonant",
        # Author / editor names that bleed from citations
        "gotham","levine","stefan","david","matthew","andrew","mark",
        "lawrence","linda","coleridge","appendix",
        "copyright","reserved","rights","permission",
        # Browser/reader artifacts
        "browser","browserbased","based","online","ereader",
        "click","download","link","links","accessed","access",
        # Instrument names — uniformly distributed, not framing signals
        "violin","viola","cello","flute","oboe","clarinet","bassoon",
        "trumpet","trombone","piano","guitar","organ","harpsichord",
        # Italian tempo / expression marks — appear in any tradition, no framing signal
        "molto","allegro","andante","moderato","presto","adagio",
        "vivace","largo","lento","poco","assai","sostenuto","cantabile",
        # Piece-title tokens that escape catalog stripping
        "dichterliebe","clavier","tempered","sonatina","etude",
        # Genre / form labels that are broadly distributed enough to suppress
        "waltzes","waltz","trio","trios","overture","overtures",
        "mazurka","sonatina","minuets","fugues","variations",
        # Analytical terms too generic (context-window switch increased their IDF)
        "pivot","tonic","tonicized","tonicizing","tonicization",
        "harmonies","auxiliary","resolve","downbeat","sequential",
        "recording","translation","recomposition","embedded","melodically",
        "moderato",
        # ── Terms surfaced by log-odds from Gotham bibliography bleed ──────────
        # These are proper names / surnames from Gotham OMT reference sections
        # that appear in context windows for nearby composers.  The _clean_for_tfidf
        # bibliography-stripping regex removes most, but residual fragments still
        # land in the vocab.  Belt-and-suspenders suppression here.
        "slonimsky","lackner","spinner","wolpe","stefan","matthias","hauer",
        "nicholas","permutation","semitone","progression",
        # Terms from Clendinning anthology preface boilerplate
        "anthology","integral","strophic","emerge","emerges","emerged",
        "integral","believe","chosen","strongly",
        # Remaining analytical noise from 08_tfidf_keywords.csv inspection
        "concerto","invention","partita","division","neighbor","element",
        "nineteenth","traditional","exchange","variation","minimalism",
        "dissonance","rhythmic","baroque","verse","play","local","metrically",
        "falling","compound","represented","high","modally","accented",
        "pulses","mixed","arrives","grouped","retrograde","thirds",
        "anthology","hamilton","retrograde","compound",
        "neighbor","neighboring","dissonant",
    }
    concept_suppression = set()
    if concept_patterns:
        for concept_name in concept_patterns.keys():
            words = re.findall(r"[a-z]+", concept_name.lower())
            for w in words:
                if len(w) >= 3: concept_suppression.add(w)
            for i in range(len(words) - 1):
                a, b = words[i], words[i+1]
                if len(a) >= 3 and len(b) >= 3:
                    concept_suppression.add(f"{a} {b}")
        new_bigrams = sorted(w for w in concept_suppression if " " in w)
        if new_bigrams:
            print(f"[stopwords] +{len(new_bigrams)} concept bigrams suppressed: {new_bigrams}")
    return (set(ENGLISH_STOP_WORDS) | music_abbrev | foreign_sw
            | composer_tokens | universal | concept_suppression)


def _clean_for_tfidf(text: str) -> str:
    """
    Strip figure refs, catalog numbers, cited names, boilerplate, and markup artefacts.

    Key additions over previous version:
    - Gotham OMT bibliography format: "Lastname, Firstname: Title. Date;" bullet lists
    - Clendinning anthology preface boilerplate
    - German-language date patterns (28. 6. 2016)
    - Bullet-point citation blocks (• entry; entry; entry)
    - Row/pitch-class set notation artifacts
    """
    if not isinstance(text, str): return ""

    # ── Copyright / rights lines ──────────────────────────────────────────────
    text = re.sub(r'Copyright\s+\d{4}[^\n]*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'All\s+rights\s+reserved[^\n]*', ' ', text, flags=re.IGNORECASE)

    # ── Gotham OMT bibliography bleed ─────────────────────────────────────────
    # Format: "Lastname, Firstname: Title. 28. 6. 2016;" (semicolon-separated list)
    # Also catches "Slonimsky, Nicholas: No. 648-654 Permutations…"
    # Strategy: strip any token sequence matching Surname, Name: ... up to next ; or \n
    text = re.sub(r'[A-Z][a-zA-Z\-]+,\s+[A-Z][a-zA-Z\s\-]+:\s+[^\n;]{0,120}[;\n]?',
                  ' ', text)
    # German-language date patterns bleeding from OMT (28. 6. 2016)
    text = re.sub(r'\b\d{1,2}\.\s+\d{1,2}\.\s+\d{4}\b', ' ', text)
    # Bullet-point list items (• ...) — common in Gotham reference sections
    text = re.sub(r'•\s*[^\n•]{0,200}', ' ', text)
    # Pitch-class set / tone-row notation artifacts: [0, 11, 6, 5, 4, …]
    text = re.sub(r'\[\s*\d+(?:,\s*\d+){2,}\s*\]', ' ', text)
    # "für drei Tasteninstrumente" etc. — German title fragments
    text = re.sub(r'\bfür\s+\w+\s+\w+\b', ' ', text, flags=re.IGNORECASE)

    # ── Clendinning anthology boilerplate ─────────────────────────────────────
    # Phrases from the standard Clendinning preface that bleed into marginalized
    # composer contexts and produce false framing signals.
    boilerplate_phrases = [
        r'Study of the Anthology works is integral to[^.]{0,120}\.',
        r'we strongly believe that the concepts[^.]{0,120}\.',
        r'integral to the book',
        r'emerge from the music itself',
        r'Anthology works',
    ]
    for bp in boilerplate_phrases:
        text = re.sub(bp, ' ', text, flags=re.IGNORECASE)

    # ── Browser/ereader navigation artifacts ──────────────────────────────────
    text = re.sub(r'\b(?:Browse|View|Order|Download|Click|Access)\s+(?:content|order|here|now)\b',
                  ' ', text, flags=re.IGNORECASE)

    # ── Publisher watermarks ──────────────────────────────────────────────────
    text = re.sub(r'Benw[^\n]{0,80}Sak[^\n]*', ' ', text, flags=re.IGNORECASE)

    # ── Inline citation patterns ──────────────────────────────────────────────
    # Em-dash author-year: "— Levine, 2004"
    text = re.sub(r'—\s*[A-Z][a-z]+,?\s*\d{4}', ' ', text)
    # Parenthetical year-only: (2016), (c. 1750)
    text = re.sub(r'\((?:c\.?\s*)?\d{4}(?:[–\-]\d{2,4})?\)', ' ', text)
    text = re.sub(r'\b(?:et al|ibid|op cit)\b\.?', ' ', text, flags=re.IGNORECASE)

    # ── Figure / example references ───────────────────────────────────────────
    text = re.sub(r'\b(?:Figure|Example|Subsection|Definition|Principle)\s+[\d.]+',
                  ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[Ff]ig\.?\s*[\d.]+', ' ', text)
    text = re.sub(r'\b(?:Op|BWV|K|KV|WoO|Hob|RV|Sz|BB|No|Vol)\b\.?\s*[\dIVXivx,.\s]{1,15}',
                  ' ', text, flags=re.IGNORECASE)

    # ── Quoted titles ─────────────────────────────────────────────────────────
    text = re.sub(r'"[^"]{2,80}"', ' ', text)
    text = re.sub(r'\u201c[^\u201d]{2,80}\u201d', ' ', text)
    text = re.sub(r'[\u201c\u201d\u2018\u2019\u201e\u201f\u2039\u203a]', ' ', text)

    # ── Contractions ──────────────────────────────────────────────────────────
    text = re.sub(r"n't\b", ' not', text)
    text = re.sub(r"'s\b",  ' ',    text)
    text = re.sub(r"'re\b", ' are', text)
    text = re.sub(r"'ve\b", ' have',text)

    # ── Music notation symbols and numerals ───────────────────────────────────
    text = re.sub(r'[♯♭♮𝄞𝄢𝄫𝄪]', '', text)
    text = re.sub(r'\b\d+(?:st|nd|rd|th)?\b', '', text)
    text = re.sub(r'\b[IiVvXxLlCcDdMm]{1,7}\b', '', text)

    # ── LaTeX / markdown / shortcode artifacts ────────────────────────────────
    text = re.sub(r'\[/?caption[^\]]*\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[/?[a-z_]+[^\]]{0,40}\]', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]{0,60}\}', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)

    # ── File extensions / URLs ────────────────────────────────────────────────
    text = re.sub(r'\b\w+\.(?:indd|mscz|docx?|xlsx?|pdf|mp[34]|xml|midi?|svg|png|jpe?g|html?)\b',
                  ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'\S+/\S+\.\w{2,5}\b', ' ', text)

    # ── PDF OCR character-doubling artefact (benward_2008 scanned PDF) ────────
    def _collapse_doubled(m: re.Match) -> str:
        s = m.group(0)
        if len(s) < 4 or len(s) % 2 != 0: return s
        if all(s[i] == s[i+1] for i in range(0, len(s)-1, 2)): return s[::2]
        return s
    text = re.sub(r'\b[A-Za-z]{4,}\b', _collapse_doubled, text)

    text = re.sub(r'\b\d{1,2}[:/]\d{2}[:/]\d{2,4}\b', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_tfidf_model(entity_df: pd.DataFrame, concept_patterns: dict) -> tuple:
    """Fit TF-IDF on all context windows + prototypes. Returns (vec, matrix, centroids)."""
    all_texts   = entity_df["context_window"].fillna("").apply(_clean_for_tfidf).tolist()
    proto_texts = [s for sents in FRAMING_PROTOTYPES.values() for s in sents]
    vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATS,
                          stop_words=list(_build_tfidf_stopwords(concept_patterns)),
                          ngram_range=(1,2), min_df=1, sublinear_tf=True)
    full = vec.fit_transform(all_texts + proto_texts)
    pm   = full[:len(all_texts)]
    centroids = {}
    offset = len(all_texts)
    for cat, sents in FRAMING_PROTOTYPES.items():
        n = len(sents)
        block = full[offset:offset+n].toarray().mean(axis=0)
        norm  = np.linalg.norm(block)
        centroids[cat] = block / norm if norm > 0 else block
        offset += n
    print(f"[build_tfidf_model] vocab={len(vec.vocabulary_)} | matrix={pm.shape}")
    return vec, pm, centroids


def _lex_scores(text: str) -> dict:
    t = text.lower()
    return {cat: sum(len(re.findall(p,t))*w for p,w in pats)
            for cat, pats in FRAMING_LEXICONS.items()}


def classify_framing_backbone(entity_df: pd.DataFrame, pm, centroids: dict,
                               alpha: float = TFIDF_ALPHA) -> pd.DataFrame:
    """Rule + TF-IDF combined framing classifier."""
    edf  = entity_df.copy()
    cats = list(FRAMING_LEXICONS.keys())
    lex  = edf["context_window"].fillna("").apply(_lex_scores)
    for cat in cats:
        edf[f"framing_{cat}"] = lex.apply(lambda d: d[cat])
    pm_norm    = normalize(pm, norm="l2")
    sim_matrix = np.zeros((len(edf), len(cats)))
    for j, cat in enumerate(cats):
        sims = pm_norm.dot(centroids[cat].T).flatten()
        sim_matrix[:,j] = np.clip(sims, 0, 1)
        edf[f"tfidf_sim_{cat}"] = sim_matrix[:,j].round(4)
    labels, confs = [], []
    for i in range(len(edf)):
        combined = {cat: lex.iloc[i][cat] + alpha*sim_matrix[i,j]
                    for j,cat in enumerate(cats)}
        for j,cat in enumerate(cats):
            edf.at[i,f"combined_{cat}"] = round(combined[cat],4)
        total  = sum(combined.values()) + 1e-9
        best   = max(combined, key=combined.get)
        bscore = combined[best]
        if bscore < 0.15:
            labels.append("neutral"); confs.append(0.0)
        else:
            labels.append(best); confs.append(round(bscore/total,3))
    edf["framing_category"]   = labels
    edf["framing_confidence"] = confs
    edf["framing_source"]     = "rules"
    print(f"[classify_framing] {Counter(labels)}")
    return edf


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  BERT SUPPORT LAYER  (optional — BAAI/bge-large-en-v1.5)                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _resolve_device(req: str) -> str:
    if req != "auto":
        return req
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
    except ImportError:
        pass
    return "cpu"


def run_bert_pass(entity_df: pd.DataFrame, bert_model_name: str,
                  device: str, threshold: float, batch_size: int,
                  n_clusters: int) -> pd.DataFrame:
    """
    Encode context windows with a sentence-transformer model and upgrade
    neutral framing labels by cosine similarity to prototype centroids.

    BAAI/bge-large-en-v1.5 is recommended: it achieves SOTA performance on
    semantic similarity tasks (MTEB leaderboard) and fits within 3 GB of VRAM
    at batch_size=128, leaving headroom for the zero-shot model.

    BGE models benefit from an instruction prefix on query texts:
    "Represent this sentence for searching relevant passages: <text>"
    We apply this to context windows but not to prototypes, matching the
    retrieval-style use case.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print(f"⚠  sentence-transformers could not be imported — BERT pass skipped.")
        print(f"   Error: {e}")
        print(f"   Fix: pip install --force-reinstall 'transformers==4.46.3'")
        return entity_df

    print(f"[BERT] Loading '{bert_model_name}' on {device} …")
    model = SentenceTransformer(bert_model_name, device=device)

    # BGE models use an instruction prefix for query/retrieval tasks
    is_bge = "bge" in bert_model_name.lower()
    prefix = "Represent this sentence for framing classification: " if is_bge else ""

    context_texts = [
        prefix + t for t in entity_df["context_window"].fillna("").tolist()
    ]
    bert_embeddings = model.encode(
        context_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"  Embeddings: {bert_embeddings.shape}")

    proto_embeddings = {}
    for cat, sents in FRAMING_PROTOTYPES.items():
        vecs = model.encode(sents, normalize_embeddings=True, convert_to_numpy=True)
        cent = vecs.mean(axis=0)
        norm = np.linalg.norm(cent)
        proto_embeddings[cat] = cent / norm if norm > 0 else cent

    edf = entity_df.copy()
    cats_nonneutral = [c for c in FRAMING_PROTOTYPES if c != "neutral"]
    sim_mat = np.stack(
        [bert_embeddings.dot(proto_embeddings[c]) for c in cats_nonneutral], axis=1
    )
    for j, cat in enumerate(cats_nonneutral):
        edf[f"bert_sim_{cat}"] = sim_mat[:, j].round(4)

    n_upgraded = 0
    for i, row in edf.iterrows():
        if row["framing_category"] != "neutral":
            continue
        loc_i = edf.index.get_loc(i)
        sims  = {c: sim_mat[loc_i, j] for j, c in enumerate(cats_nonneutral)}
        best  = max(sims, key=sims.get)
        bsim  = sims[best]
        edf.at[i, "bert_best_cat"] = best
        edf.at[i, "bert_best_sim"] = round(float(bsim), 4)
        if bsim >= threshold:
            edf.at[i, "framing_category"]  = best
            edf.at[i, "framing_source"]    = "bert_upgrade"
            edf.at[i, "framing_confidence"]= round(float(bsim), 3)
            n_upgraded += 1
    print(f"  BERT upgrades: {n_upgraded} neutral rows promoted "
          f"(threshold={threshold})")

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    edf["bert_cluster"] = km.fit_predict(bert_embeddings)
    print(f"  Clusters: {Counter(edf['bert_cluster'].tolist())}")

    # Explicitly release BGE-large from VRAM so the LLM classifier
    # has the full GPU budget.  Without this, BGE occupies ~10+ GB
    # (model weights + embedding matrix) and leaves insufficient
    # headroom for Llama-3.1-8B-Instruct (~16 GB fp16).
    del model
    del bert_embeddings
    try:
        import torch as _torch
        _torch.cuda.empty_cache()
        print("  [BERT] Model unloaded from VRAM.")
    except Exception:
        pass

    return edf


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LLM FRAMING CLASSIFIER  (Llama-3.1-8B-Instruct)                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_LLM_SYSTEM_PROMPT = """You are a musicology research assistant classifying how composers are rhetorically introduced in music theory textbook passages.

Classify each passage into exactly one of these four framing categories:
  normative   — the composer/tradition is presented as standard, fundamental, typical, or the default example (e.g. "the standard approach", "common practice", "Bach exemplifies")
  additive    — the composer/tradition is introduced as an additional or supplementary example alongside an already-established one (e.g. "also appears in", "another example is", "in addition to")
  exceptional — the composer/tradition is marked as unusual, rare, exotic, or an exception to the rule (e.g. "unlike most composers", "this unusual feature", "atypical")
  corrective  — the passage explicitly acknowledges historical exclusion or calls for recognition (e.g. "historically overlooked", "deserves recognition", "often neglected")

Reply with ONLY the category word. No explanation."""

_LLM_USER_TEMPLATE = "Passage: {context}\n\nCategory:"


def _build_llm_prompt(context: str) -> list[dict]:
    """Build chat messages for Llama-3-Instruct format."""
    return [
        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
        {"role": "user",   "content": _LLM_USER_TEMPLATE.format(context=context[:600])},
    ]


def run_zero_shot_pass(entity_df: pd.DataFrame, model_name: str,
                        device: str, threshold: float,
                        batch_size: int) -> pd.DataFrame:
    """
    LLM-based framing classifier using Llama-3-8B-Instruct.

    An instruction-tuned LLM applies reasoning about what "normative",
    "additive", "exceptional", and "corrective" mean in music theory pedagogy,
    rather than testing labels against entailment hypotheses.

    Hardware notes (assuming ~24 GB VRAM):
      fp16 full precision (default) → ~16 GB VRAM; BGE-large adds ~1.3 GB = ~17.3 GB total
      4-bit NF4 quantisation        → ~5 GB VRAM (set LLM_CLASSIFIER_LOAD_4BIT = True)
      fp16 is preferred — no VRAM pressure, no bitsandbytes dependency, better accuracy.

    The vLLM inference engine is used when available because it parallelises
    generation across CPU threads.  Falls back to direct HuggingFace
    transformers if vLLM is not installed.

    Targets rows that are neutral after the BERT pass, or have low BERT
    confidence (< threshold × 0.8).  Rows already labelled by rules or BERT
    are not re-classified.
    """
    valid_labels = {"normative", "additive", "exceptional", "corrective"}

    # ── Select rows to classify ───────────────────────────────────────────────
    bert_sim_col = "bert_best_sim" if "bert_best_sim" in entity_df.columns else None
    if bert_sim_col is not None:
        zs_mask = ((entity_df["framing_category"] == "neutral") |
                   (entity_df[bert_sim_col].fillna(0.0) < threshold * 0.8))
    else:
        zs_mask = entity_df["framing_category"] == "neutral"

    zs_indices = entity_df.index[zs_mask].tolist()
    zs_texts   = entity_df.loc[zs_mask, "context_window"].fillna("").tolist()

    if not zs_texts:
        print("[LLM] No rows to classify — skipping.")
        return entity_df

    print(f"[LLM] Classifying {len(zs_texts)} rows with '{model_name}' …")

    # ── Try vLLM first (faster on multi-core server) ──────────────────────────
    labels_out: list[str] = []
    used_vllm = False
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model_name,
            dtype="bfloat16",
            quantization="bitsandbytes" if LLM_CLASSIFIER_LOAD_4BIT else None,
            max_model_len=1024,
            # 0.72 ≈ 17 GB on a 23.5 GB card.  BGE-large is unloaded before
            # this point so the full GPU should be available; the margin
            # handles driver overhead and any other resident processes.
            gpu_memory_utilization=0.72,
        )
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=LLM_MAX_NEW_TOKENS,
            stop=["\n", " ", "."],
        )
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        prompts = [
            tok.apply_chat_template(
                _build_llm_prompt(t), tokenize=False, add_generation_prompt=True
            )
            for t in zs_texts
        ]
        outputs = llm.generate(prompts, sampling)
        labels_out = [o.outputs[0].text.strip().lower() for o in outputs]
        used_vllm = True
        print(f"  [LLM] vLLM inference complete ({len(labels_out)} outputs)")

    except ImportError:
        print("  [LLM] vLLM not available — falling back to HuggingFace transformers.")
    except Exception as e:
        print(f"  [LLM] vLLM error ({e}) — falling back to HuggingFace transformers.")

    # ── HuggingFace fallback ──────────────────────────────────────────────────
    if not used_vllm:
        try:
            import torch
            from transformers import (AutoTokenizer, AutoModelForCausalLM,
                                      BitsAndBytesConfig)
        except ImportError:
            print("⚠  transformers/torch not installed — LLM pass skipped.")
            return entity_df

        bnb_cfg = None
        if LLM_CLASSIFIER_LOAD_4BIT:
            try:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except Exception:
                print("  [LLM] bitsandbytes not available; loading in fp16.")

        tok = AutoTokenizer.from_pretrained(model_name)
        # Llama-3 uses eos as pad token by default, which triggers a warning
        # when no explicit attention_mask is passed.  Setting a distinct pad
        # token avoids the warning and ensures correct masking.
        if tok.pad_token_id is None or tok.pad_token_id == tok.eos_token_id:
            tok.pad_token = tok.eos_token   # keep same token, suppress warning path
            tok.padding_side = "left"       # left-pad for causal generation

        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map=device if device != "auto" else "auto",
            dtype=torch.bfloat16,           # `dtype` replaces deprecated `torch_dtype`
        )
        mdl.eval()

        for i in range(0, len(zs_texts), batch_size):
            batch = zs_texts[i : i + batch_size]
            for text in batch:
                messages = _build_llm_prompt(text)
                encoded = tok.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                )
                # apply_chat_template returns a plain tensor when a single
                # message is passed; wrap in a dict with attention_mask so
                # generate() doesn't warn about missing mask.
                if isinstance(encoded, torch.Tensor):
                    input_ids = encoded.to(mdl.device)
                    attention_mask = torch.ones_like(input_ids).to(mdl.device)
                else:
                    input_ids      = encoded["input_ids"].to(mdl.device)
                    attention_mask = encoded["attention_mask"].to(mdl.device)

                with torch.no_grad():
                    out = mdl.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=LLM_MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )
                gen = out[0][input_ids.shape[-1]:]
                label = tok.decode(gen, skip_special_tokens=True).strip().lower()
                labels_out.append(label)
            if (i // batch_size) % 5 == 0:
                print(f"  [LLM] {min(i+batch_size, len(zs_texts))}/{len(zs_texts)} done")

    # ── Apply labels ──────────────────────────────────────────────────────────
    edf = entity_df.copy()
    n_upgraded = 0
    for idx, raw_label in zip(zs_indices, labels_out):
        # Normalise: take the first valid framing word found in the output
        label = raw_label.strip().rstrip(".:,").lower()
        matched = next((v for v in valid_labels if v in label), None)

        edf.at[idx, "zs_best_cat"]   = matched or label
        edf.at[idx, "zs_best_score"] = 1.0 if matched else 0.0   # LLM = deterministic

        if matched and edf.at[idx, "framing_category"] == "neutral":
            edf.at[idx, "framing_category"]  = matched
            edf.at[idx, "framing_source"]    = "llm_classifier"
            edf.at[idx, "framing_confidence"]= 1.0
            n_upgraded += 1

    label_dist = Counter(edf.loc[zs_mask, "framing_category"].tolist())
    print(f"  [LLM] Upgrades: {n_upgraded}/{len(zs_indices)} neutral rows → {dict(label_dist)}")
    return edf


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  COMPUTE METRICS                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_tfidf_keywords(df: pd.DataFrame, entity_df: pd.DataFrame,
                            concept_patterns: dict, top_n: int=20) -> dict:
    """
    Mention-level TF-IDF keyword extraction.

    Uses the ±2-sentence context_window around each composer mention rather than
    the full passage body_text.  Full-passage TF-IDF picks up tradition-specific
    musical vocabulary (guitar, violin, suite) that reflects WHAT is being taught,
    not HOW the composer is being framed.  Context windows are already composed of
    framing-relevant text and give sharper contrast on rhetorical signals.

    Each context window = one document; IDF is computed across all windows.
    Per-group keywords = terms with highest (group_mean − other_mean) contrast.
    """
    stopwords = _build_tfidf_stopwords(concept_patterns)

    # One row per (passage, composer) after dedup — avoids inflating counts when
    # the same composer appears in multiple sentences of the same passage.
    dom_rows  = entity_df[entity_df["dominant"]].drop_duplicates(
                    ["passage_id","composer_canonical"])
    marg_rows = entity_df[entity_df["marginalized"]].drop_duplicates(
                    ["passage_id","composer_canonical"])

    dom_texts  = dom_rows["context_window"].fillna("").apply(_clean_for_tfidf).tolist()
    marg_texts = marg_rows["context_window"].fillna("").apply(_clean_for_tfidf).tolist()

    if not dom_texts and not marg_texts:
        print("[tfidf_keywords] No dominant or marginalized context windows — skipping.")
        return {}

    all_texts = dom_texts + marg_texts
    vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATS,
                          stop_words=list(stopwords),
                          ngram_range=(1, 2), min_df=2, max_df=0.90,
                          token_pattern=r'\b[a-zA-Z][a-zA-Z]{3,}\b',
                          sublinear_tf=True)
    mat = vec.fit_transform(all_texts)
    fnames = vec.get_feature_names_out()

    n_dom  = len(dom_texts)
    dom_idx  = list(range(n_dom))
    marg_idx = list(range(n_dom, len(all_texts)))

    dom_mean  = mat[dom_idx].toarray().mean(axis=0)  if dom_idx  else np.zeros(len(fnames))
    marg_mean = mat[marg_idx].toarray().mean(axis=0) if marg_idx else np.zeros(len(fnames))

    kw = {}
    for label, mean_arr, other_arr in [("dominant",  dom_mean,  marg_mean),
                                        ("marginalized", marg_mean, dom_mean)]:
        contrast = mean_arr - other_arr
        top_idx  = contrast.argsort()[::-1][:top_n]
        kw[label] = [(fnames[i], round(float(mean_arr[i]), 4), round(float(contrast[i]), 4))
                     for i in top_idx if mean_arr[i] > 0]

    print(f"[tfidf_keywords] vocab={len(fnames)} | "
          f"dom_contexts={len(dom_texts)} marg_contexts={len(marg_texts)}")
    print(f"  TOP dominant    : {[t for t,_,_ in kw['dominant'][:8]]}")
    print(f"  TOP marginalized: {[t for t,_,_ in kw['marginalized'][:8]]}")
    return kw


def compute_metrics(df: pd.DataFrame, entity_df: pd.DataFrame,
                    concept_patterns: dict) -> dict:
    M = {}
    edf = entity_df.drop_duplicates(["passage_id","composer_canonical"]).copy()
    cats = list(FRAMING_LEXICONS.keys()) + ["neutral"]

    # Representation
    rep = (edf.groupby("composer_canonical")
           .agg(mention_count=("passage_id","count"),
                tradition=("tradition","first"), era=("era","first"),
                gender=("gender","first"), ethnicity=("ethnicity","first"),
                dominant=("dominant","first"), marginalized=("marginalized","first"),
                margin_reason=("margin_reason","first"),
                avg_chapter_position=("chapter_position","mean"),
                n_central=("passage_role",lambda x:(x=="central").sum()),
                n_supplementary=("passage_role",lambda x:(x=="supplementary").sum()),
                n_application=("passage_role",lambda x:(x=="application").sum()),
                books_in=("book_id",lambda x:sorted(x.unique().tolist())))
           .reset_index().sort_values("mention_count",ascending=False))
    rep["tradition_label"] = rep["tradition"].map(TRADITION_LABELS).fillna(rep["tradition"])
    M["representation_df"] = rep
    total = len(edf); dom = rep.loc[rep["dominant"],"mention_count"].sum()
    M["dominant_ratio"] = dom/total if total else 0.0
    print(f"[compute_metrics] dominant ratio {M['dominant_ratio']:.1%} ({int(dom)}/{total})")

    trad = (rep.groupby("tradition")
            .agg(total_mentions=("mention_count","sum"),
                 unique_composers=("composer_canonical","count"),
                 dominant_flag=("dominant","first"),
                 marginalized_flag=("marginalized","first"))
            .reset_index().sort_values("total_mentions",ascending=False))
    trad["tradition_label"] = trad["tradition"].map(TRADITION_LABELS).fillna(trad["tradition"])
    M["tradition_df"] = trad
    M["gender_df"] = rep.groupby("gender")["mention_count"].sum().reset_index().rename(columns={"mention_count":"total"})
    M["era_df"]    = rep.groupby("era")["mention_count"].sum().reset_index().rename(columns={"mention_count":"total"}).sort_values("total",ascending=False)
    edf["tradition_label"] = edf["tradition"].map(TRADITION_LABELS).fillna(edf["tradition"])
    M["role_tradition_heatmap"]    = pd.crosstab(edf["passage_role"],edf["tradition_label"]).fillna(0)
    M["framing_tradition_heatmap"] = pd.crosstab(edf["framing_category"],edf["tradition_label"]).fillna(0)
    M["framing_by_status"] = pd.crosstab(
        edf["framing_category"],
        edf["dominant"].map({True:"Dominant",False:"Marginalized"})).reindex(
        [c for c in cats if c in edf["framing_category"].unique()])

    # Curriculum presence: raw mention count + book breadth.
    # The previous composite centrality score (frequency × placement × framing)
    # was removed because its chapter-position placement term structurally
    # penalises composers who appear throughout a book (genuinely central) and
    # rewards those with a single appearance in an early chapter — the opposite
    # of what centrality means. Framing and structural placement are captured
    # by dedicated figures (figs 16–18) and should not be blended into a
    # presence metric. The table below is intentionally simple: mention count
    # (frequency) and book breadth are the only defensible presence measures
    # when corpora span textbooks of different lengths and structures.
    n_books_total = edf["book_id"].nunique()
    pres = (edf.drop_duplicates(["passage_id","composer_canonical"])
               .groupby("composer_canonical")
               .agg(mention_count=("passage_id","count"),
                    n_books=("book_id","nunique"),
                    n_central=("passage_role", lambda x: (x=="central").sum()),
                    n_supplementary=("passage_role", lambda x: (x=="supplementary").sum()),
                    n_application=("passage_role", lambda x: (x=="application").sum()),
                    tradition=("tradition","first"),
                    dominant=("dominant","first"),
                    marginalized=("marginalized","first"),
                    gender=("gender","first"))
               .reset_index())
    pres["pct_central"]  = (pres["n_central"] / pres["mention_count"]).round(3)
    pres["book_breadth"] = (pres["n_books"]   / n_books_total).round(3)
    pres["tradition_label"] = pres["tradition"].map(TRADITION_LABELS).fillna(pres["tradition"])
    M["centrality_df"] = pres.sort_values("mention_count", ascending=False)

    M["tfidf_keywords"] = (
        compute_tfidf_keywords(df, entity_df, concept_patterns)
        if TFIDF_KEYWORDS_ENABLED else {}
    )
    if not TFIDF_KEYWORDS_ENABLED:
        print("[tfidf_keywords] Skipped (TFIDF_KEYWORDS_ENABLED=False). "
              "Use --tfidf-keywords flag to enable.")

    # Co-occurrence
    cooc = Counter()
    for _, grp in edf.groupby("passage_id"):
        names = grp["composer_canonical"].unique().tolist()
        for a,b in combinations(sorted(names),2): cooc[(a,b)] += 1
    M["cooccurrence_edges"] = [(a,b,w) for (a,b),w in cooc.items()]

    # Per-book summary
    per_book = (edf.groupby("book_id")
                .agg(n_passages=("passage_id",pd.Series.nunique),
                     n_mentions=("composer_canonical","count"),
                     n_unique=("composer_canonical","nunique"),
                     pct_dominant=("dominant","mean"),
                     pct_marginalized=("marginalized","mean"),
                     pct_central=("passage_role",lambda x:(x=="central").mean()),
                     pct_normative=("framing_category",lambda x:(x=="normative").mean()),
                     pct_additive=("framing_category",lambda x:(x=="additive").mean()),
                     pct_neutral=("framing_category",lambda x:(x=="neutral").mean()))
                .reset_index())
    for c in ["pct_dominant","pct_marginalized","pct_central","pct_normative","pct_additive","pct_neutral"]:
        per_book[c] = per_book[c].mul(100).round(1)
    M["per_book_summary"] = per_book

    # Framing gap: how much MORE normative framing do dominant composers receive
    # vs marginalized composers within each book
    framing_gap_rows = []
    for book in edf["book_id"].unique():
        b = edf[edf["book_id"]==book]
        dom_b  = b[b["dominant"]==True]
        marg_b = b[b["marginalized"]==True]
        dom_norm  = (dom_b["framing_category"]=="normative").mean() if len(dom_b)  else 0.0
        marg_norm = (marg_b["framing_category"]=="normative").mean() if len(marg_b) else 0.0
        dom_add   = (dom_b["framing_category"]=="additive").mean()   if len(dom_b)  else 0.0
        marg_add  = (marg_b["framing_category"]=="additive").mean()  if len(marg_b) else 0.0
        dom_exc   = (dom_b["framing_category"]=="exceptional").mean() if len(dom_b) else 0.0
        marg_exc  = (marg_b["framing_category"]=="exceptional").mean() if len(marg_b) else 0.0
        framing_gap_rows.append({
            "book_id":          book,
            "dom_normative":    round(dom_norm,  3),
            "marg_normative":   round(marg_norm, 3),
            "dom_additive":     round(dom_add,   3),
            "marg_additive":    round(marg_add,  3),
            "dom_exceptional":  round(dom_exc,   3),
            "marg_exceptional": round(marg_exc,  3),
            "normative_gap":    round(dom_norm - marg_norm, 3),
            "additive_gap":     round(marg_add  - dom_add,  3),
            "n_dom_mentions":   len(dom_b),
            "n_marg_mentions":  len(marg_b),
        })
    M["framing_gap_df"] = pd.DataFrame(framing_gap_rows).sort_values("normative_gap",ascending=False)

    # Integration index: % of marginalized-composer passages that also cite a dominant composer
    integration_rows = []
    for book in edf["book_id"].unique():
        b = edf[edf["book_id"]==book]
        marg_passages = set(b[b["marginalized"]==True]["passage_id"])
        dom_passages  = set(b[b["dominant"]==True]["passage_id"])
        integrated    = marg_passages & dom_passages
        n_marg = len(marg_passages)
        integration_rows.append({
            "book_id":         book,
            "n_marg_passages": n_marg,
            "n_integrated":    len(integrated),
            "integration_pct": round(len(integrated)/n_marg, 3) if n_marg else 0.0,
        })
    M["integration_df"] = pd.DataFrame(integration_rows).sort_values("integration_pct",ascending=False)

    # Structural placement: % of mentions in central vs supplementary passages
    placement_rows = []
    for book in edf["book_id"].unique():
        b = edf[edf["book_id"]==book]
        for status, label in [(True,"dominant"),(False,"marginalized")]:
            col = "dominant" if status else "marginalized"
            sub = b[b[col]==True]
            if len(sub) == 0: continue
            roles = sub["passage_role"].value_counts(normalize=True)
            placement_rows.append({
                "book_id":           book,
                "status":            label,
                "pct_central":       round(roles.get("central",   roles.get("primary",  0)), 3),
                "pct_supplementary": round(roles.get("supplementary", roles.get("secondary", 0)), 3),
                "pct_application":   round(roles.get("application",0), 3),
                "n_mentions":        len(sub),
            })
    M["placement_df"] = pd.DataFrame(placement_rows)

    # ── Era × demographic breakdown ─────────────────────────────────────────
    M["era_demographic_df"] = compute_era_representation(edf)

    return M


def compute_era_representation(edf: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tabulate composer mentions by era × gender and era × BIPOC status.

    Returns a DataFrame indexed by era with columns:
        total, n_female, n_male, n_bipoc, n_nonbipoc,
        pct_female, pct_bipoc, pct_marginalized,
        n_dominant, n_marginalized

    Used by visualize_results() for era-period representation plots (figs 3b/3c).
    """
    ERA_ORDER = ["renaissance","baroque","classical","romantic","modern","contemporary","unknown"]

    if edf is None or edf.empty:
        return pd.DataFrame()

    rows = []
    for era in ERA_ORDER:
        sub = edf[edf["era"] == era]
        if len(sub) == 0:
            continue
        n_total  = len(sub)
        n_female = (sub["gender"] == "female").sum()
        n_male   = (sub["gender"] == "male").sum()
        n_bipoc  = sub.get("is_bipoc", sub["marginalized"]).sum() \
                   if "is_bipoc" in sub.columns \
                   else (sub["marginalized"] & (sub["gender"] != "female")).sum()
        n_dom    = sub["dominant"].sum()
        n_marg   = sub["marginalized"].sum()
        rows.append({
            "era":             era,
            "total":           n_total,
            "n_female":        int(n_female),
            "n_male":          int(n_male),
            "n_bipoc":         int(n_bipoc),
            "pct_female":      round(100 * n_female / n_total, 1),
            "pct_bipoc":       round(100 * n_bipoc  / n_total, 1),
            "pct_marginalized":round(100 * n_marg   / n_total, 1),
            "n_dominant":      int(n_dom),
            "n_marginalized":  int(n_marg),
        })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out["era"] = pd.Categorical(df_out["era"], categories=ERA_ORDER, ordered=True)
        df_out = df_out.sort_values("era").reset_index(drop=True)
    return df_out


def compute_concept_representation(df: pd.DataFrame, entity_df: pd.DataFrame,
                                    concept_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse which music theory concepts co-occur with dominant vs. marginalized
    composer mentions. Requires concept_df from detect_concepts_in_passages().
    """
    if concept_df is None or concept_df.shape[1] == 0:
        print("[compute_concept_representation] No concept columns — skipping.")
        return pd.DataFrame()
    if concept_df.shape[0] == 0:
        print("[compute_concept_representation] No rows in concept_df — skipping.")
        return pd.DataFrame()

    print(f"[compute_concept_representation] concept_df shape: {concept_df.shape}")

    pid_to_idx = {row["passage_id"]: i for i,(_, row) in enumerate(df.iterrows())}
    dom_ids    = set(entity_df.loc[entity_df["dominant"],    "passage_id"])
    marg_ids   = set(entity_df.loc[entity_df["marginalized"],"passage_id"])
    dom_idx    = [pid_to_idx[p] for p in dom_ids  if p in pid_to_idx]
    marg_idx   = [pid_to_idx[p] for p in marg_ids if p in pid_to_idx]

    if not dom_idx or not marg_idx:
        print("[compute_concept_representation] No dominant or marginalized passages matched.")
        return pd.DataFrame()

    arr       = concept_df.values
    dom_mean  = arr[dom_idx, :].mean(axis=0)
    marg_mean = arr[marg_idx,:].mean(axis=0)
    all_mean  = arr.mean(axis=0)
    contrast  = dom_mean - marg_mean

    result = pd.DataFrame({
        "concept":         concept_df.columns,
        "overall_freq":    all_mean.round(3),
        "dom_freq":        dom_mean.round(3),
        "marg_freq":       marg_mean.round(3),
        "contrast":        contrast.round(3),
        "n_dom_passages":  arr[dom_idx, :].sum(axis=0).astype(int),
        "n_marg_passages": arr[marg_idx,:].sum(axis=0).astype(int),
        "n_all_passages":  arr.sum(axis=0).astype(int),
    }).sort_values("contrast", ascending=False).reset_index(drop=True)

    result["association"] = pd.cut(
        result["contrast"],
        bins  =[-1.0,-0.10,-0.03,0.03,0.10,1.0],
        labels=["strongly marginalized","marginalized","neutral",
                "dominant","strongly dominant"],
    )
    print(f"[compute_concept_representation] {len(result)} concepts analysed")
    print("  Dominant-associated (Δ > 0.10):")
    for _,r in result[result["contrast"] > 0.10].iterrows():
        print(f"    {r['concept']:40s}  dom={r['dom_freq']:.2f}  marg={r['marg_freq']:.2f}  Δ={r['contrast']:+.2f}")
    print("  Marginalized-associated (Δ < -0.10):")
    for _,r in result[result["contrast"] < -0.10].iterrows():
        print(f"    {r['concept']:40s}  dom={r['dom_freq']:.2f}  marg={r['marg_freq']:.2f}  Δ={r['contrast']:+.2f}")
    return result


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VISUALISE                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _fp(name: str, output_dir: str) -> Path:
    p = Path(output_dir)/"figures"/f"{name}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _safe_node_label(name: str, all_nodes: list) -> str:
    """Return short but unambiguous graph label."""
    parts = name.split()
    last  = parts[-1] if parts else name
    clash = any(n != name and n.split()[-1].lower() == last.lower() for n in all_nodes)
    if not clash: return last[:18]
    return (f"{parts[0][0]}. {last}" if len(parts) >= 2 else name)[:18]


def plot_cooccurrence_network(metrics: dict, entity_df: pd.DataFrame,
                               fpath: Path) -> None:
    """Deduplicated co-occurrence network with unambiguous node labels."""
    cooc   = metrics["cooccurrence_edges"]
    cent   = metrics["centrality_df"]
    thresh = COOC_THRESHOLD
    edges  = [(a,b,w) for a,b,w in cooc if w >= thresh and a != b]
    if len(edges) < 5:
        edges = sorted(cooc, key=lambda x:-x[2])[:40]
        edges = [(a,b,w) for a,b,w in edges if a != b]
    G = nx.Graph()
    for a,b,w in edges:
        G.add_node(a, **_get_meta(a)); G.add_node(b, **_get_meta(b))
        G.add_edge(a, b, weight=w)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    all_nodes  = list(G.nodes())
    node_cols  = [_composer_color(G.nodes[n].get("dominant",False),
                                  G.nodes[n].get("marginalized",False))
                  for n in all_nodes]
    node_sizes = []
    for n in all_nodes:
        row = cent.loc[cent["composer_canonical"]==n, "mention_count"]
        mc = float(row.iloc[0]) if len(row) else 1.0
        max_mc = cent["mention_count"].max() if len(cent) else 1.0
        node_sizes.append(80 + 620 * (mc / max_mc))
    ew    = [G[u][v]["weight"] for u,v in G.edges()]
    max_w = max(ew) if ew else 1
    labels = {n: _safe_node_label(n, all_nodes) for n in all_nodes}
    fig, ax = plt.subplots(figsize=(15,11))
    pos = nx.spring_layout(G, k=2.8, seed=42, weight="weight")
    nx.draw_networkx_edges(G,pos,ax=ax,alpha=0.3,width=[1.2+3.5*w/max_w for w in ew],edge_color="#aaaaaa")
    nx.draw_networkx_nodes(G,pos,ax=ax,node_color=node_cols,node_size=node_sizes,alpha=0.88,edgecolors="white",linewidths=0.8)
    nx.draw_networkx_labels(G,pos,labels=labels,ax=ax,font_size=7)
    ax.set_title(f"Co-occurrence Network (≥{thresh} shared passages; {G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=PALETTE["dominant"],label="Dominant"),
                       mpatches.Patch(color=PALETTE["marginalized"],label="Marginalized"),
                       mpatches.Patch(color=PALETTE["unclassified"],label="Not yet classified")])
    ax.axis("off"); plt.tight_layout()
    fig.savefig(fpath); plt.close(fig)
    print(f"  ✓ fig_08_cooccurrence_network ({G.number_of_nodes()} nodes)")


def visualize_results(metrics: dict, entity_df: pd.DataFrame, df: pd.DataFrame,
                       output_dir: str) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 130
    rep=metrics["representation_df"]; trad=metrics["tradition_df"]
    gender=metrics["gender_df"]; era=metrics["era_df"]
    cent=metrics["centrality_df"]; kw=metrics["tfidf_keywords"]
    framing_status=metrics["framing_by_status"]; per_book=metrics["per_book_summary"]
    role_hm=metrics["role_tradition_heatmap"]; frame_hm=metrics["framing_tradition_heatmap"]
    dom_patch  = mpatches.Patch(color=PALETTE["dominant"],    label="Dominant (Western Canon)")
    marg_patch = mpatches.Patch(color=PALETTE["marginalized"],label="Marginalized / Non-Dominant")

    def fp(name): return _fp(name, output_dir)

    # Fig 1: Tradition
    fig, ax = plt.subplots(figsize=(11,6))
    tp = trad.sort_values("total_mentions")
    cols = [_composer_color(d, m) for d, m in zip(tp["dominant_flag"], tp["marginalized_flag"])]
    bars = ax.barh(tp["tradition_label"],tp["total_mentions"],color=cols,edgecolor="white")
    for bar,v in zip(bars,tp["total_mentions"]):
        ax.text(bar.get_width()+.3,bar.get_y()+bar.get_height()/2,str(int(v)),va="center",fontsize=9)
    ax.set_xlabel("Composer-Passage Mentions"); ax.set_title("Representation by Musical Tradition",fontweight="bold")
    ax.legend(handles=[dom_patch,marg_patch]); ax.set_xlim(0,tp["total_mentions"].max()*1.18)
    plt.tight_layout(); fig.savefig(fp("fig_01_representation_by_tradition")); plt.close(fig)

    # Fig 2: Gender
    fig, ax = plt.subplots(figsize=(7,4))
    gp = gender.sort_values("total",ascending=False)
    ax.bar(gp["gender"].str.title(),gp["total"],color=[PALETTE.get(g,PALETTE["neutral"]) for g in gp["gender"]],edgecolor="white")
    for i,(_,row) in enumerate(gp.iterrows()): ax.text(i,row["total"]+.5,str(int(row["total"])),ha="center",fontsize=10)
    ax.set_ylabel("Mentions"); ax.set_title("Composer Mentions by Gender",fontweight="bold")
    plt.tight_layout(); fig.savefig(fp("fig_02_gender_distribution")); plt.close(fig)

    # Fig 3: Era — total mentions
    era_ord = ["renaissance","baroque","classical","romantic","modern","contemporary","unknown"]
    ep = era.set_index("era").reindex([e for e in era_ord if e in era["era"].values]).reset_index()
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar(ep["era"].str.title(),ep["total"],color=PALETTE["dominant"],edgecolor="white")
    ax.set_ylabel("Mentions"); ax.set_title("Composer Mentions by Era — Total",fontweight="bold")
    plt.tight_layout(); fig.savefig(fp("fig_03_era_distribution")); plt.close(fig)

    # Fig 3b: Era × gender + BIPOC — stacked absolute counts
    era_demo = metrics.get("era_demographic_df", pd.DataFrame())
    if not era_demo.empty:
        ep3 = era_demo[era_demo["era"].isin(era_ord)].copy()
        ep3["era_label"] = ep3["era"].str.title()

        # Stacked bar: male (dominant palette) / female (female palette) / BIPOC overlap (group palette)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.bar(ep3["era_label"], ep3["n_male"],   color=PALETTE["male"],   label="Male",   edgecolor="white")
        ax.bar(ep3["era_label"], ep3["n_female"], color=PALETTE["female"], label="Female",
               bottom=ep3["n_male"], edgecolor="white")
        ax.set_ylabel("Composer-Passage Mentions")
        ax.set_title("Representation by Era × Gender", fontweight="bold")
        ax.legend(loc="upper right")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

        ax = axes[1]
        ax.bar(ep3["era_label"], ep3["pct_female"], color=PALETTE["female"], edgecolor="white", label="% Female")
        ax.bar(ep3["era_label"], ep3["pct_bipoc"],  color=PALETTE["group"],  edgecolor="white",
               alpha=0.75, label="% BIPOC", bottom=ep3["pct_female"])
        ax.set_ylabel("% of Mentions in Era")
        ax.set_title("% Female and BIPOC Mentions by Era", fontweight="bold")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 110)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

        plt.suptitle("Era-Period Demographic Representation", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(fp("fig_03b_era_demographic"), bbox_inches="tight"); plt.close(fig)
        print("  ✓ fig_03b_era_demographic")

        # Fig 3c: Era × dominant / marginalized / unclassified — stacked %
        fig, ax = plt.subplots(figsize=(10, 4))
        n_unc = ep3["total"] - ep3["n_dominant"] - ep3["n_marginalized"]
        n_unc = n_unc.clip(lower=0)
        pct_dom  = (ep3["n_dominant"]   / ep3["total"] * 100).round(1)
        pct_marg = (ep3["n_marginalized"]/ ep3["total"] * 100).round(1)
        pct_unc  = (n_unc                / ep3["total"] * 100).round(1)
        ax.bar(ep3["era_label"], pct_dom,  color=PALETTE["dominant"],      label="Dominant",          edgecolor="white")
        ax.bar(ep3["era_label"], pct_marg, color=PALETTE["marginalized"],  label="Marginalized",
               bottom=pct_dom, edgecolor="white")
        ax.bar(ep3["era_label"], pct_unc,  color=PALETTE["unclassified"],  label="Not yet classified",
               bottom=pct_dom + pct_marg, edgecolor="white")
        ax.set_ylabel("% of Mentions")
        ax.set_title("Dominant vs. Marginalized Composer Mentions by Era (%)", fontweight="bold")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 115)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        plt.tight_layout()
        fig.savefig(fp("fig_03c_era_dominance"), bbox_inches="tight"); plt.close(fig)
        print("  ✓ fig_03c_era_dominance")

    # Fig 4: Role × Tradition heatmap
    if not role_hm.empty:
        fig, ax = plt.subplots(figsize=(14,4))
        ro = [r for r in ["central","application","supplementary"] if r in role_hm.index]
        rp = role_hm.reindex(ro).dropna(how="all",axis=1)
        sns.heatmap(rp.div(rp.sum(axis=0)+1e-9)*100,annot=rp.astype(int),fmt="d",cmap="YlOrRd",linewidths=.5,ax=ax,cbar_kws={"label":"% of tradition"})
        ax.set_title("Structural Placement: Passage Role × Tradition\n(count; colour=% of column)",fontweight="bold")
        plt.xticks(rotation=35,ha="right"); plt.tight_layout()
        fig.savefig(fp("fig_04_role_tradition_heatmap")); plt.close(fig)

    # Fig 5: Framing × Tradition heatmap
    if not frame_hm.empty:
        fig, ax = plt.subplots(figsize=(14,5))
        fo = [f for f in ["normative","additive","neutral","corrective","exceptional"] if f in frame_hm.index]
        fp_ = frame_hm.reindex(fo).dropna(how="all",axis=1)
        sns.heatmap(fp_.div(fp_.sum(axis=0)+1e-9)*100,annot=fp_.astype(int),fmt="d",cmap="Blues",linewidths=.5,ax=ax,cbar_kws={"label":"% within tradition"})
        ax.set_title("Framing Category × Tradition\n(count; colour=% within tradition)",fontweight="bold")
        plt.xticks(rotation=35,ha="right"); plt.tight_layout()
        fig.savefig(fp("fig_05_framing_tradition_heatmap")); plt.close(fig)

    # Fig 6: Framing by status
    fig, ax = plt.subplots(figsize=(9,5))
    if framing_status is not None and not framing_status.empty:
        framing_status.plot(kind="bar",ax=ax,color=[PALETTE["dominant"],PALETTE["marginalized"]],edgecolor="white",width=.7)
    ax.set_title("Framing Category: Dominant vs. Marginalized",fontweight="bold")
    ax.set_xlabel("Framing Category"); ax.set_ylabel("Mentions")
    ax.legend(title="Status"); plt.xticks(rotation=20,ha="right")
    plt.tight_layout(); fig.savefig(fp("fig_06_framing_by_dominance")); plt.close(fig)

    # Fig 7: Top 30 composers — structural placement (central / supplementary / application)
    # This replaces the uninformative raw mention-count bar chart.  Structural
    # placement directly answers the representation question: are marginalized
    # composers integrated into core pedagogy (central passages) or confined to
    # exercises and supplementary sidebars?
    top30 = cent.head(30).copy()

    # Colour each row by dominant / marginalized / unclassified
    bar_cols = [_composer_color(d, m)
                for d, m in zip(top30["dominant"], top30["marginalized"])]

    # Stacked segment widths: central (full opacity) / supplementary (light) / application (neutral)
    # n_central, n_supplementary, n_application are already in centrality_df
    n_central     = top30.get("n_central",     pd.Series([0]*len(top30))).fillna(0).astype(int)
    n_supplementary = top30.get("n_supplementary", pd.Series([0]*len(top30))).fillna(0).astype(int)
    n_application = top30.get("n_application", pd.Series([0]*len(top30))).fillna(0).astype(int)
    total_mentions = n_central + n_supplementary + n_application
    # Fall back to mention_count if sub-columns are all zero
    if total_mentions.sum() == 0:
        n_central = top30["mention_count"].astype(int)
        n_supplementary = pd.Series([0]*len(top30))
        n_application   = pd.Series([0]*len(top30))
        total_mentions  = n_central

    names_rev = list(top30["composer_canonical"])[::-1]
    nc_rev  = list(n_central)[::-1]
    ns_rev  = list(n_supplementary)[::-1]
    na_rev  = list(n_application)[::-1]
    col_rev = bar_cols[::-1]

    fig, ax = plt.subplots(figsize=(12, 9))
    y_pos = range(len(names_rev))

    # Central mentions — full saturation of the group colour
    bars_c = ax.barh(list(y_pos), nc_rev,
                     color=col_rev, edgecolor="white", alpha=1.0, label="Central")
    # Supplementary — desaturated (alpha 0.45)
    ax.barh(list(y_pos), ns_rev, left=nc_rev,
            color=col_rev, edgecolor="white", alpha=0.40, label="Supplementary")
    # Application — neutral grey
    ax.barh(list(y_pos), na_rev, left=[c+s for c,s in zip(nc_rev, ns_rev)],
            color=PALETTE["neutral"], edgecolor="white", alpha=0.55, label="Application")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names_rev, fontsize=9)
    ax.set_xlabel("Composer-Passage Mentions")
    ax.set_title(
        "Top 30 Composers: Structural Placement\n"
        "(dark = central passages  ·  light = supplementary  ·  grey = application/exercises)",
        fontweight="bold", fontsize=10)

    # Legend: group colours + placement meaning
    unc_patch  = mpatches.Patch(color=PALETTE["unclassified"], label="Not yet classified")
    cen_patch  = mpatches.Patch(color="#555555",               label="Dark fill = central passages")
    sup_patch  = mpatches.Patch(facecolor="#555555", alpha=0.40, edgecolor="none",
                                label="Light fill = supplementary")
    app_patch  = mpatches.Patch(color=PALETTE["neutral"],      label="Grey fill = application")
    ax.legend(handles=[dom_patch, marg_patch, unc_patch,
                        mpatches.Patch(color="none", label=""),
                        cen_patch, sup_patch, app_patch],
              loc="lower right", fontsize=8, framealpha=0.85)

    # Annotate: total count + book breadth
    max_total = total_mentions.max() if total_mentions.max() > 0 else 1
    for i, (_, row) in enumerate(top30[::-1].iterrows()):
        n_tot = int(n_central.iloc[::-1].iloc[i] + n_supplementary.iloc[::-1].iloc[i]
                    + n_application.iloc[::-1].iloc[i])
        n_books = int(row.get("n_books", 0))
        pct_c   = 100 * int(n_central.iloc[::-1].iloc[i]) / max(n_tot, 1)
        lbl = f"  {n_tot}  ({n_books}bk, {pct_c:.0f}%↑)"
        ax.text(max_total * 0.01 + n_tot, i, lbl, va="center", fontsize=7.5)

    ax.set_xlim(0, max_total * 1.30)
    plt.tight_layout()
    fig.savefig(fp("fig_07_structural_placement"))
    plt.close(fig)
    print("  ✓ fig_07_structural_placement")

    # Fig 8: Co-occurrence network
    plot_cooccurrence_network(metrics, entity_df, fp("fig_08_cooccurrence_network"))

    # Fig 9: TF-IDF keywords
    if kw and "dominant" in kw and "marginalized" in kw:
        fig, axes = plt.subplots(1,2,figsize=(14,6))
        for ax,(lbl,col) in zip(axes,[("dominant",PALETTE["dominant"]),("marginalized",PALETTE["marginalized"])]):
            items = kw[lbl][:15]
            terms=[t for t,_,_ in items]; vals=[m for _,m,_ in items]; contrasts=[c for _,_,c in items]
            ax.barh(terms[::-1],vals[::-1],color=col,edgecolor="white",alpha=.85,label="Mean TF-IDF")
            ax.barh(terms[::-1],contrasts[::-1],color=col,edgecolor="white",alpha=.40,hatch="///",label="Contrast vs other group")
            ax.set_title(f"Top Keywords — {lbl.title()} Passages\n(solid=mean TF-IDF; hatched=contrast score)",fontsize=10,fontweight="bold")
            ax.set_xlabel("Score"); ax.legend(fontsize=8)
        plt.suptitle("Linguistic Profile: Dominant vs. Marginalized Vocabulary",fontsize=11,fontweight="bold",y=1.01)
        plt.tight_layout(); fig.savefig(fp("fig_09_tfidf_keywords"),bbox_inches="tight"); plt.close(fig)

    # Fig 10: Chapter position by book — faceted density plots
    ep2 = entity_df.drop_duplicates(["passage_id","composer_canonical"]).copy()
    ep2["Status"] = ep2.apply(
        lambda r: "Dominant"      if r["dominant"]
             else "Marginalized"  if r["marginalized"]
             else "Unclassified",
        axis=1
    )
    ep2["tradition_label"] = ep2["tradition"].map(TRADITION_LABELS).fillna(ep2["tradition"])

    books = sorted(ep2["book_id"].unique())
    n_books = len(books)
    ncols = min(4, n_books)
    nrows = (n_books + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                              sharex=True, sharey=False)
    axes_flat = np.array(axes).flatten() if n_books > 1 else [axes]

    status_cols = [("Dominant",     PALETTE["dominant"]),
                   ("Marginalized", PALETTE["marginalized"]),
                   ("Unclassified", PALETTE["unclassified"])]
    for idx, book in enumerate(books):
        ax = axes_flat[idx]
        sub = ep2[ep2["book_id"] == book]
        any_data = False
        for status, col in status_cols:
            s = sub.loc[sub["Status"] == status, "chapter_position"]
            if not s.empty:
                ax.hist(s, bins=10, alpha=0.65, color=col, label=status,
                        edgecolor="white", density=True)
                any_data = True
        ax.set_title(book.replace("_"," "), fontsize=9, fontweight="bold")
        ax.set_xlabel("Chapter position (0→1)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0: ax.legend(fontsize=7, loc="upper right")

    # Hide any spare axes
    for ax in axes_flat[n_books:]:
        ax.set_visible(False)

    plt.suptitle("Chapter Position of Composer Mentions by Textbook",
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(fp("fig_10_chapter_position_by_book"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_10_chapter_position_by_book")

    # Fig 10b: Passage Role by Tradition (separated from Fig 10)
    rc = ep2.groupby(["tradition_label","passage_role"]).size().unstack(fill_value=0)
    rc = rc[rc.sum(axis=1) > 1]
    role_col_map = {"central":   PALETTE["dominant"],
                    "application":PALETTE["neutral"],
                    "supplementary":PALETTE["marginalized"]}
    cols_p = [c for c in ["central","application","supplementary"] if c in rc.columns]
    fig, ax = plt.subplots(figsize=(10, max(4, len(rc) * 0.55)))
    rc[cols_p].plot(kind="barh", ax=ax, stacked=True,
                    color=[role_col_map[c] for c in cols_p], edgecolor="white")
    ax.set_title("Passage Role by Musical Tradition", fontweight="bold")
    ax.set_xlabel("Composer-Passage Mentions")
    ax.set_ylabel("")
    ax.legend(title="Role", fontsize=9)
    plt.tight_layout()
    fig.savefig(fp("fig_10b_passage_role_by_tradition"))
    plt.close(fig)
    print("  ✓ fig_10b_passage_role_by_tradition")

    # Fig 11: Genre mention × passage_role
    if "genre_mentions" in df.columns:
        genre_role = [{"genre":g,"passage_role":row["passage_role"]}
                      for _,row in df.iterrows() for g in row["genre_mentions"]]
        if genre_role:
            gr_df = pd.DataFrame(genre_role)
            gr_ct = pd.crosstab(gr_df["genre"], gr_df["passage_role"])
            fig, ax = plt.subplots(figsize=(10,5))
            gr_ct.plot(kind="barh",ax=ax,stacked=True,
                       color=[PALETTE["dominant"],PALETTE["neutral"],PALETTE["marginalized"]][:len(gr_ct.columns)],
                       edgecolor="white")
            ax.set_title("Explicit Genre Mentions × Passage Role\n(text-evidence genre, independent of composer metadata)",fontweight="bold")
            ax.set_xlabel("Passages"); ax.legend(title="Role",fontsize=8)
            plt.tight_layout(); fig.savefig(fp("fig_11_genre_mentions")); plt.close(fig)
            print("  ✓ fig_11_genre_mentions")

    # Fig 12: BERT clusters (if active)
    if "bert_cluster" in entity_df.columns and (entity_df["bert_cluster"]>=0).any():
        ct = pd.crosstab(entity_df["bert_cluster"],entity_df["framing_category"])
        fig, ax = plt.subplots(figsize=(10,5))
        sns.heatmap(ct,annot=True,fmt="d",cmap="Purples",linewidths=.5,ax=ax)
        ax.set_title("BERT Clusters × Rule Framing\n(off-diagonal = potential rule gaps)",fontweight="bold")
        plt.tight_layout(); fig.savefig(fp("fig_12_bert_clusters")); plt.close(fig)
        print("  ✓ fig_12_bert_clusters")

    # Fig 13a: Dominant vs. Marginalized representation by book
    if len(per_book) > 1:
        x = np.arange(len(per_book)); w = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(per_book)*1.2), 5))
        ax.bar(x - w/2, per_book["pct_dominant"],    w, label="% Dominant",
               color=PALETTE["dominant"],    edgecolor="white")
        ax.bar(x + w/2, per_book["pct_marginalized"], w, label="% Marginalized",
               color=PALETTE["marginalized"], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(per_book["book_id"], rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("% of composer-passage mentions")
        ax.set_title("Dominant vs. Marginalized Representation by Textbook",
                     fontweight="bold")
        ax.legend()
        # Annotate with raw counts
        for i, (_, row) in enumerate(per_book.iterrows()):
            ax.text(i - w/2, row["pct_dominant"] + 0.5,
                    f"{row['pct_dominant']:.0f}%", ha="center", fontsize=7.5)
            ax.text(i + w/2, row["pct_marginalized"] + 0.5,
                    f"{row['pct_marginalized']:.0f}%", ha="center", fontsize=7.5)
        plt.tight_layout()
        fig.savefig(fp("fig_13a_representation_by_book"))
        plt.close(fig)
        print("  ✓ fig_13a_representation_by_book")

    # Fig 13b: Framing distribution by book
    if len(per_book) > 1:
        x = np.arange(len(per_book)); w = 0.28
        fig, ax = plt.subplots(figsize=(max(8, len(per_book)*1.2), 5))
        for i, (col, label, col_pal) in enumerate([
            ("pct_normative", "Normative",   PALETTE["dominant"]),
            ("pct_additive",  "Additive",    PALETTE["neutral"]),
            ("pct_neutral",   "Neutral",     PALETTE["unclassified"]),
        ]):
            ax.bar(x + (i-1)*w, per_book[col], w, label=label,
                   color=col_pal, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(per_book["book_id"], rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("% of composer-passage mentions")
        ax.set_title("Framing Category Distribution by Textbook", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        fig.savefig(fp("fig_13b_framing_by_book"))
        plt.close(fig)
        print("  ✓ fig_13b_framing_by_book")

    # Fig 16: Framing gap — normative and additive by book
    gap_df = metrics.get("framing_gap_df")
    if gap_df is not None and not gap_df.empty:
        books_ordered = gap_df.sort_values("normative_gap", ascending=False)["book_id"].tolist()
        x = np.arange(len(books_ordered)); w = 0.22
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        for i, (col, label, col_pal) in enumerate([
            ("dom_normative",  "Dominant",    PALETTE["dominant"]),
            ("marg_normative", "Marginalized", PALETTE["marginalized"]),
        ]):
            vals = [gap_df.loc[gap_df["book_id"]==b, col].iloc[0]
                    if b in gap_df["book_id"].values else 0 for b in books_ordered]
            ax.bar(x + (i-0.5)*w, [v*100 for v in vals], w,
                   label=label, color=col_pal, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(books_ordered, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("% of mentions with normative framing")
        ax.set_title("Normative Framing: Dominant vs. Marginalized",fontweight="bold", fontsize=10)
        ax.legend()
        ax = axes[1]
        for i, (col, label, col_pal) in enumerate([
            ("dom_additive",  "Dominant",    PALETTE["dominant"]),
            ("marg_additive", "Marginalized", PALETTE["marginalized"]),
        ]):
            vals = [gap_df.loc[gap_df["book_id"]==b, col].iloc[0]
                    if b in gap_df["book_id"].values else 0 for b in books_ordered]
            ax.bar(x + (i-0.5)*w, [v*100 for v in vals], w,
                   label=label, color=col_pal, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(books_ordered, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("% of mentions with additive framing")
        ax.set_title("Additive Framing: Dominant vs. Marginalized",fontweight="bold", fontsize=10)
        ax.legend()
        plt.suptitle("Framing Equity: How Each Book Introduces Dominant vs. Marginalized Composers",
                     fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        fig.savefig(fp("fig_16_framing_gap_by_book"), bbox_inches="tight")
        plt.close(fig)
        print("  ✓ fig_16_framing_gap_by_book")

    # Fig 17: Structural placement — % central by dominant/marginalized
    placement = metrics.get("placement_df")
    if placement is not None and not placement.empty:
        books_with_roles = placement[placement["pct_central"] > 0]["book_id"].unique()
        if len(books_with_roles):
            pivot = placement[placement["book_id"].isin(books_with_roles)].pivot(
                index="book_id", columns="status", values="pct_central").fillna(0) * 100
            pivot = pivot.sort_values("dominant", ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(pivot)); w = 0.35
            for i, (col, col_pal) in enumerate([("dominant", PALETTE["dominant"]),
                                                 ("marginalized", PALETTE["marginalized"])]):
                if col in pivot.columns:
                    ax.bar(x + (i-0.5)*w, pivot[col], w, label=col.title(),
                           color=col_pal, edgecolor="white")
            ax.set_xticks(x); ax.set_xticklabels(pivot.index, rotation=20, ha="right")
            ax.set_ylabel("% in central (core) passages"); ax.set_ylim(0, 110)
            ax.axhline(100, color="grey", linewidth=0.5, linestyle=":")
            ax.set_title("Structural Placement: % of Mentions in Central Passages",fontweight="bold")
            ax.legend(title="Composer group")
            plt.tight_layout()
            fig.savefig(fp("fig_17_structural_placement"))
            plt.close(fig)
            print("  ✓ fig_17_structural_placement")

    # Fig 18: Integration index
    integration = metrics.get("integration_df")
    if integration is not None and not integration.empty and integration["n_marg_passages"].sum() > 0:
        intg = integration.sort_values("integration_pct", ascending=True)
        intg = intg[intg["n_marg_passages"] > 0]
        fig, ax = plt.subplots(figsize=(9, 4))
        colours_intg = [PALETTE["dominant"] if v >= 0.5 else
                        PALETTE["neutral"] if v >= 0.25 else
                        PALETTE["marginalized"] for v in intg["integration_pct"]]
        bars = ax.barh(intg["book_id"], intg["integration_pct"] * 100,
                       color=colours_intg, edgecolor="white")
        for bar, (_, row) in zip(bars, intg.iterrows()):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{row['integration_pct']:.0%}  (n={int(row['n_marg_passages'])})",
                    va="center", fontsize=9)
        ax.set_xlabel("% of marginalized-composer passages that also cite a dominant composer")
        ax.set_title("Integration Index: Are Diverse Composers Cited Alongside the Canon?",fontweight="bold")
        ax.set_xlim(0, 120)
        plt.tight_layout()
        fig.savefig(fp("fig_18_integration_index"))
        plt.close(fig)
        print("  ✓ fig_18_integration_index")

    print(f"\n✓ All figures saved → {Path(output_dir)/'figures'}")


def plot_concept_representation(concept_rep: pd.DataFrame,
                                 concept_df: pd.DataFrame,
                                 entity_df: pd.DataFrame,
                                 df: pd.DataFrame,
                                 output_dir: str) -> None:
    """
    Fig 14  Concept contrast chart — horizontal bars coloured by dominant/marginalized
            association, annotated with raw passage counts.

    Fig 15  Concept × Tradition heatmap — rows = musical traditions (≥3 passages),
            columns = concepts. Cell = fraction of that tradition's passages containing
            the concept. Reveals which theoretical apparatus is deployed per tradition.
    """
    if concept_rep is None or concept_rep.empty:
        print("[plot_concept_representation] No concept data — skipping figs 14–15.")
        return

    def fp(name): return _fp(name, output_dir)

    out = Path(output_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 130

    # Fig 14: Contrast chart
    cr = concept_rep.copy().sort_values("contrast")
    colours = [PALETTE["dominant"] if c > 0 else PALETTE["marginalized"]
               for c in cr["contrast"]]
    fig, ax = plt.subplots(figsize=(11, max(5, len(cr) * 0.52)))
    bars = ax.barh(cr["concept"], cr["contrast"], color=colours,
                   edgecolor="white", linewidth=0.5)
    ax.axvspan(-0.05, 0.05, color="grey", alpha=0.08)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    for bar, (_, row) in zip(bars, cr.iterrows()):
        x = bar.get_width()
        label = f"d={row['n_dom_passages']} m={row['n_marg_passages']}"
        ax.text(x + (0.01 if x >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                label, va="center", ha="left" if x >= 0 else "right", fontsize=7)
    ax.set_xlabel("Contrast (dominant freq − marginalized freq)")
    ax.set_title("Concept Association: Dominant vs. Marginalized Composer Passages\n"
                 "(grey band = neutral zone |Δ| < 0.05)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(fp("fig_14_concept_contrast"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_14_concept_contrast")

    # Fig 15: Concept × Tradition heatmap
    if concept_df.shape[1] > 0:
        df2 = df.copy()
        df2["tradition_label"] = (entity_df.groupby("passage_id")["tradition"]
                                  .first().map(TRADITION_LABELS)
                                  .reindex(df2["passage_id"]).values)
        trad_groups = df2.groupby("tradition_label")
        min_passages = 3
        valid_trads = [t for t, g in trad_groups if len(g) >= min_passages]
        if valid_trads:
            hm_data = {}
            for trad in valid_trads:
                idx = df2[df2["tradition_label"] == trad].index
                valid_idx = [i for i in idx if i in concept_df.index]
                if valid_idx:
                    hm_data[trad] = concept_df.loc[valid_idx].mean()
            if hm_data:
                hm_df = pd.DataFrame(hm_data).T
                fig, ax = plt.subplots(figsize=(max(10, len(hm_df.columns)*0.7),
                                                max(5, len(hm_df)*0.6)))
                sns.heatmap(hm_df, annot=True, fmt=".2f", cmap="RdYlBu_r",
                            linewidths=0.3, ax=ax, vmin=0, vmax=hm_df.values.max())
                ax.set_title("Concept Presence by Musical Tradition\n"
                             "(fraction of tradition's passages containing each concept)",
                             fontweight="bold")
                plt.xticks(rotation=40, ha="right", fontsize=8)
                plt.yticks(fontsize=9)
                plt.tight_layout()
                fig.savefig(fp("fig_15_concept_tradition_heatmap"), bbox_inches="tight")
                plt.close(fig)
                print("  ✓ fig_15_concept_tradition_heatmap")

    # Fig 19: Concept exclusivity — never taught alongside marginalized
    if concept_rep is not None and not concept_rep.empty:
        dom_only = concept_rep[concept_rep["n_marg_passages"] == 0].copy()
        dom_only = dom_only[dom_only["n_dom_passages"] >= 3].sort_values("n_dom_passages", ascending=True)
        if not dom_only.empty:
            fig, ax = plt.subplots(figsize=(10, max(4, len(dom_only) * 0.45)))
            ax.barh(dom_only["concept"], dom_only["n_dom_passages"],
                    color=PALETTE["dominant"], edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Number of dominant-composer passages containing this concept")
            ax.set_title("Concept Exclusivity: Concepts Never Taught Alongside Marginalized Composers",
                         fontweight="bold")
            for i, (_, row) in enumerate(dom_only.iterrows()):
                ax.text(row["n_dom_passages"] + 0.3, i,
                        f"{row['overall_freq']:.0%} of all passages",
                        va="center", fontsize=8, color="#555555")
            ax.set_xlim(0, dom_only["n_dom_passages"].max() * 1.35)
            plt.tight_layout()
            fig.savefig(fp("fig_19_concept_exclusivity"))
            plt.close(fig)
            print("  ✓ fig_19_concept_exclusivity")


def plot_external_figures(xl_data: dict, entity_df: pd.DataFrame,
                           metrics: dict, output_dir: str) -> None:
    """Figures from theory_diversity_full_export.xlsx (if loaded)."""
    if not xl_data:
        return
    def fp(name): return _fp(name, output_dir)
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 130
    div = xl_data.get("diversity_summary", pd.DataFrame())
    if div.empty:
        return

    # Fig 20: BIPOC % by book and edition
    if "bipoc_pct" in div.columns and "short_label" in div.columns:
        fig, ax = plt.subplots(figsize=(11, 5))
        div_s = div.sort_values("bipoc_pct", ascending=True)
        ax.barh(div_s["short_label"], div_s["bipoc_pct"] * 100,
                color=PALETTE["marginalized"], edgecolor="white")
        ax.set_xlabel("% BIPOC composer mentions")
        ax.set_title("BIPOC Representation by Textbook (External Dataset)",fontweight="bold")
        plt.tight_layout()
        fig.savefig(fp("fig_20_bipoc_by_book"))
        plt.close(fig)
        print("  ✓ fig_20_bipoc_by_book")

    # Fig 21: Female % by book
    if "female_pct" in div.columns and "short_label" in div.columns:
        fig, ax = plt.subplots(figsize=(11, 5))
        div_s = div.sort_values("female_pct", ascending=True)
        ax.barh(div_s["short_label"], div_s["female_pct"] * 100,
                color=PALETTE["female"], edgecolor="white")
        ax.set_xlabel("% female composer mentions")
        ax.set_title("Female Representation by Textbook (External Dataset)",fontweight="bold")
        plt.tight_layout()
        fig.savefig(fp("fig_21_female_by_book"))
        plt.close(fig)
        print("  ✓ fig_21_female_by_book")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SAVE OUTPUTS                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def save_all_outputs(metrics: dict, entity_df: pd.DataFrame, df: pd.DataFrame,
                     output_dir: str,
                     concept_rep: pd.DataFrame | None = None,
                     concept_df: pd.DataFrame | None = None) -> None:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    metrics["representation_df"].to_csv(out/"01_representation.csv",       index=False)
    metrics["tradition_df"].to_csv(     out/"02_tradition_summary.csv",    index=False)
    metrics["gender_df"].to_csv(        out/"03_gender_summary.csv",       index=False)
    metrics["era_df"].to_csv(           out/"04_era_summary.csv",          index=False)
    if "era_demographic_df" in metrics and not metrics["era_demographic_df"].empty:
        metrics["era_demographic_df"].to_csv(out/"04b_era_demographic.csv", index=False)
    metrics["centrality_df"].to_csv(    out/"05_curriculum_presence.csv",   index=False)
    entity_df.to_csv(                   out/"06_entity_framing_detail.csv",index=False)
    metrics["per_book_summary"].to_csv( out/"07_per_book_summary.csv",     index=False)
    if metrics.get("tfidf_keywords"):
        kw_rows = [{"group":g,"rank":r+1,"term":t,"mean_score":m,"contrast":c}
                   for g,kws in metrics["tfidf_keywords"].items()
                   for r,(t,m,c) in enumerate(kws)]
        pd.DataFrame(kw_rows).to_csv(out/"08_tfidf_keywords.csv",index=False)
    pd.DataFrame(metrics["cooccurrence_edges"],columns=["composer_a","composer_b","weight"])\
      .sort_values("weight",ascending=False).to_csv(out/"09_cooccurrence_edges.csv",index=False)
    if "genre_mentions" in df.columns:
        genre_rows = [{"passage_id":row["passage_id"],"book_id":row["book_id"],
                       "chapter_title":row.get("chapter_title",""),
                       "passage_role":row["passage_role"],"genre":g}
                      for _,row in df.iterrows() for g in row["genre_mentions"]]
        pd.DataFrame(genre_rows).to_csv(out/"10_genre_mentions.csv",index=False)
    if concept_rep is not None and not concept_rep.empty:
        concept_rep.to_csv(out/"11_concept_representation.csv", index=False)
        print(f"  Saved 11_concept_representation.csv ({len(concept_rep)} concepts)")
    if concept_df is not None and not concept_df.empty:
        concept_export = concept_df.copy()
        concept_export.insert(0, "passage_id", df["passage_id"].values)
        concept_export.insert(1, "book_id",    df["book_id"].values)
        concept_export.to_csv(out/"12_concept_passage_matrix.csv", index=False)
        print(f"  Saved 12_concept_passage_matrix.csv ({concept_export.shape})")
    if metrics.get("framing_gap_df") is not None and not metrics["framing_gap_df"].empty:
        metrics["framing_gap_df"].to_csv(out/"13_framing_gap.csv", index=False)
        print(f"  Saved 13_framing_gap.csv")
    if metrics.get("placement_df") is not None and not metrics["placement_df"].empty:
        metrics["placement_df"].to_csv(out/"14_structural_placement.csv", index=False)
        print(f"  Saved 14_structural_placement.csv")
    if metrics.get("integration_df") is not None and not metrics["integration_df"].empty:
        metrics["integration_df"].to_csv(out/"15_integration_index.csv", index=False)
        print(f"  Saved 15_integration_index.csv")

    # ── R / ggplot data exports ───────────────────────────────────────────────
    save_r_data(metrics, entity_df, df, output_dir, concept_rep)


def save_r_data(metrics: dict, entity_df: pd.DataFrame, df: pd.DataFrame,
                output_dir: str,
                concept_rep: pd.DataFrame | None = None) -> None:
    """
    Export tidy long-form CSVs for R/ggplot re-plotting.

    Files written to  <output_dir>/r_data/  named to match figure numbers.
    All files use snake_case column names and factor levels that map cleanly
    to ggplot aesthetics (colour, fill, facet_wrap).

    Figures covered: 1, 2, 3a, 3b, 3c, 6, 7, 8, 10, 10b, 13a, 13b, 14, 16
    """
    out = Path(output_dir) / "r_data"
    out.mkdir(parents=True, exist_ok=True)

    def _save(name: str, frame: pd.DataFrame, msg: str = "") -> None:
        path = out / name
        frame.to_csv(path, index=False)
        print(f"  [r_data] {name}  ({len(frame)} rows{msg})")

    edf = entity_df.drop_duplicates(["passage_id","composer_canonical"]).copy()
    edf["status"] = edf.apply(
        lambda r: "dominant"      if r["dominant"]
             else "marginalized"  if r["marginalized"]
             else "unclassified",
        axis=1
    )
    edf["tradition_label"] = edf["tradition"].map(TRADITION_LABELS).fillna(edf["tradition"])

    # ── Fig 1: Representation by tradition ───────────────────────────────────
    trad = metrics["tradition_df"].copy()
    trad["status"] = trad.apply(
        lambda r: "dominant"      if r["dominant_flag"]
             else "marginalized"  if r.get("marginalized_flag", False)
             else "unclassified", axis=1)
    _save("fig01_tradition.csv", trad[["tradition","tradition_label","total_mentions",
                                        "unique_composers","status"]])

    # ── Fig 2: Gender distribution ────────────────────────────────────────────
    _save("fig02_gender.csv", metrics["gender_df"].rename(columns={"total":"mentions"}))

    # ── Fig 3a: Era — total mentions ──────────────────────────────────────────
    era_ord = ["renaissance","baroque","classical","romantic","modern","contemporary","unknown"]
    era = metrics["era_df"].copy()
    era["era"] = pd.Categorical(era["era"], categories=era_ord, ordered=True)
    _save("fig03a_era_totals.csv", era.sort_values("era"))

    # ── Fig 3b/3c: Era × demographics ────────────────────────────────────────
    if "era_demographic_df" in metrics and not metrics["era_demographic_df"].empty:
        _save("fig03bc_era_demographic.csv", metrics["era_demographic_df"])

    # ── Fig 6: Framing category by dominant / marginalized status ────────────
    framing_status = metrics.get("framing_by_status")
    if framing_status is not None and not framing_status.empty:
        fs = framing_status.reset_index().melt(
            id_vars="framing_category", var_name="status", value_name="mentions")
        fs["status"] = fs["status"].str.lower()
        _save("fig06_framing_by_status.csv", fs)

    # ── Fig 7: Structural placement (top 30 composers) ────────────────────────
    cent = metrics["centrality_df"].head(30).copy()
    cent["status"] = cent.apply(
        lambda r: "dominant"      if r["dominant"]
             else "marginalized"  if r["marginalized"]
             else "unclassified", axis=1)
    keep_cols = ["composer_canonical","mention_count","n_books","n_central",
                 "n_supplementary","n_application","pct_central","book_breadth",
                 "tradition","tradition_label","status","gender"]
    available = [c for c in keep_cols if c in cent.columns]
    _save("fig07_structural_placement.csv", cent[available])

    # ── Fig 8: Co-occurrence edges + node metadata ────────────────────────────
    # R/ggraph plotting: read fig08_cooccurrence_edges.csv and
    # fig08_cooccurrence_nodes.csv from r_data/.  Filter edges to
    # weight >= COOC_THRESHOLD (stored in the 'threshold' column below) and
    # build an igraph object; see Section N8 of Textbook_Representation_Analysis.R.
    edges = pd.DataFrame(metrics["cooccurrence_edges"],
                          columns=["composer_a","composer_b","weight"])
    edges["threshold"] = COOC_THRESHOLD   # threshold used by Python plot; match in R
    _save("fig08_cooccurrence_edges.csv", edges.sort_values("weight", ascending=False))

    # Node metadata: merge composer metadata onto unique node names
    all_nodes = sorted(set(edges["composer_a"]) | set(edges["composer_b"]))
    node_meta = pd.DataFrame({"composer": all_nodes})
    node_meta = node_meta.merge(
        cent[["composer_canonical","status","tradition_label","gender","mention_count"]].rename(
            columns={"composer_canonical":"composer","mention_count":"total_mentions"}),
        on="composer", how="left"
    )
    node_meta["status"] = node_meta["status"].fillna("unclassified")
    _save("fig08_cooccurrence_nodes.csv", node_meta)

    # ── Fig 10: Chapter position by book ─────────────────────────────────────
    pos_cols = ["passage_id","book_id","composer_canonical","chapter_position","status"]
    _save("fig10_chapter_position.csv", edf[pos_cols])

    # ── Fig 10b: Passage role by tradition ────────────────────────────────────
    role_trad = (edf.groupby(["tradition_label","passage_role"])
                    .size().reset_index(name="mentions"))
    _save("fig10b_passage_role_by_tradition.csv", role_trad)

    # ── Fig 13a: Dominant / marginalized % by book ────────────────────────────
    per_book = metrics["per_book_summary"].copy()
    pb13a = per_book[["book_id","pct_dominant","pct_marginalized",
                       "n_mentions","n_unique"]].copy()
    # Convert to long for ggplot
    pb13a_long = pb13a.melt(id_vars=["book_id","n_mentions","n_unique"],
                             value_vars=["pct_dominant","pct_marginalized"],
                             var_name="status", value_name="pct")
    pb13a_long["status"] = pb13a_long["status"].str.replace("pct_","")
    _save("fig13a_representation_by_book.csv", pb13a_long)

    # ── Fig 13b: Framing % by book ────────────────────────────────────────────
    pb13b_long = per_book.melt(id_vars=["book_id"],
                                value_vars=["pct_normative","pct_additive","pct_neutral"],
                                var_name="framing", value_name="pct")
    pb13b_long["framing"] = pb13b_long["framing"].str.replace("pct_","")
    _save("fig13b_framing_by_book.csv", pb13b_long)

    # ── Fig 14: Concept contrast ──────────────────────────────────────────────
    if concept_rep is not None and not concept_rep.empty:
        _save("fig14_concept_contrast.csv", concept_rep)

    # ── Fig 16: Framing gap by book ───────────────────────────────────────────
    gap_df = metrics.get("framing_gap_df")
    if gap_df is not None and not gap_df.empty:
        # Long-form: one row per (book × composer_status × framing_type)
        rows = []
        for _, r in gap_df.iterrows():
            for framing, dom_col, marg_col in [
                ("normative",  "dom_normative",  "marg_normative"),
                ("additive",   "dom_additive",   "marg_additive"),
                ("exceptional","dom_exceptional","marg_exceptional"),
            ]:
                rows.append({"book_id":r["book_id"],"framing":framing,
                              "status":"dominant",    "pct": r[dom_col]*100})
                rows.append({"book_id":r["book_id"],"framing":framing,
                              "status":"marginalized","pct": r[marg_col]*100})
        _save("fig16_framing_gap.csv", pd.DataFrame(rows))

    print(f"  [r_data] All files written → {out}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    args = parse_args()

    use_bert      = not args.no_bert
    use_zero_shot = not args.no_zero_shot
    bert_threshold = args.bert_upgrade_threshold
    zs_threshold   = args.zero_shot_threshold
    output_dir     = args.output_dir
    resolved_device = _resolve_device(BERT_DEVICE)

    # Override module-level flag from CLI arg
    global TFIDF_KEYWORDS_ENABLED
    TFIDF_KEYWORDS_ENABLED = args.tfidf_keywords

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Music Theory Representation Analysis                    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  data-dir       : {args.data_dir}")
    print(f"  output-dir     : {output_dir}")
    print(f"  debug mode     : {args.debug} (n={args.sample_n})")
    print(f"  BERT model     : {BERT_MODEL}  (enabled={use_bert})")
    print(f"  BERT threshold : {bert_threshold}")
    print(f"  LLM classifier : {LLM_CLASSIFIER_MODEL}  (enabled={use_zero_shot})")
    print(f"    precision    : {'4-bit NF4' if LLM_CLASSIFIER_LOAD_4BIT else 'fp16 (~17.3 GB VRAM with BGE)'}")
    print(f"  LLM threshold  : {zs_threshold}")
    print(f"  TF-IDF keywords: {'enabled (--tfidf-keywords)' if TFIDF_KEYWORDS_ENABLED else 'disabled (use --tfidf-keywords to enable)'}")
    print(f"  device         : {resolved_device}")
    print()

    # ── § 1  Load auxiliary data ──────────────────────────────────────────────
    concept_patterns = load_concept_patterns(args.concept_csv)
    concept_df = pd.DataFrame()
    xl_data = load_external_diversity_data(args.excel_data, EXCEL_BOOK_MAP)

    # ── § 2  Name normalisation + biographical merge ──────────────────────────
    _tests = [("Debussy\u2019s","Claude Debussy"),("Debussy's","Claude Debussy"),
              ("Debussy","Claude Debussy"),("J.S. Bach","Johann Sebastian Bach"),
              ("Schumann","Robert Schumann"),("Debussy, Préludes, Book II","Claude Debussy")]
    _ok = sum(1 for r,e in _tests if resolve_composer(r)==e)
    print(f"Name normalisation: {_ok}/{len(_tests)} self-tests passed.")

    bio_lookup = load_biographical_data(args.bio_data, resolve_composer)
    if bio_lookup:
        merge_biographical_data(bio_lookup)
        print(f"  COMPOSER_METADATA: {len(COMPOSER_METADATA)} entries after bio merge.")

    print(f"Metadata loaded: {len(COMPOSER_METADATA)} composers, "
          f"{len(COMPOSER_ALIASES)} aliases, {len(FRAMING_LEXICONS)} framing categories.")

    # ── § 3  Load + preprocess data ───────────────────────────────────────────
    df = load_data(DATA_FILES, data_dir=args.data_dir,
                   debug=args.debug, sample_n=args.sample_n)
    df = preprocess_text(df)
    df = detect_genre_mentions(df)
    df, concept_df = detect_concepts_in_passages(df, concept_patterns)

    # ── § 4  Extract entities ─────────────────────────────────────────────────
    entity_df = extract_entities(df)

    # ── § 5  TF-IDF + rule framing classifier (backbone) ─────────────────────
    tfidf_vec, passage_matrix, proto_centroids = build_tfidf_model(entity_df, concept_patterns)
    entity_df = classify_framing_backbone(entity_df, passage_matrix, proto_centroids)

    # ── § 6  BERT upgrade pass ────────────────────────────────────────────────
    if use_bert:
        entity_df = run_bert_pass(
            entity_df,
            bert_model_name=BERT_MODEL,
            device=resolved_device,
            threshold=bert_threshold,
            batch_size=BERT_BATCH_SIZE,
            n_clusters=N_CLUSTERS,
        )
    else:
        entity_df["bert_cluster"]  = -1
        entity_df["bert_best_cat"] = ""
        entity_df["bert_best_sim"] = 0.0
        print("[BERT] Skipped (--no-bert).")

    # ── § 7  Zero-shot upgrade pass ───────────────────────────────────────────
    if use_zero_shot:
        entity_df = run_zero_shot_pass(
            entity_df,
            model_name=ZERO_SHOT_MODEL,
            device=resolved_device,
            threshold=zs_threshold,
            batch_size=ZERO_SHOT_BATCH,
        )
    else:
        print("[ZeroShot] Skipped (--no-zero-shot).")

    # ── § 8  Compute metrics ──────────────────────────────────────────────────
    metrics = compute_metrics(df, entity_df, concept_patterns)
    representation_df = metrics["representation_df"]
    centrality_df     = metrics["centrality_df"]

    print("\nTop 10 by centrality:")
    print(centrality_df.head(10)[["composer_canonical","mention_count","n_books","pct_central","tradition_label","dominant"]].to_string(index=False))

    concept_rep = compute_concept_representation(df, entity_df, concept_df)

    # ── § 9  Visualise ────────────────────────────────────────────────────────
    visualize_results(metrics, entity_df, df, output_dir)
    plot_concept_representation(concept_rep, concept_df, entity_df, df, output_dir)
    plot_external_figures(xl_data, entity_df, metrics, output_dir)

    # ── § 10  Save CSVs ───────────────────────────────────────────────────────
    save_all_outputs(metrics, entity_df, df, output_dir, concept_rep, concept_df)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = metrics["representation_df"]["mention_count"].sum()
    dom   = metrics["representation_df"].loc[metrics["representation_df"]["dominant"],
                                              "mention_count"].sum()
    marg  = metrics["representation_df"].loc[metrics["representation_df"]["marginalized"],
                                              "mention_count"].sum()
    n_composers = len(metrics["representation_df"])
    bert_lbl = f"BERT={BERT_MODEL}" if use_bert else "BERT=off"
    zs_lbl   = f"ZeroShot={ZERO_SHOT_MODEL}" if use_zero_shot else "ZeroShot=off"

    print("\n" + "═"*66)
    print("  MUSIC THEORY REPRESENTATION ANALYSIS — SUMMARY")
    print("═"*66)
    print(f"  Passages      : {len(df):,}  ({'DEBUG' if args.debug else 'full corpus'})")
    print(f"  Books         : {df['book_id'].nunique()}")
    print(f"  Mentions      : {int(total):,}")
    print(f"  Unique comps  : {n_composers:,}")
    print(f"  Dominant      : {int(dom):,}  ({dom/total:.1%})")
    print(f"  Marginalized  : {int(marg):,}  ({marg/total:.1%})")
    print(f"  Classifiers   : TF-IDF+rules → {bert_lbl} → {zs_lbl}")
    print("\n  Framing distribution:")
    for cat, n in entity_df["framing_category"].value_counts().items():
        n_bert = (entity_df.loc[entity_df["framing_category"] == cat,
                                 "framing_source"] == "bert_upgrade").sum() \
                 if "framing_source" in entity_df.columns else 0
        n_zs   = (entity_df.loc[entity_df["framing_category"] == cat,
                                 "framing_source"] == "zero_shot").sum() \
                 if "framing_source" in entity_df.columns else 0
        src = ""
        if n_bert: src += f"  ({n_bert} BERT upgrades)"
        if n_zs:   src += f"  ({n_zs} zero-shot upgrades)"
        print(f"    {cat:15s} {n:4d}{src}")
    print(f"\n  Outputs: {Path(output_dir).resolve()}")
    print("═"*66)


if __name__ == "__main__":
    main()
