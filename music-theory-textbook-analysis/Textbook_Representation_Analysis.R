# =============================================================================
# Music Theory Textbook Diversity Analysis
# =============================================================================
#
# Section map
#   A  Libraries
#   B  Parameters & paths
#   C  Color palette & theme
#   D  Load & clean data
#   E  Normalize composer names
#   F  Parse & recode demographic fields
#   G  "The Boys" canon flag
#   H  Short textbook labels
#   I  Summary helper
#    1  Core diversity metrics
#    2  Tokenization metrics (BIPOC / Female)
#    3  Identity of "token" composers
#    4  Multi-edition subsets
#    5  Ordered x-axis levels & zero-fill
#    6  Stacked bar plots — Race & Gender
#    7  Multi-edition line plots — Race & Gender
#    8  "The Boys" stacked bar
#    9  Tokenization plots
#   10  Diversity dashboard summary table
#   11  Geographic metrics & visualizations
#   12  Print & display
#   13  Data exports
#
# =============================================================================


# =============================================================================
# A. Libraries
# =============================================================================

library(tidyverse)
library(readxl)
library(janitor)
library(DBI)         # database interface         (Section N15 — reads textrep.db)
library(RSQLite)     # SQLite driver for textrep.db (Section N15)
library(igraph)      # graph construction for co-occurrence network (Section N8)
library(ggraph)      # ggplot2-based network visualisation  (Section N8)
# igraph exports diversity() which conflicts with vegan::diversity().
# Load igraph BEFORE vegan so vegan's version wins the namespace.
library(vegan)       # Shannon / Simpson diversity indices
library(scales)      # percent_format()
library(maps)        # map_data("world")
library(geosphere)   # distHaversine()
library(ggrepel)     # non-overlapping labels

# Uncomment for choropleth maps (Section 11d):
# library(sf)
# library(rnaturalearth)
# library(rnaturalearthdata)

# writexl is loaded conditionally in Section 13.


# =============================================================================
# B. Parameters & paths
# =============================================================================

TOP_N   <- 5   # composers included in the "top-N share" metric (Section 1)
TOKEN_N <- 3   # top-N composers examined within BIPOC / Female subgroups (Section 2)

DATA_PATH   <- "composers_pieces_index.xlsx"
GEO_PATH    <- "composers_countries.csv"
RESULTS_DIR <- "analysis_results"
PLOTS_DIR   <- "analysis_plots"

SAVE_PLOTS   <- TRUE   # write PNGs to PLOTS_DIR  (Section 12)
SAVE_RESULTS <- TRUE   # write CSVs / Excel to RESULTS_DIR (Section 13)

dir.create(RESULTS_DIR, showWarnings = FALSE)
dir.create(PLOTS_DIR,   showWarnings = FALSE)

# ---- NLP pipeline output (set this to wherever the Python script writes) ----
#
# PY_RESULTS_DIR  — output folder from analysis_server.py
#                   (NOT the same as RESULTS_DIR, which is where *this* script
#                   writes its own CSVs and plots)
# NLP_DATA_DIR    — the r_data/ sub-folder inside PY_RESULTS_DIR (Sections N*)

PY_RESULTS_DIR <- "results"
NLP_DATA_DIR   <- file.path(PY_RESULTS_DIR, "r_data")

# textrep.db — the assembled SQLite database (build_db.py output).
# Section N15 reads framing/placement data directly from this DB rather than
# the r_data CSVs, so the statistical-support plots stay in lock-step with
# framing_analysis.py and the article's Results section.
DB_PATH <- file.path(PY_RESULTS_DIR, "textrep.db")

NLP_PLOTS_DIR  <- file.path(PLOTS_DIR, "nlp")
dir.create(NLP_PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)


# =============================================================================
# C. Color palette & theme
# =============================================================================

# ---- Base colors ------------------------------------------------------------
COL_BLUE       <- "#13294B"
COL_LIGHT_BLUE <- "#4B9CD3"
COL_MID_BLUE   <- "#306998"
COL_GRAY       <- "#A7A8AA"
COL_DARK_GRAY  <- "#58595B"
COL_AMBER      <- "#E87722"   # tokenization accent

# ---- Demographic palettes ---------------------------------------------------
race_palette <- c(
  "White"   = COL_GRAY,
  "BIPOC"   = COL_BLUE,
  "Unknown" = COL_DARK_GRAY
)

gender_palette <- c(
  "M"       = COL_BLUE,
  "F"       = COL_LIGHT_BLUE,
  "Unknown" = COL_GRAY
)

# ---- "The Boys" palette: dark → light (Bach, Haydn, Mozart, Beethoven) ------
boys_palette <- c(
  "Bach"      = COL_BLUE,
  "Haydn"     = COL_MID_BLUE,
  "Mozart"    = COL_LIGHT_BLUE,
  "Beethoven" = COL_GRAY
)

# ---- Continental palette ----------------------------------------------------
continent_palette <- c(
  "Europe"        = COL_BLUE,
  "North America" = COL_LIGHT_BLUE,
  "Asia"          = COL_AMBER,
  "South America" = "#2ca02c",
  "Africa"        = "#d62728",
  "Oceania"       = "#9467bd",
  "Unknown"       = COL_GRAY
)

# ---- ggplot theme -----------------------------------------------------------
theme_clean <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      text             = element_text(color = COL_DARK_GRAY),
      plot.title       = element_text(face = "bold", color = COL_BLUE,
                                      size = base_size + 2),
      plot.subtitle    = element_text(color = COL_DARK_GRAY, size = base_size - 1),
      plot.caption     = element_text(color = COL_GRAY, size = base_size - 2,
                                      hjust = 0),
      axis.title       = element_text(size = base_size - 1),
      legend.title     = element_blank(),
      panel.grid.minor = element_blank(),
      strip.text       = element_text(face = "bold", color = COL_BLUE)
    )
}


# =============================================================================
# D. Load & clean data
# =============================================================================

df_raw <- read_excel(DATA_PATH) %>% clean_names()

# Column inventory (for reference):
#   textbook   — "Laitz, 4th ed. (2016)", etc.
#   composer   — raw composer name as entered in the workbook
#   born, died — integers or "NA" strings
#   sex        — "M", "F", "—" (anonymous), "M/F" (group), NA
#   bipoc      — "Y", "N", "Y/N" (mixed group), NA
#   piece_name — work title

geo_raw <- read_csv(GEO_PATH) %>% clean_names()

# ---- Country → Continent lookup ---------------------------------------------
country_to_continent <- function(country) {
  case_when(
    country %in% c(
      "Austria", "Germany", "France", "Italy", "Spain", "United Kingdom",
      "Netherlands", "Belgium", "Poland", "Czech Republic", "Hungary",
      "Romania", "Russia", "Finland", "Norway", "Sweden", "Denmark",
      "Holy Roman Empire", "Roman Empire"       # historical polities
    ) ~ "Europe",
    country %in% c("United States", "Canada", "Mexico") ~ "North America",
    country %in% c(
      "Brazil", "Argentina", "Chile", "Peru", "Colombia", "Venezuela"
    ) ~ "South America",
    country %in% c(
      "China", "Japan", "South Korea", "India", "Iran", "Turkey"
    ) ~ "Asia",
    country %in% c(
      "Nigeria", "South Africa", "Egypt", "Ghana", "Ethiopia"
    ) ~ "Africa",
    country %in% c("Australia", "New Zealand") ~ "Oceania",
    TRUE ~ "Unknown"
  )
}

geo_lookup <- geo_raw %>%
  mutate(continent = country_to_continent(country)) %>%
  select(composer_variant = composer, country, continent, latitude, longitude)
# geo_lookup is keyed on the *variant* composer column (confirmed unique in the CSV).
# This is the correct join target: the CSV was built so that each textbook name
# variant appears as its own row, all pointing to the same coordinates.
# Joining on matched_name (the canonical form) was wrong because composer_clean
# after normalization is still a variant, not always the canonical form.
#
# Two-pass join strategy:
#   Pass 1 — composer_clean (post-normalization) vs. composer_variant
#   Pass 2 — raw composer (pre-normalization) vs. composer_variant, for rows
#             that normalization moved away from the geo key (e.g. Bologne
#             variants → canonical long form not in CSV)
# This captures ~99% of geo-codeable composers without requiring the CSV to be
# re-keyed to match every possible normalized form.


# =============================================================================
# E. Normalize composer names
# =============================================================================
#
# The raw data contains multiple name variants for the same composer.
# Strategy:
#   1. Collapse all "Anon*" variants to "Anonymous".
#   2. Strip parenthetical glosses and "a.k.a." clauses.
#   3. Apply a manual lookup table for the most common cross-textbook variants.
#
# Extend composer_lookup as new variants are discovered in the data.

composer_lookup <- c(
  # Florence Price
  "Florence Price"                                  = "Florence B. Price",
  # Joseph Bologne (many spelling variants)
  "Joseph Bologne"                                  = "Joseph Bologne, Chev. de Saint-Georges",
  "Joseph Bologne Chevalier de Saint-Georges"       = "Joseph Bologne, Chev. de Saint-Georges",
  "Joseph Boulogne Chevalier de Saint-George"       = "Joseph Bologne, Chev. de Saint-Georges",
  "Joseph Bologne Chevalier de Saint\u2011Georges"  = "Joseph Bologne, Chev. de Saint-Georges",
  "Joseph Boulogne, Chevalier de Saint-Georges"     = "Joseph Bologne, Chev. de Saint-Georges",
  # Clara Schumann
  "Clara Schumann"                                  = "Clara Wieck Schumann",
  # Harry Burleigh
  "Harry Thacker Burleigh"                          = "Harry T. Burleigh",
  # Fanny Hensel
  "Fanny Hensel"                                    = "Fanny Mendelssohn Hensel",
  # W.C. Handy
  "W. C. Handy"                                     = "W.C. Handy",
  "William Christopher Handy"                       = "W.C. Handy",
  # Lin-Manuel Miranda (non-breaking hyphen variant)
  "Lin\u2011Manuel Miranda"                         = "Lin-Manuel Miranda",
  # Pauline Viardot
  "Pauline Viardot\u2011Garc\u00eda"               = "Pauline Viardot-García",
  # Nathaniel Dett
  "Nathaniel Dett"                                  = "R. Nathaniel Dett",
  # Toru Takemitsu
  "Toru Takemitsu"                                  = "T\u014dru Takemitsu",
  # Elisabeth Jacquet de la Guerre (capitalization variants)
  "\u00c9lisabeth Jacquet de La Guerre"             = "\u00c9lisabeth Jacquet de la Guerre",
  "Elisabeth-Claude Jacquet de la Guerre"           = "\u00c9lisabeth Jacquet de la Guerre",
  "Elizabeth Jacquet de la Guerre"                  = "\u00c9lisabeth Jacquet de la Guerre",
  "Elizabeth Jacquet de La Guerre"                  = "\u00c9lisabeth Jacquet de la Guerre",
  # Maria Szymanowska
  "Maria Agata Szymanowska"                         = "Maria Szymanowska",
  "Maria Wolowska Szymanowska"                      = "Maria Szymanowska",
  "Maria Wo\u0142owska Szymanowska"                = "Maria Szymanowska",
  # Beyoncé
  "Beyonc\u00e9 Knowles"                           = "Beyonc\u00e9",
  # James P. Johnson
  "James P. Johnson"                                = "James Price Johnson",
  # W.C. Handy duplicate
  "William Christopher Handy"                       = "W.C. Handy",
  # Duke Ellington
  "Edward Kennedy Ellington"                        = "Duke Ellington"
)

normalize_composer <- function(name) {
  name <- str_trim(name)
  # Collapse all anonymous variants to a single token
  name <- if_else(
    str_detect(name, regex("^anon|anonymous", ignore_case = TRUE)),
    "Anonymous", name
  )
  # Strip parenthetical glosses and "a.k.a." clauses
  name <- str_remove(name, "\\s*\\(.*?\\)")
  name <- str_remove(name, "(?i)\\s*a\\.k\\.a\\..*")
  name <- str_trim(name)
  # Apply manual lookup (returns NA if name not in table → keep original)
  if_else(!is.na(composer_lookup[name]), composer_lookup[name], name)
}


# =============================================================================
# F. Parse & recode demographic fields
# =============================================================================

# ---- Ordinal suffix helper (used for short labels) --------------------------
ordinal_suffix <- function(n) {
  case_when(
    n %% 100 %in% 11:13 ~ "th",   # 11th, 12th, 13th
    n %% 10  == 1       ~ "st",
    n %% 10  == 2       ~ "nd",
    n %% 10  == 3       ~ "rd",
    TRUE                ~ "th"
  )
}

# ---- Parse textbook metadata from the Textbook string -----------------------
# Expected format: "Author(s), Nth ed. (YYYY)"  or  "Author(s) (YYYY)"
# Single-edition books (Gotham, Hutchinson, Mount) have no edition clause
#   → edition defaults to 1.

df <- df_raw %>%
  mutate(
    composer_clean = normalize_composer(composer),

    # Textbook identity
    textbook_name = str_extract(textbook, "^[^(,]+") %>% str_trim(),
    year          = str_extract(textbook, "\\((\\d{4})\\)") %>%
                      str_remove_all("[()]") %>% as.integer(),
    edition       = str_extract(textbook, "\\d+(?=(?:st|nd|rd|th) ed)") %>%
                      as.integer() %>% replace_na(1L),

    # Race
    # "Y/N" (mixed-race group) is treated as BIPOC — the group contains
    # BIPOC members, so excluding them would undercount representation.
    race = case_when(
      bipoc %in% c("Y", "Y/N") ~ "BIPOC",
      bipoc == "N"              ~ "White",
      TRUE                      ~ "Unknown"   # NA = anonymous / not coded
    ),

    # Gender
    # "—", "M/F" (group), and NA are all treated as Unknown.
    gender = case_when(
      sex == "M" ~ "M",
      sex == "F" ~ "F",
      TRUE       ~ "Unknown"
    )
  )

# ---- Join geographic coordinates (two-pass) ---------------------------------
# Pass 1: join on normalized name
df <- df %>%
  left_join(geo_lookup, by = c("composer_clean" = "composer_variant"))

# Pass 2: for rows still missing geo data, try the raw pre-normalization name
geo_fallback <- geo_lookup %>%
  rename(country2 = country, continent2 = continent,
         latitude2 = latitude, longitude2 = longitude)

df <- df %>%
  left_join(geo_fallback, by = c("composer" = "composer_variant")) %>%
  mutate(
    country   = coalesce(country,   country2),
    continent = coalesce(continent, continent2),
    latitude  = coalesce(latitude,  latitude2),
    longitude = coalesce(longitude, longitude2)
  ) %>%
  select(-country2, -continent2, -latitude2, -longitude2) %>%
  mutate(continent = replace_na(continent, "Unknown"))


# =============================================================================
# G. "The Boys" canon flag
# =============================================================================
#
# "The Boys" = J.S. Bach, Haydn, Mozart, Beethoven — the four composers who
# dominate the Common Practice canon in most Anglo-American theory textbooks.
# C.P.E. Bach and other Bach family members are intentionally excluded.
#
# Matching uses composer_clean (post-normalization) to handle initials and
# middle-name variants robustly.

js_bach_pattern <- regex(
  "^(j\\.?\\s*s\\.?\\s*bach|johann sebastian bach|bach,?\\s*j\\.?s\\.?)$",
  ignore_case = TRUE
)

df <- df %>%
  mutate(
    is_boy = case_when(
      str_detect(composer_clean, js_bach_pattern)                                          ~ TRUE,
      str_detect(composer_clean, regex("^haydn$|joseph haydn",       ignore_case = TRUE)) ~ TRUE,
      str_detect(composer_clean, regex("^mozart$|wolfgang.*mozart",  ignore_case = TRUE)) ~ TRUE,
      str_detect(composer_clean, regex("^beethoven$|ludwig.*beethoven", ignore_case = TRUE)) ~ TRUE,
      TRUE ~ FALSE
    ),
    boy_label = case_when(
      str_detect(composer_clean, js_bach_pattern)                                            ~ "Bach",
      str_detect(composer_clean, regex("^haydn$|joseph haydn",       ignore_case = TRUE))   ~ "Haydn",
      str_detect(composer_clean, regex("^mozart$|wolfgang.*mozart",  ignore_case = TRUE))   ~ "Mozart",
      str_detect(composer_clean, regex("^beethoven$|ludwig.*beethoven", ignore_case = TRUE)) ~ "Beethoven",
      TRUE ~ NA_character_
    )
  )


# =============================================================================
# H. Short textbook labels
# =============================================================================
#
# Short labels are used wherever full textbook names would be too wide:
#   scatter plot annotations, multi-edition x-axes, map legends.
#
# Format: "Laitz, 3rd" / "Clendenning, 4th" / "Gotham" / "Hutchinson"
# (first surname only; ordinal edition appended when edition > 1)

make_short_label <- function(textbook_name, edition) {
  first_author <- word(textbook_name, 1) %>%
    str_remove("\\s+et$")              # "Gotham et al." → "Gotham"
  if_else(
    !is.na(edition) & edition > 1,
    paste0(first_author, ", ", edition, ordinal_suffix(edition)),
    first_author
  )
}

df <- df %>%
  mutate(
    short_label  = make_short_label(textbook_name, edition),
    # x-axis label: short name + year on second line
    textbook_id  = paste0(short_label, "\n(", year, ")")
  )


# =============================================================================
# I. Summary helper
# =============================================================================
#
# per_book() groups by all textbook-identity columns so that summarise()
# retains them without needing to re-join afterwards.

per_book <- function(data, ...) {
  data %>%
    group_by(
      textbook_name, year, edition, textbook_id, short_label,
      ...
    )
}

# NOTE: edition_fct is intentionally omitted from per_book() grouping.
# Joining factor columns with potentially differing level sets causes silent
# mismatches in dplyr. Downstream join keys use the integer `edition` instead.


# =============================================================================
# 1. Core Diversity Metrics
# =============================================================================

# NOTE: Pielou's J' is the sole corpus-wide concentration/evenness metric,
# positioned as this study's methodological contribution beyond Ewell 2020.

diversity_metrics <- df %>%
  per_book() %>%
  summarise(
    n_examples  = n(),
    n_composers = n_distinct(composer_clean),
    boys_n      = sum(is_boy, na.rm = TRUE),
    boys_pct    = boys_n / n_examples,
    bipoc_n     = sum(race == "BIPOC"),
    bipoc_pct   = bipoc_n / n_examples,
    female_n    = sum(gender == "F"),
    female_pct  = female_n / n_examples,
    shannon     = vegan::diversity(table(composer_clean), index = "shannon"),
    simpson     = vegan::diversity(table(composer_clean), index = "simpson"),
    .groups     = "drop"
  ) %>%
  mutate(pielou_evenness = shannon / log(n_composers))

# ---- Top-N composer share ---------------------------------------------------
top_n_share <- df %>%
  per_book() %>%
  count(composer_clean, name = "n") %>%
  arrange(desc(n), .by_group = TRUE) %>%
  mutate(rank = row_number()) %>%
  summarise(
    top_n_share = sum(n[rank <= TOP_N]) / sum(n),
    .groups     = "drop"
  )

diversity_metrics <- diversity_metrics %>%
  left_join(top_n_share,
            by = c("textbook_name", "year", "edition", "textbook_id", "short_label"))


# =============================================================================
# 2. Tokenization Metrics
# =============================================================================
#
# When a textbook includes BIPOC or female composers, does it lean on a tiny
# handful (tokenization) or spread credit across many?
#
# Metrics computed within each underrepresented subgroup per textbook × edition:
#
#   top1_share      % of subgroup appearances from the single most-cited composer.
#   top_n_share     % from the top TOKEN_N composers.
#   within_shannon  Shannon entropy within subgroup (higher = more spread).
#   within_evenness Pielou's J within subgroup.
#   breadth_ratio   unique composers ÷ total appearances.
#   singleton_pct   % of subgroup composers appearing only once.
#
# Textbooks with zero members in a subgroup receive NA for ratio metrics and
# 0 for count metrics after the zero-fill step in Section 5.

tokenization_metrics <- function(data, group_filter, group_label) {
  data %>%
    filter({{ group_filter }}) %>%
    per_book() %>%
    add_count(name = "subgroup_n") %>%
    count(composer_clean, subgroup_n, name = "composer_n") %>%
    summarise(
      subgroup_n     = first(subgroup_n),
      n_unique       = n(),
      top1_share     = max(composer_n)  / first(subgroup_n),
      top_n_share    = sum(sort(composer_n, decreasing = TRUE)[seq_len(min(TOKEN_N, n()))]) /
                         first(subgroup_n),
      within_shannon = -sum((composer_n / first(subgroup_n)) *
                              log(composer_n / first(subgroup_n))),
      breadth_ratio  = n() / first(subgroup_n),
      singleton_pct  = sum(composer_n == 1) / n(),
      .groups        = "drop"
    ) %>%
    mutate(
      within_evenness = within_shannon / log(n_unique),
      group           = group_label
    )
}

bipoc_token  <- tokenization_metrics(df, race   == "BIPOC", "BIPOC")
female_token <- tokenization_metrics(df, gender == "F",     "Female")

token_df <- bind_rows(bipoc_token, female_token)


# =============================================================================
# 3. Identity of "Token" Composers
# =============================================================================
#
# Top TOKEN_N composers within BIPOC / Female per textbook × edition.
# Used in Section 9 dot plots to name the individuals carrying the load.

top_within_group <- function(data, group_filter, group_label, top_n = TOKEN_N) {
  data %>%
    filter({{ group_filter }}) %>%
    per_book() %>%
    count(composer_clean, name = "n") %>%
    arrange(desc(n), composer_clean, .by_group = TRUE) %>%
    mutate(rank = row_number()) %>%
    filter(rank <= top_n) %>%
    mutate(
      total = sum(n),
      pct   = n / total,
      group = group_label
    ) %>%
    ungroup()
}

top_bipoc  <- top_within_group(df, race   == "BIPOC", "BIPOC")
top_female <- top_within_group(df, gender == "F",     "Female")


# =============================================================================
# 4. Multi-Edition Subsets
# =============================================================================

multi_ed_books <- df %>%
  group_by(textbook_name) %>%
  filter(n_distinct(edition) > 1) %>%
  pull(textbook_name) %>%
  unique()

multi_ed_df    <- df       %>% filter(textbook_name %in% multi_ed_books)
multi_ed_token <- token_df %>% filter(textbook_name %in% multi_ed_books)


# =============================================================================
# 5. Ordered X-Axis Levels & Zero-Fill
# =============================================================================

# ---- Chronological x-axis order --------------------------------------------
book_order <- df %>%
  distinct(textbook_id, year) %>%
  arrange(year) %>%
  pull(textbook_id)

df                <- df                %>% mutate(textbook_id = factor(textbook_id, levels = book_order))
diversity_metrics <- diversity_metrics %>% mutate(textbook_id = factor(textbook_id, levels = book_order))

# ---- Divider between pre-2020 and post-2020 books ---------------------------
pre2020_books  <- df %>% filter(year <  2020) %>% pull(textbook_id) %>% levels() %>%
                    intersect(as.character(unique(df$textbook_id[df$year < 2020])))
post2020_books <- df %>% filter(year >= 2020) %>% pull(textbook_id) %>% unique()

divider_pos <- if (length(pre2020_books) > 0 && length(post2020_books) > 0) {
  max(which(levels(df$textbook_id) %in% as.character(pre2020_books))) + 0.5
} else {
  NULL
}

# ---- Zero-fill token_df so every textbook × group row exists ----------------
#
# Textbooks with zero BIPOC or zero female composers (e.g. Aldwell & Schachter)
# produce no rows in bipoc_token / female_token, and therefore disappear from
# all tokenization plots.  We complete the grid here so those books appear
# explicitly as 0, making the absence of representation visible rather than
# ambiguous.
#
# Ratio metrics (top1_share, breadth_ratio, etc.) remain NA for zero-member
# subgroups — they are mathematically undefined and plotted accordingly.

all_book_keys <- df %>%
  distinct(textbook_name, year, edition, textbook_id, short_label)

token_df <- bind_rows(bipoc_token, female_token) %>%
  complete(
    nesting(textbook_name, year, edition, textbook_id, short_label),
    group,
    fill = list(
      subgroup_n     = 0L,
      n_unique       = 0L,
      top1_share     = 0,
      top_n_share    = 0,
      within_shannon = NA_real_,
      within_evenness = NA_real_,
      breadth_ratio  = NA_real_,
      singleton_pct  = NA_real_
    )
  ) %>%
  # Ensure every book appears, not just those already in token_df
  right_join(
    tidyr::crossing(all_book_keys, tibble(group = c("BIPOC", "Female"))),
    by = c("textbook_name", "year", "edition", "textbook_id", "short_label", "group")
  ) %>%
  replace_na(list(subgroup_n = 0L, n_unique = 0L, top1_share = 0, top_n_share = 0)) %>%
  mutate(textbook_id = factor(textbook_id, levels = book_order))

# Rebuild multi-edition token subset after zero-fill
multi_ed_token <- token_df %>% filter(textbook_name %in% multi_ed_books)


# =============================================================================
# 6. Stacked Bar Plots — Race & Gender
# =============================================================================
#
# Two views for each demographic dimension:
#   (a) Examples   — weighted by how often each composer is cited
#   (b) Composers  — each composer counted once per textbook × edition
#
# The composer view shows roster breadth; the example view shows page time.
# A gap between the two signals tokenization (many names, little space).

# ---- Data builders ----------------------------------------------------------
stacked_bar_examples <- function(data, dem_col) {
  data %>%
    per_book({{ dem_col }}) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(textbook_name, year, edition, textbook_id, short_label) %>%
    mutate(pct = n / sum(n)) %>%
    ungroup() %>%
    mutate(textbook_id = factor(textbook_id, levels = book_order))
}

stacked_bar_composers <- function(data, dem_col) {
  data %>%
    distinct(textbook_name, year, edition, textbook_id, short_label,
             composer_clean, {{ dem_col }}) %>%
    count(textbook_name, year, edition, textbook_id, short_label,
          {{ dem_col }}, name = "n") %>%
    group_by(textbook_name, year, edition, textbook_id, short_label) %>%
    mutate(pct = n / sum(n)) %>%
    ungroup() %>%
    mutate(textbook_id = factor(textbook_id, levels = book_order))
}

# ---- Plot builder -----------------------------------------------------------
stacked_bar_plot <- function(data, fill_col, palette, title, subtitle, y_label) {
  ggplot(data, aes(x = textbook_id, y = pct, fill = {{ fill_col }})) +
    geom_col(width = 0.75) +
    { if (!is.null(divider_pos))
        geom_vline(xintercept = divider_pos, linetype = "dashed",
                   color = COL_DARK_GRAY, linewidth = 0.7)
    } +
    scale_fill_manual(values = palette) +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       expand = expansion(mult = c(0, .02))) +
    labs(title = title, subtitle = subtitle, x = NULL, y = y_label) +
    theme_clean() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))
}

# ---- Build data & plots -----------------------------------------------------
race_bar_ex    <- stacked_bar_examples(df,  race)
race_bar_comp  <- stacked_bar_composers(df, race)
gender_bar_ex   <- stacked_bar_examples(df,  gender)
gender_bar_comp <- stacked_bar_composers(df, gender)

plot_race_bar_ex <- stacked_bar_plot(
  race_bar_ex, race, race_palette,
  "Racial Representation by Textbook — Share of Examples",
  "Each example weighted equally; dashed line = 2020",
  "Share of examples"
)
plot_race_bar_comp <- stacked_bar_plot(
  race_bar_comp, race, race_palette,
  "Racial Representation by Textbook — Share of Composers",
  "Each composer counted once per book; dashed line = 2020",
  "Share of unique composers"
)
plot_gender_bar_ex <- stacked_bar_plot(
  gender_bar_ex, gender, gender_palette,
  "Gender Representation by Textbook — Share of Examples",
  "Each example weighted equally; dashed line = 2020",
  "Share of examples"
)
plot_gender_bar_comp <- stacked_bar_plot(
  gender_bar_comp, gender, gender_palette,
  "Gender Representation by Textbook — Share of Composers",
  "Each composer counted once per book; dashed line = 2020",
  "Share of unique composers"
)


# =============================================================================
# 7. Multi-Edition Line Plots — Race & Gender
# =============================================================================

multi_line_examples <- function(data, dem_col) {
  data %>%
    per_book({{ dem_col }}) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(textbook_name, year, edition, short_label) %>%
    mutate(pct = n / sum(n)) %>%
    ungroup()
}

multi_line_composers <- function(data, dem_col) {
  data %>%
    distinct(textbook_name, year, edition, textbook_id, short_label,
             composer_clean, {{ dem_col }}) %>%
    count(textbook_name, year, edition, textbook_id, short_label,
          {{ dem_col }}, name = "n") %>%
    group_by(textbook_name, year, edition, short_label) %>%
    mutate(pct = n / sum(n)) %>%
    ungroup()
}

multi_line_plot <- function(data, color_col, palette, title, subtitle) {
  ggplot(data,
         aes(x = factor(edition), y = pct,
             color = {{ color_col }}, group = {{ color_col }})) +
    geom_line(linewidth = 1) +
    geom_point(size = 2.5) +
    facet_wrap(~ textbook_name, scales = "free_x") +
    scale_color_manual(values = palette) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(title = title, subtitle = subtitle, x = "Edition", y = "Share") +
    theme_clean()
}

race_multi_ex     <- multi_line_examples(multi_ed_df,  race)
race_multi_comp   <- multi_line_composers(multi_ed_df, race)
gender_multi_ex   <- multi_line_examples(multi_ed_df,  gender)
gender_multi_comp <- multi_line_composers(multi_ed_df, gender)

plot_race_multi_ex <- multi_line_plot(
  race_multi_ex, race, race_palette,
  "Racial Representation Across Editions — Share of Examples",
  "Multi-edition textbooks only"
)
plot_race_multi_comp <- multi_line_plot(
  race_multi_comp, race, race_palette,
  "Racial Representation Across Editions — Share of Composers",
  "Multi-edition textbooks only; each composer counted once per edition"
)
plot_gender_multi_ex <- multi_line_plot(
  gender_multi_ex, gender, gender_palette,
  "Gender Representation Across Editions — Share of Examples",
  "Multi-edition textbooks only"
)
plot_gender_multi_comp <- multi_line_plot(
  gender_multi_comp, gender, gender_palette,
  "Gender Representation Across Editions — Share of Composers",
  "Multi-edition textbooks only; each composer counted once per edition"
)


# =============================================================================
# 8. "The Boys" Stacked Bar
# =============================================================================
#
# Each segment = one composer's share of ALL examples in the textbook
# (not share of boys-only examples, which would obscure absolute dominance).
# Stacked bottom to top: Bach → Haydn → Mozart → Beethoven.

boys_totals <- df %>%
  per_book() %>%
  summarise(total = n(), .groups = "drop")

boys_bar <- df %>%
  filter(is_boy) %>%
  per_book(boy_label) %>%
  summarise(n = n(), .groups = "drop") %>%
  # Join on integer keys only — avoids factor-level mismatches on edition_fct
  left_join(boys_totals,
            by = c("textbook_name", "year", "edition", "textbook_id", "short_label")) %>%
  mutate(
    pct       = n / total,
    boy_label = factor(boy_label, levels = c("Beethoven", "Mozart", "Haydn", "Bach"))
  ) %>%
  mutate(textbook_id = factor(textbook_id, levels = book_order))

plot_boys <- ggplot(boys_bar,
       aes(x = textbook_id, y = pct, fill = boy_label)) +
  geom_col(width = 0.75) +
  { if (!is.null(divider_pos))
      geom_vline(xintercept = divider_pos, linetype = "dashed",
                 color = COL_DARK_GRAY, linewidth = 0.7)
  } +
  scale_fill_manual(values = boys_palette,
                    breaks = c("Bach", "Haydn", "Mozart", "Beethoven")) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, .02))) +
  labs(
    title    = '"The Boys": Bach, Haydn, Mozart, & Beethoven',
    subtitle = "Share of all examples; stacked Bach (bottom) → Haydn → Mozart → Beethoven (top)",
    x        = NULL,
    y        = "Share of examples"
  ) +
  theme_clean() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))


# =============================================================================
# 9. Tokenization Plots
# =============================================================================

# 9a. Top-1 share within BIPOC / Female ----------------------------------------
#     The single sharpest tokenization signal.
#     Books with zero subgroup members show 0% (explicit absence, not missing).

plot_top1 <- ggplot(token_df,
       aes(x = textbook_id, y = top1_share, fill = group)) +
  geom_col(position = "dodge", width = 0.7) +
  { if (!is.null(divider_pos))
      geom_vline(xintercept = divider_pos, linetype = "dashed",
                 color = COL_DARK_GRAY, linewidth = 0.7)
  } +
  scale_fill_manual(values = c("BIPOC" = COL_BLUE, "Female" = COL_LIGHT_BLUE)) +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, .02))) +
  labs(
    title    = "Tokenization Signal: Top-1 Composer Share Within Subgroup",
    subtitle = "% of BIPOC / female examples attributed to the single most-cited composer; 0% = no subgroup members",
    x        = NULL,
    y        = "Top-1 composer's share"
  ) +
  theme_clean() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))

# 9b. Absolute breadth scatter -------------------------------------------------
#     x = total appearances in subgroup  (how much space is given)
#     y = unique composers               (how many distinct voices)
#     Point size = breadth ratio         (for reference; unreliable at small n)
#     Reference lines = iso-breadth-ratio curves at 0.25, 0.50, 0.75

ref_lines <- tibble(
  ratio = c(0.25, 0.50, 0.75),
  label = c("1 unique per 4 appearances", "1 per 2", "3 per 4")
)

plot_breadth <- token_df %>%
  filter(subgroup_n > 0) %>%   # exclude zero-member subgroups from scatter
  ggplot(aes(x = subgroup_n, y = n_unique,
             color = group, label = short_label)) +
  geom_abline(data = ref_lines,
              aes(slope = ratio, intercept = 0),
              color = COL_GRAY, linetype = "dotted", linewidth = 0.5,
              inherit.aes = FALSE) +
  geom_text(data = ref_lines,
            aes(x = max(token_df$subgroup_n[token_df$subgroup_n > 0]) * 0.85,
                y = ratio * max(token_df$subgroup_n[token_df$subgroup_n > 0]) * 0.85,
                label = label),
            color = COL_GRAY, size = 2.8, inherit.aes = FALSE) +
  geom_point(aes(size = breadth_ratio), alpha = 0.75) +
  geom_text_repel(size = 2.8, max.overlaps = 15, show.legend = FALSE) +
  scale_color_manual(values = c("BIPOC" = COL_BLUE, "Female" = COL_LIGHT_BLUE)) +
  scale_size_continuous(
    name   = "Breadth ratio\n(unique ÷ appearances)",
    range  = c(2, 8),
    labels = percent_format(accuracy = 1)
  ) +
  labs(
    title    = "Subgroup Breadth: Unique Composers vs. Total Appearances",
    subtitle = "Dotted lines = iso-breadth-ratio; size = breadth ratio (unreliable at small n); zero-member textbooks omitted",
    x        = "Total examples in subgroup",
    y        = "Unique composers in subgroup"
  ) +
  theme_clean()

# 9c. Within-group evenness (Pielou's J') scatter ------------------------------
#     Pielou's J' is the report's sole diversity metric.
#     High J' = subgroup examples spread evenly across composers;
#     low J' = one or two composers dominate. J' is undefined for
#     single-composer subgroups (log(1) = 0), so n_unique >= 2 is required.

plot_evenness_scatter <- token_df %>%
  filter(subgroup_n > 0, n_unique >= 2) %>%
  ggplot(aes(x = subgroup_n, y = within_evenness,
             color = group, label = short_label)) +
  geom_point(size = 3, alpha = 0.85) +
  geom_text_repel(size = 3, max.overlaps = 15, show.legend = FALSE) +
  scale_color_manual(values = c("BIPOC" = COL_BLUE, "Female" = COL_LIGHT_BLUE)) +
  scale_x_log10() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title    = "Within-Group Evenness (Pielou's J\u2032) vs. Subgroup Size",
    subtitle = "High J\u2032 = examples spread evenly across the subgroup's composers; low J\u2032 = one or two dominate; x-axis log-scaled; subgroups with <2 unique composers omitted (J\u2032 undefined)",
    x        = "Number of examples in subgroup (log scale)",
    y        = "Pielou's J\u2032 within subgroup"
  ) +
  theme_clean()

# 9d. Multi-edition tokenization trends ---------------------------------------

if (nrow(multi_ed_token %>% filter(subgroup_n > 0)) > 0) {
  plot_token_multi <- multi_ed_token %>%
    filter(subgroup_n > 0) %>%
    ggplot(aes(x = factor(edition), y = top1_share,
               color = group, group = group)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2.5) +
    facet_wrap(~ textbook_name, scales = "free_x") +
    scale_color_manual(values = c("BIPOC" = COL_BLUE, "Female" = COL_LIGHT_BLUE)) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(
      title    = "Tokenization Across Editions",
      subtitle = "Top-1 composer share within BIPOC / Female subgroup; zero-member editions omitted",
      x        = "Edition",
      y        = "Top-1 share within subgroup"
    ) +
    theme_clean()
}

# 9e. Who are the token composers? (BIPOC) ------------------------------------

plot_token_who_bipoc <- top_bipoc %>%
  group_by(textbook_id) %>%
  mutate(is_top1 = n == max(n)) %>%
  ungroup() %>%
  mutate(textbook_id = factor(textbook_id, levels = book_order)) %>%
  ggplot(aes(x = textbook_id, y = pct, color = is_top1)) +
  geom_point(size = 3, alpha = 0.8, show.legend = FALSE) +
  geom_text_repel(
    data = . %>% filter(is_top1),
    aes(label = composer_clean),
    size = 2.8, max.overlaps = 15, show.legend = FALSE
  ) +
  scale_color_manual(values = c("FALSE" = COL_GRAY, "TRUE" = COL_BLUE)) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title    = "Who Are the Top BIPOC Composers per Textbook?",
    subtitle = paste0("Top ", TOKEN_N, " BIPOC composers; top composer labeled"),
    x        = NULL,
    y        = "Share of BIPOC examples"
  ) +
  theme_clean() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))


# =============================================================================
# 10. Diversity Dashboard Summary Table
# =============================================================================

diversity_summary <- diversity_metrics %>%
  left_join(
    bipoc_token %>% select(textbook_name, year, edition,
                           bipoc_top1     = top1_share,
                           bipoc_evenness = within_evenness,
                           bipoc_breadth  = breadth_ratio,
                           bipoc_unique   = n_unique),
    by = c("textbook_name", "year", "edition")
  ) %>%
  left_join(
    female_token %>% select(textbook_name, year, edition,
                            female_top1     = top1_share,
                            female_evenness = within_evenness,
                            female_breadth  = breadth_ratio,
                            female_unique   = n_unique),
    by = c("textbook_name", "year", "edition")
  ) %>%
  arrange(year, textbook_name, edition) %>%
  select(
    textbook_name, short_label, year, edition,
    n_examples, n_composers,
    boys_pct, bipoc_pct, female_pct,
    pielou_evenness,
    bipoc_top1, bipoc_evenness, bipoc_breadth, bipoc_unique,
    female_top1, female_evenness, female_breadth, female_unique
  )


# =============================================================================
# 11. Geographic Metrics & Visualizations
# =============================================================================

# 11a. Europe-only share -------------------------------------------------------
geo_europe <- df %>%
  mutate(is_europe = continent == "Europe") %>%
  per_book() %>%
  summarise(
    pct_examples_europe  = mean(is_europe, na.rm = TRUE),
    pct_composers_europe = n_distinct(composer_clean[is_europe]) /
                             n_distinct(composer_clean),
    .groups = "drop"
  ) %>%
  mutate(textbook_id = factor(textbook_id, levels = book_order))

# 11b. West (Europe + North America) share ------------------------------------
geo_west <- df %>%
  mutate(is_west = continent %in% c("Europe", "North America")) %>%
  per_book() %>%
  summarise(
    pct_examples_west  = mean(is_west, na.rm = TRUE),
    pct_composers_west = n_distinct(composer_clean[is_west]) /
                           n_distinct(composer_clean),
    .groups = "drop"
  ) %>%
  mutate(textbook_id = factor(textbook_id, levels = book_order))

# 11c. Per-continent stacked bars ---------------------------------------------
geo_cont_examples <- df %>%
  per_book(continent) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(textbook_name, year, edition, textbook_id, short_label) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup() %>%
  mutate(
    continent  = factor(continent, levels = names(continent_palette)),
    textbook_id = factor(textbook_id, levels = book_order)
  )

geo_cont_composers <- df %>%
  distinct(textbook_name, year, edition, textbook_id, short_label,
           composer_clean, continent) %>%
  count(textbook_name, year, edition, textbook_id, short_label,
        continent, name = "n") %>%
  group_by(textbook_name, year, edition, textbook_id, short_label) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup() %>%
  mutate(
    continent  = factor(continent, levels = names(continent_palette)),
    textbook_id = factor(textbook_id, levels = book_order)
  )

plot_geo_cont_examples <- stacked_bar_plot(
  geo_cont_examples, continent, continent_palette,
  "Geographic Representation — Examples",
  "Share of examples by continent of composer birth; dashed line = 2020",
  "Share of examples"
)

plot_geo_cont_composers <- stacked_bar_plot(
  geo_cont_composers, continent, continent_palette,
  "Geographic Representation — Composers",
  "Each composer counted once; dashed line = 2020",
  "Share of unique composers"
)

# 11d. Geographic concentration (HHI by continent) ----------------------------
geo_hhi <- df %>%
  per_book(continent) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(textbook_name, year, edition) %>%
  summarise(
    geo_hhi     = sum((n / sum(n)) ^ 2),
    geo_shannon = -sum((n / sum(n)) * log(n / sum(n))),
    .groups     = "drop"
  )

# 11e. Distance from Vienna — per-composer rows for boxplot + summary stats ------
#
# Distribution shape: strongly right-skewed and effectively bimodal.
# The vast majority of composers were born within ~2 000 km of Vienna (the
# European core); the remainder stretch to 9 000–13 000 km (Americas, East
# Asia, Oceania).  SD is dominated by that tail and is therefore misleading as
# a spread measure.  Better summaries:
#
#   Boxplot — shows median, IQR, whiskers, and individual outlier dots honestly,
#             making the skew and any bimodal structure visible rather than
#             hidden inside a single statistic.
#
#   pct_beyond_2000km — % of geo-coded composers born > 2 000 km from Vienna
#             (roughly the boundary of continental Europe).  Directly answers
#             "how much of the repertoire steps outside the European core?"
#             and is the most interpretable single-number complement to the
#             continent stacked bars.

EUROPE_THRESHOLD_KM <- 2000   # adjust here if needed

vienna <- c(lon = 16.37, lat = 48.21)

# Row-level distance table (kept for the boxplot)
df_dist <- df %>%
  filter(!is.na(latitude), !is.na(longitude)) %>%
  mutate(
    dist_vienna_km  = geosphere::distHaversine(
      cbind(longitude, latitude),
      c(vienna["lon"], vienna["lat"])
    ) / 1000,
    beyond_threshold = dist_vienna_km > EUROPE_THRESHOLD_KM
  )

# Per-textbook summary (for geo_export and the % bar chart)
geo_distance <- df_dist %>%
  per_book() %>%
  summarise(
    n_geocoded            = n(),
    mean_dist_vienna_km   = mean(dist_vienna_km),
    median_dist_vienna_km = median(dist_vienna_km),
    q25_dist_vienna_km    = quantile(dist_vienna_km, 0.25),
    q75_dist_vienna_km    = quantile(dist_vienna_km, 0.75),
    pct_beyond_2000km     = mean(beyond_threshold),
    .groups = "drop"
  )

# ── Boxplot: distribution of composer distances per textbook ──────────────────
#   Each box summarises all geo-coded composer appearances in that textbook.
#   Outlier dots expose the handful of very distant composers that would
#   otherwise inflate a mean or SD measure.
#   Reference line at EUROPE_THRESHOLD_KM marks the European-core boundary.

plot_geo_distance_box <- df_dist %>%
  mutate(textbook_id = factor(textbook_id, levels = book_order)) %>%
  ggplot(aes(x = textbook_id, y = dist_vienna_km)) +
  geom_hline(yintercept = EUROPE_THRESHOLD_KM, linetype = "dotted",
             color = COL_AMBER, linewidth = 0.6) +
  # NOTE: annotate() with a numeric x forces a continuous x scale, which
  # conflicts with the discrete factor textbook_id. Label is in subtitle instead.
  geom_boxplot(
    fill          = COL_LIGHT_BLUE,
    color         = COL_BLUE,
    outlier.color = COL_AMBER,
    outlier.size  = 1.5,
    outlier.alpha = 0.7,
    linewidth     = 0.5,
    width         = 0.65
  ) +
  { if (!is.null(divider_pos))
      geom_vline(xintercept = divider_pos, linetype = "dashed",
                 color = COL_DARK_GRAY, linewidth = 0.7)
  } +
  scale_y_continuous(labels = scales::comma_format(suffix = " km"),
                     expand = expansion(mult = c(0.02, 0.05))) +
  labs(
    title    = "Geographic Spread: Distance from Vienna per Textbook",
    subtitle = paste0(
      "Each box = all geo-coded composer appearances; ",
      "amber dots = outliers beyond 1.5 \u00d7 IQR; ",
      "amber dotted line = ~", EUROPE_THRESHOLD_KM, " km European-core boundary; ",
      "dashed line = 2020"
    ),
    x = NULL,
    y = "Distance from Vienna (km)"
  ) +
  theme_clean() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))

# ── Bar chart: % of composers beyond the European-core threshold ──────────────
#   A single exportable number per textbook that complements the continent
#   stacked bars without being redundant with them.

plot_geo_pct_beyond <- geo_distance %>%
  mutate(textbook_id = factor(textbook_id, levels = book_order)) %>%
  ggplot(aes(x = textbook_id, y = pct_beyond_2000km)) +
  geom_col(fill = COL_BLUE, width = 0.75) +
  { if (!is.null(divider_pos))
      geom_vline(xintercept = divider_pos, linetype = "dashed",
                 color = COL_DARK_GRAY, linewidth = 0.7)
  } +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, .04))) +
  labs(
    title    = paste0("Share of Composers Born > ", EUROPE_THRESHOLD_KM, " km from Vienna"),
    subtitle = "Geo-coded composer appearances only; dashed line = 2020",
    x        = NULL,
    y        = paste0("% beyond ", EUROPE_THRESHOLD_KM, " km")
  ) +
  theme_clean() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))

# 11f. Mean composer location map ---------------------------------------------
#     Each point = centroid of all geo-coded composer birthplaces for one
#     textbook × edition.  Color encodes publication year (sequential scale)
#     so temporal trends in repertoire geography are immediately visible.
#     ggrepel labels prevent overlap for closely-clustered Central European
#     centroids.

world <- map_data("world")

mean_coords <- df %>%
  filter(!is.na(latitude), !is.na(longitude)) %>%
  per_book() %>%
  summarise(
    mean_lat = mean(latitude),
    mean_lon = mean(longitude),
    .groups  = "drop"
  )

plot_geo_map <- ggplot() +
  geom_polygon(data  = world,
               aes(x = long, y = lat, group = group),
               fill  = "gray92", color = "white", linewidth = 0.2) +
  geom_point(data = mean_coords,
             aes(x = mean_lon, y = mean_lat, color = year),
             size = 3.5, alpha = 0.95) +
  geom_text_repel(data = mean_coords,
                  aes(x = mean_lon, y = mean_lat,
                      label = short_label, color = year),
                  size = 2.6, max.overlaps = 25, show.legend = FALSE,
                  box.padding = 0.4, point.padding = 0.3) +
  # Dark blue (oldest) → amber (newest): both ends are fully readable
  scale_color_gradient(
    name = "Publication\nyear",
    low  = COL_BLUE,
    high = COL_AMBER
  ) +
  # Cropped to Europe + Western Russia; all centroids fall within this window
  # because the overwhelming majority of composers in every textbook are European.
  coord_fixed(xlim = c(-38, 35), ylim = c(35, 70), ratio = 1.3) +
  theme_void(base_size = 11) +
  theme(
    plot.title      = element_text(face = "bold", color = COL_BLUE, size = 14),
    plot.subtitle   = element_text(color = COL_DARK_GRAY, size = 10),
    legend.position = "right"
  ) +
  labs(
    title    = "Mean Composer Birthplace per Textbook",
    subtitle = "Centroid of all geo-coded composers; color = publication year (blue = older, amber = newer)"
  )


# =============================================================================
# GEOGRAPHIC ANALYSIS — DEVELOPMENT SUGGESTIONS
# =============================================================================
#
# The analyses below are sketched with running code (marked RUN) or annotated
# scaffolds (marked TODO) for further development.
#
# ── A. Country-level tokenization ───────────────────────────────────────────
#
#  Analogous to composer tokenization: among non-European composers, does one
#  country dominate?  E.g. "90% of non-European examples are American."
#
#  [RUN] Geographic HHI at country level, restricted to non-European composers:

geo_token_non_european <- df %>%
  filter(continent != "Europe", continent != "Unknown", !is.na(country)) %>%
  per_book(country) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(textbook_name, year, edition, textbook_id, short_label) %>%
  summarise(
    n_countries_non_eu = n(),
    geo_hhi_non_eu     = sum((n / sum(n)) ^ 2),
    top1_country_share = max(n) / sum(n),
    top1_country       = country[which.max(n)],
    .groups            = "drop"
  )

#
# ── B. Choropleth map ────────────────────────────────────────────────────────
#
#  [TODO] One choropleth per textbook showing composer count or % by country.
#  Requires sf + rnaturalearth; most useful as a faceted small-multiple or an
#  animated GIF across editions of the same series.
#
#  Scaffold:
#
#  library(sf); library(rnaturalearth)
#  world_sf <- ne_countries(scale = "medium", returnclass = "sf")
#
#  country_counts <- df %>%
#    filter(!is.na(country)) %>%
#    per_book(country) %>%
#    summarise(n = n(), .groups = "drop") %>%
#    group_by(textbook_name, year, edition) %>%
#    mutate(pct = n / sum(n)) %>% ungroup()
#
#  # Join to sf geometries (match on iso_a3 or name_long after manual alignment)
#  choropleth_data <- world_sf %>%
#    left_join(country_counts, by = c("name_long" = "country"))
#
#  ggplot(choropleth_data) +
#    geom_sf(aes(fill = pct), color = "white", linewidth = 0.1) +
#    scale_fill_viridis_c(na.value = "gray92", labels = percent_format()) +
#    facet_wrap(~ paste0(short_label, "\n(", year, ")")) +
#    theme_void() + labs(fill = "Share of examples")
#
# ── C. Geographic entropy (Shannon over countries) ──────────────────────────
#
#  [RUN] Country-level Shannon entropy — more sensitive to rare countries than
#  continent-level HHI.  High entropy = broadly international repertoire.

geo_shannon_country <- df %>%
  filter(!is.na(country)) %>%
  per_book(country) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(textbook_name, year, edition, textbook_id, short_label) %>%
  summarise(
    n_countries      = n(),
    geo_shannon_ctry = -sum((n / sum(n)) * log(n / sum(n))),
    geo_evenness_ctry = geo_shannon_ctry / log(n_countries),
    .groups           = "drop"
  )

#
# ── D. Distance-diversity scatter ───────────────────────────────────────────
#
#  [RUN] Plot mean distance-from-Vienna against overall diversity (HHI or
#  Shannon), colored by BIPOC%.  Tests whether geographic breadth tracks
#  demographic breadth.

geo_diversity_scatter_data <- diversity_metrics %>%
  left_join(geo_distance,   by = c("textbook_name", "year", "edition", "textbook_id", "short_label")) %>%
  left_join(geo_shannon_country, by = c("textbook_name", "year", "edition", "textbook_id", "short_label"))

plot_geo_diversity_scatter <- ggplot(
  geo_diversity_scatter_data,
  aes(x = mean_dist_vienna_km, y = geo_evenness_ctry,
      color = bipoc_pct, label = short_label)
) +
  geom_point(size = 3, alpha = 0.85) +
  geom_text_repel(size = 2.8, max.overlaps = 15, show.legend = FALSE) +
  scale_color_gradient(
    name   = "BIPOC\nexamples",
    labels = percent_format(accuracy = 1),
    low    = COL_GRAY,
    high   = COL_BLUE
  ) +
  labs(
    title    = "Geographic Breadth vs. Distance from Vienna",
    subtitle = "Country-level evenness (Pielou J) vs. mean birthplace distance from Vienna; color = BIPOC %",
    x        = "Mean distance from Vienna (km)",
    y        = "Geographic evenness (country-level Pielou J)"
  ) +
  theme_clean()

#
# ── E. "Geographic tokenization" bar chart ───────────────────────────────────
#
#  [RUN] Mirrors plot_top1 but for countries: what % of non-European examples
#  come from the single most-represented non-European country?

plot_geo_token_top1 <- geo_token_non_european %>%
  mutate(textbook_id = factor(textbook_id, levels = book_order)) %>%
  ggplot(aes(x = textbook_id, y = top1_country_share)) +
  geom_col(fill = COL_AMBER, width = 0.75) +
  geom_text(aes(label = top1_country), angle = 90, hjust = -0.1,
            size = 2.6, color = COL_DARK_GRAY) +
  { if (!is.null(divider_pos))
      geom_vline(xintercept = divider_pos, linetype = "dashed",
                 color = COL_DARK_GRAY, linewidth = 0.7)
  } +
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     expand = expansion(mult = c(0, .15))) +
  labs(
    title    = "Geographic Tokenization: Dominant Non-European Country",
    subtitle = "Share of non-European examples from the single most-represented country; label = that country",
    x        = NULL,
    y        = "Top-1 country share (non-European examples)"
  ) +
  theme_clean() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))


# =============================================================================
# 12. Print & Display
# =============================================================================
#
# All plots printed to the active graphics device.
# Set SAVE_PLOTS <- TRUE at the top to also write PNGs (16 × 8 in, 300 dpi).

save_plot <- function(plot, filename,
                      width = 16, height = 8, dpi = 300) {
  if (SAVE_PLOTS) {
    ggsave(file.path(PLOTS_DIR, filename), plot,
           width = width, height = height, dpi = dpi)
    message("Saved: ", filename)
  }
}

message("=== Diversity & Tokenization Summary ===")
print(diversity_summary, n = Inf)

# ---- Race -------------------------------------------------------------------
print(plot_race_bar_ex);     save_plot(plot_race_bar_ex,     "01_race_examples.png")
print(plot_race_bar_comp);   save_plot(plot_race_bar_comp,   "02_race_composers.png")
print(plot_race_multi_ex);   save_plot(plot_race_multi_ex,   "03_race_editions_ex.png")
print(plot_race_multi_comp); save_plot(plot_race_multi_comp, "04_race_editions_comp.png")

# ---- Gender -----------------------------------------------------------------
print(plot_gender_bar_ex);     save_plot(plot_gender_bar_ex,     "05_gender_examples.png")
print(plot_gender_bar_comp);   save_plot(plot_gender_bar_comp,   "06_gender_composers.png")
print(plot_gender_multi_ex);   save_plot(plot_gender_multi_ex,   "07_gender_editions_ex.png")
print(plot_gender_multi_comp); save_plot(plot_gender_multi_comp, "08_gender_editions_comp.png")

# ---- Canon dominance --------------------------------------------------------
print(plot_boys); save_plot(plot_boys, "09_boys.png")

# ---- Tokenization -----------------------------------------------------------
print(plot_top1);         save_plot(plot_top1,         "10_tokenization_top1.png")
print(plot_breadth);      save_plot(plot_breadth,      "11_subgroup_breadth.png")
print(plot_evenness_scatter); save_plot(plot_evenness_scatter, "12_evenness_scatter.png")

if (exists("plot_token_multi")) {
  print(plot_token_multi); save_plot(plot_token_multi, "13_tokenization_editions.png")
}

print(plot_token_who_bipoc); save_plot(plot_token_who_bipoc, "14_token_who_bipoc.png")

# ---- Geography --------------------------------------------------------------
print(plot_geo_cont_examples);  save_plot(plot_geo_cont_examples,  "15_geo_cont_examples.png")
print(plot_geo_cont_composers); save_plot(plot_geo_cont_composers, "16_geo_cont_composers.png")
print(plot_geo_map);            save_plot(plot_geo_map,            "17_geo_mean_location.png",
                                          width = 14, height = 7)
print(plot_geo_distance_box);   save_plot(plot_geo_distance_box,  "18_geo_distance_boxplot.png")
print(plot_geo_pct_beyond);     save_plot(plot_geo_pct_beyond,    "19_geo_pct_beyond_2000km.png")
print(plot_geo_diversity_scatter); save_plot(plot_geo_diversity_scatter,
                                             "20_geo_diversity_scatter.png")
print(plot_geo_token_top1);     save_plot(plot_geo_token_top1,    "21_geo_tokenization.png")


# =============================================================================
# 13. Data Exports
# =============================================================================
#
# Set SAVE_RESULTS <- TRUE at the top to write all files to RESULTS_DIR.

if (SAVE_RESULTS) {

  write_csv_safe <- function(data, filename, description = NULL) {
    path <- file.path(RESULTS_DIR, filename)
    write_csv(data, path, na = "")
    if (!is.null(description))
      message("Exported: ", filename, "  [", description, "]")
  }

  # 13a. Core summary — one row per textbook × edition
  write_csv_safe(diversity_summary, "01_diversity_summary.csv",
                 "core metrics per textbook × edition")

  # 13b. Boys summary
  boys_export <- diversity_metrics %>%
    select(textbook_name, short_label, year, edition,
           n_examples, boys_n, boys_pct, top_n_share) %>%
    arrange(year, textbook_name, edition)
  write_csv_safe(boys_export, "02_boys_summary.csv",
                 "Bach/Haydn/Mozart/Beethoven share per textbook × edition")

  # 13c. Tokenization metrics — BIPOC and Female (includes zero rows)
  write_csv_safe(
    token_df %>% arrange(group, year, textbook_name, edition),
    "03_token_metrics.csv",
    "tokenization metrics, BIPOC & Female (zero-filled)"
  )

  # 13d. Top composers within each subgroup
  write_csv_safe(
    top_bipoc  %>% arrange(year, textbook_name, edition, desc(n)),
    "04_top_bipoc_composers.csv",
    "top BIPOC composers per textbook × edition"
  )
  write_csv_safe(
    top_female %>% arrange(year, textbook_name, edition, desc(n)),
    "05_top_female_composers.csv",
    "top female composers per textbook × edition"
  )

  # 13e. Full composer frequency table
  composer_freq <- df %>%
    per_book(composer_clean, race, gender, is_boy) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(textbook_name, year, edition) %>%
    mutate(
      pct_of_book  = n / sum(n),
      rank_in_book = rank(-n, ties.method = "min")
    ) %>%
    ungroup() %>%
    arrange(year, textbook_name, edition, rank_in_book)
  write_csv_safe(composer_freq, "06_composer_frequency.csv",
                 "every composer × textbook × edition with counts and flags")

  # 13f. Cross-edition delta table (multi-edition books only)
  edition_delta <- diversity_summary %>%
    filter(textbook_name %in% multi_ed_books) %>%
    arrange(textbook_name, edition) %>%
    group_by(textbook_name) %>%
    mutate(
      prev_edition    = lag(edition),
      d_bipoc_pct     = bipoc_pct       - lag(bipoc_pct),
      d_female_pct    = female_pct      - lag(female_pct),
      d_boys_pct      = boys_pct        - lag(boys_pct),
      d_bipoc_top1    = bipoc_top1      - lag(bipoc_top1),
      d_female_top1   = female_top1     - lag(female_top1),
      d_bipoc_unique  = bipoc_unique    - lag(bipoc_unique),
      d_female_unique = female_unique   - lag(female_unique),
      d_evenness      = pielou_evenness - lag(pielou_evenness)
    ) %>%
    filter(!is.na(prev_edition)) %>%
    select(textbook_name, short_label, year, edition, prev_edition,
           starts_with("d_")) %>%
    ungroup()
  write_csv_safe(edition_delta, "07_edition_deltas.csv",
                 "change in key metrics between consecutive editions")

  # 13g. Geographic summary
  # geo_distance now carries: n_geocoded, mean_dist_vienna_km,
  # median_dist_vienna_km, q25/q75_dist_vienna_km, pct_beyond_2000km.
  # sd_dist_vienna_km is intentionally omitted (misleading for this skewed
  # distribution; the boxplot and pct_beyond_2000km replace it).
  geo_export <- diversity_metrics %>%
    select(textbook_name, short_label, year, edition) %>%
    left_join(geo_europe   %>% select(-textbook_id, -short_label),
              by = c("textbook_name", "year", "edition")) %>%
    left_join(geo_west     %>% select(-textbook_id, -short_label),
              by = c("textbook_name", "year", "edition")) %>%
    left_join(geo_distance %>% select(-textbook_id, -short_label),
              by = c("textbook_name", "year", "edition")) %>%
    left_join(geo_hhi,
              by = c("textbook_name", "year", "edition")) %>%
    left_join(geo_shannon_country %>% select(-textbook_id, -short_label),
              by = c("textbook_name", "year", "edition")) %>%
    arrange(year, textbook_name, edition)
  write_csv_safe(geo_export, "08_geographic_summary.csv",
                 "geographic concentration metrics per textbook × edition")

  # 13h. Excel workbook with all sheets
  library(writexl)
  write_xlsx(
    list(
      "diversity_summary"   = diversity_summary,
      "boys_summary"        = boys_export,
      "token_metrics"       = token_df %>% arrange(group, year, textbook_name, edition),
      "top_bipoc"           = top_bipoc  %>% arrange(year, textbook_name, edition, desc(n)),
      "top_female"          = top_female %>% arrange(year, textbook_name, edition, desc(n)),
      "composer_frequency"  = composer_freq,
      "edition_deltas"      = edition_delta,
      "geographic_summary"  = geo_export
    ),
    path = file.path(RESULTS_DIR, "theory_diversity_full_export.xlsx")
  )
  message("Excel workbook written to: ",
          file.path(RESULTS_DIR, "theory_diversity_full_export.xlsx"))

} # end SAVE_RESULTS

cat("Composers with missing location info:",
    n_distinct(df$composer_clean[is.na(df$latitude) | is.na(df$longitude)]),
    "\n")


# =============================================================================
# NLP Representation Analysis plots
# =============================================================================
#
# Reads the r_data/ CSVs produced by analysis_server.py
# and plots them in the same visual style as Sections 1–13 above.
#
# Section map
#   N1   Paths & load helpers
#   N2   Representation by musical tradition  (fig01)
#   N3   Gender distribution                  (fig02)
#   N4   Era — total, gender, dominance       (fig03a / 03bc)
#   N5   Framing by dominant / marginalized   (fig06)  →  N06a (counts) + N06b (% stacked)
#   N6   Structural placement (top 30)        (fig07)
#   N8   Co-occurrence network                (fig08)
#   N8b  Within-chapter position             (fig10 — primary research question)
#   N8c  Whole-book position                 (fig10 — secondary)
#   N9   Passage role by tradition            (fig10b)
#   N10  Dominant / marginalized by book      (fig13a)
#   N11  Framing distribution by book         (fig13b)
#   N12  Concept contrast                     (fig14)
#   N13  Framing gap by book                  (fig16)
#   N14  Print & save all NLP figures


# =============================================================================
# N1.  Load helpers, palettes, save helper
# =============================================================================
#
# Paths are defined in Section B above (PY_RESULTS_DIR / NLP_DATA_DIR / NLP_PLOTS_DIR).

# Helper: read an r_data CSV and warn if missing rather than hard-failing.
read_nlp <- function(filename) {
  path <- file.path(NLP_DATA_DIR, filename)
  if (!file.exists(path)) {
    warning("NLP data file not found (run Python pipeline first): ", path)
    return(NULL)
  }
  read_csv(path, show_col_types = FALSE)
}

# NLP status palette — mirrors Python PALETTE keys
status_palette <- c(
  "dominant"     = COL_BLUE,
  "marginalized" = COL_AMBER,
  "unclassified" = COL_GRAY
)

status_labels <- c(
  "dominant"     = "Dominant (Western Canon)",
  "marginalized" = "Marginalized / Non-Dominant",
  "unclassified" = "Not Yet Classified"
)

role_palette <- c(
  "central"       = COL_BLUE,
  "supplementary" = COL_AMBER,
  "application"   = COL_GRAY
)

framing_palette <- c(
  "normative"   = COL_BLUE,
  "additive"    = COL_MID_BLUE,
  "exceptional" = COL_AMBER,
  "corrective"  = "#d62728",
  "neutral"     = COL_GRAY
)

era_order <- c("renaissance", "baroque", "classical", "romantic",
               "modern", "contemporary", "unknown")

save_nlp_plot <- function(plot, filename, width = 14, height = 7, dpi = 300) {
  if (SAVE_PLOTS) {
    ggsave(file.path(NLP_PLOTS_DIR, filename), plot,
           width = width, height = height, dpi = dpi, limitsize = FALSE)
    message("Saved: nlp/", filename)
  }
}


# =============================================================================
# N2.  Representation by musical tradition  (fig01)
# =============================================================================

nlp_tradition <- read_nlp("fig01_tradition.csv")

if (!is.null(nlp_tradition)) {

  trad_plot_data <- nlp_tradition %>%
    filter(!is.na(tradition_label), tradition_label != "Unknown" | total_mentions > 5) %>%
    mutate(
      tradition_label = fct_reorder(tradition_label, total_mentions),
      status = factor(status, levels = names(status_palette))
    )

  plot_nlp_tradition <- ggplot(
    trad_plot_data,
    aes(x = total_mentions, y = tradition_label, fill = status)
  ) +
    geom_col(width = 0.75, show.legend = TRUE) +
    geom_text(aes(label = scales::comma(total_mentions)),
              hjust = -0.15, size = 3.2, color = COL_DARK_GRAY) +
    scale_fill_manual(values = status_palette, labels = status_labels,
                      name = NULL) +
    # Log x-axis: Classical Canon is 5–10× larger than any other tradition;
    # a linear axis would compress all non-classical traditions into illegibility.
    scale_x_log10(labels = scales::comma_format(),
                  expand = expansion(mult = c(0, .25))) +
    labs(
      title    = "Composer-Passage Mentions by Musical Tradition (log scale)",
      subtitle = "Log x-axis; Classical Canon dominates; labels show raw counts",
      x        = "Composer-passage mentions (log\u2081\u2080 scale)",
      y        = NULL
    ) +
    theme_clean() +
    theme(legend.position = "bottom")

}


# =============================================================================
# N3.  Gender distribution  (fig02)
# =============================================================================

nlp_gender <- read_nlp("fig02_gender.csv")

if (!is.null(nlp_gender)) {

  plot_nlp_gender <- nlp_gender %>%
    mutate(gender = factor(gender, levels = c("male", "female", "group", "unknown")),
           gender_label = case_when(
             gender == "male"    ~ "Male",
             gender == "female"  ~ "Female",
             gender == "group"   ~ "Group / Band",
             TRUE                ~ "Unknown"
           )) %>%
    ggplot(aes(x = fct_reorder(gender_label, -mentions), y = mentions,
               fill = gender_label)) +
    geom_col(width = 0.65) +
    geom_text(aes(label = scales::comma(mentions)),
              vjust = -0.4, size = 3.5, color = COL_DARK_GRAY) +
    scale_fill_manual(values = c(
      "Male"        = COL_BLUE,
      "Female"      = COL_LIGHT_BLUE,
      "Group / Band"= COL_MID_BLUE,
      "Unknown"     = COL_GRAY
    ), guide = "none") +
    scale_y_continuous(expand = expansion(mult = c(0, .12))) +
    labs(
      title    = "Composer-Passage Mentions by Gender",
      subtitle = "Based on COMPOSER_METADATA entries loaded by the NLP pipeline",
      x        = NULL,
      y        = "Mentions"
    ) +
    theme_clean()

}


# =============================================================================
# N4.  Era — total, gender × era, dominance × era  (fig03a / 03bc)
# =============================================================================

nlp_era_totals <- read_nlp("fig03a_era_totals.csv")
nlp_era_demo   <- read_nlp("fig03bc_era_demographic.csv")

if (!is.null(nlp_era_totals)) {

  plot_nlp_era_totals <- nlp_era_totals %>%
    mutate(era = factor(era, levels = era_order),
           era_label = str_to_title(era)) %>%
    filter(!is.na(era)) %>%
    ggplot(aes(x = era_label, y = total)) +
    geom_col(fill = COL_BLUE, width = 0.7) +
    geom_text(aes(label = scales::comma(total)),
              vjust = -0.4, size = 3.2, color = COL_DARK_GRAY) +
    scale_x_discrete(limits = str_to_title(era_order)) +
    scale_y_continuous(expand = expansion(mult = c(0, .12))) +
    labs(
      title    = "Composer-Passage Mentions by Era",
      subtitle = "Era inferred from birth year in biographical data",
      x        = NULL,
      y        = "Mentions"
    ) +
    theme_clean() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

}

if (!is.null(nlp_era_demo)) {

  era_demo <- nlp_era_demo %>%
    mutate(era = factor(era, levels = era_order),
           era_label = str_to_title(as.character(era))) %>%
    filter(!is.na(era))

  # 3b: absolute counts stacked male / female
  era_gender_long <- era_demo %>%
    select(era_label, era, n_male, n_female) %>%
    pivot_longer(c(n_male, n_female), names_to = "gender", values_to = "n") %>%
    mutate(gender = recode(gender, n_male = "Male", n_female = "Female"),
           era_label = factor(era_label, levels = str_to_title(era_order)))

  plot_nlp_era_gender <- ggplot(era_gender_long,
                                aes(x = era_label, y = n, fill = gender)) +
    geom_col(width = 0.7) +
    geom_text(
      data = era_gender_long %>%
        group_by(era_label) %>%
        summarise(total = sum(n), .groups = "drop"),
      aes(x = era_label, y = total, label = scales::comma(total)),
      inherit.aes = FALSE,
      vjust = -0.4, size = 3.0, color = COL_DARK_GRAY
    ) +
    scale_fill_manual(values = c("Male" = COL_BLUE, "Female" = COL_LIGHT_BLUE),
                      name = NULL) +
    scale_y_continuous(expand = expansion(mult = c(0, .12))) +
    labs(
      title    = "Composer-Passage Mentions by Era × Gender",
      subtitle = "Stacked: male (dark) + female (light); era ordered chronologically",
      x        = NULL,
      y        = "Mentions"
    ) +
    theme_clean() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          legend.position = "bottom")

  # 3c: % female and % BIPOC by era
  # N03c: % Female and % BIPOC by era
  # The Python CSV stores pct_female and pct_bipoc as values in the 0–100 range
  # (e.g. 15.3 means 15.3%).  Divide by 100 before plotting so that
  # percent_format() (which expects 0–1 proportions) displays correctly.
  era_pct_long <- era_demo %>%
    select(era_label, era, pct_female, pct_bipoc) %>%
    pivot_longer(c(pct_female, pct_bipoc), names_to = "group", values_to = "pct") %>%
    mutate(
      pct       = pct / 100,           # convert 0–100 → 0–1 for percent_format
      group     = recode(group, pct_female = "Female", pct_bipoc = "BIPOC"),
      era_label = factor(era_label, levels = str_to_title(era_order))
    )

  plot_nlp_era_pct <- ggplot(era_pct_long,
                              aes(x = era_label, y = pct, fill = group)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.7) +
    geom_text(
      aes(label = percent(pct, accuracy = 0.1)),
      position = position_dodge(width = 0.75),
      vjust = -0.4, size = 2.6, color = COL_DARK_GRAY
    ) +
    scale_fill_manual(
      values = c("Female" = COL_LIGHT_BLUE, "BIPOC" = COL_AMBER),
      name = NULL
    ) +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       expand = expansion(mult = c(0, .15))) +
    labs(
      title    = "% Female and % BIPOC Mentions by Era",
      subtitle = "Dodged bars; % of all composer-passage mentions within each era",
      x        = NULL,
      y        = "% of mentions within era"
    ) +
    theme_clean() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          legend.position = "bottom")

  # N03d: stacked dominant / marginalized / unclassified % by era
  # Same fix: divide by 100 so percent_format() reads correctly.
  era_status_long <- era_demo %>%
    mutate(
      era_label    = factor(str_to_title(as.character(era)), levels = str_to_title(era_order)),
      pct_dominant = n_dominant    / total,   # 0–1
      pct_marg     = n_marginalized/ total,   # 0–1
      pct_unc      = pmax(0, 1 - pct_dominant - pct_marg)
    ) %>%
    select(era_label, pct_dominant, pct_marg, pct_unc) %>%
    pivot_longer(-era_label, names_to = "status", values_to = "pct") %>%
    mutate(status = recode(status,
                           pct_dominant = "dominant",
                           pct_marg     = "marginalized",
                           pct_unc      = "unclassified"),
           status = factor(status, levels = names(status_palette)))

  plot_nlp_era_status <- ggplot(era_status_long,
                                 aes(x = era_label, y = pct, fill = status)) +
    geom_col(width = 0.7) +
    scale_fill_manual(values = status_palette, labels = status_labels, name = NULL) +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       expand = expansion(mult = c(0, .04))) +
    labs(
      title    = "Dominant vs. Marginalized Composer Mentions by Era (%)",
      subtitle = "Stacked 100%; unclassified = not yet in COMPOSER_METADATA",
      x        = NULL,
      y        = "% of era mentions"
    ) +
    theme_clean() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          legend.position = "bottom")

}


# =============================================================================
# N5.  Framing by dominant / marginalized status  (fig06)
# =============================================================================
#
#  N06a — grouped bar, absolute counts (original plot, kept for raw volumes)
#  N06b — two stacked 100% bars: dominant | marginalized, coloured by framing
#          category. Shows the proportional framing composition within each
#          tradition group so the gap is immediately readable.

nlp_framing_status <- read_nlp("fig06_framing_by_status.csv")

if (!is.null(nlp_framing_status)) {

  # ── Shared factor prep ──────────────────────────────────────────────────────
  framing_levels <- c("normative", "additive", "exceptional", "corrective", "neutral")
  framing_labels_named <- c(
    normative   = "Normative",
    additive    = "Additive",
    exceptional = "Exceptional",
    corrective  = "Corrective",
    neutral     = "Neutral"
  )

  nlp_framing_status <- nlp_framing_status %>%
    mutate(
      status           = factor(status, levels = c("dominant", "marginalized")),
      framing_category = factor(framing_category, levels = framing_levels)
    ) %>%
    filter(!is.na(framing_category), !is.na(status))

  # ── N06a: grouped bars, absolute counts ─────────────────────────────────────
  plot_nlp_framing_status <- nlp_framing_status %>%
    ggplot(aes(x = framing_category, y = mentions, fill = status)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.7) +
    geom_text(
      aes(label = scales::comma(mentions)),
      position = position_dodge(width = 0.75),
      vjust = -0.4, size = 2.8, color = COL_DARK_GRAY
    ) +
    scale_fill_manual(values = c(dominant = COL_BLUE, marginalized = COL_AMBER),
                      labels = c(dominant     = "Dominant (Western Canon)",
                                 marginalized = "Marginalized / Non-Dominant"),
                      name = NULL) +
    scale_y_continuous(expand = expansion(mult = c(0, .14))) +
    labs(
      title    = "Framing Category: Dominant vs. Marginalized Composers",
      subtitle = "How composers are rhetorically introduced in textbook passages",
      x        = "Framing category",
      y        = "Composer-passage mentions"
    ) +
    theme_clean() +
    theme(legend.position = "bottom",
          axis.text.x = element_text(angle = 25, hjust = 1))

  # ── N06b: stacked 100% bars, one bar per tradition group ────────────────────
  framing_pct <- nlp_framing_status %>%
    group_by(status) %>%
    mutate(pct = mentions / sum(mentions) * 100) %>%
    ungroup()

  plot_nlp_framing_pct <- framing_pct %>%
    ggplot(aes(x = status, y = pct, fill = framing_category)) +
    geom_col(width = 0.55, colour = "white", linewidth = 0.3) +
    # Label segments >= 3% so text never overprints a sliver
    geom_text(
      aes(label = ifelse(pct >= 3, sprintf("%.1f%%", pct), "")),
      position  = position_stack(vjust = 0.5),
      size      = 3.4,
      colour    = "white",
      fontface  = "bold"
    ) +
    scale_fill_manual(
      values = framing_palette,
      labels = framing_labels_named,
      name   = "Framing category",
      # Reverse legend so top-of-bar category appears first
      guide  = guide_legend(reverse = TRUE)
    ) +
    scale_x_discrete(
      labels = c(dominant     = "Dominant\n(Western Canon)",
                 marginalized = "Marginalized /\nNon-Dominant")
    ) +
    scale_y_continuous(
      labels = function(x) paste0(x, "%"),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(
      title    = "Framing Composition by Tradition Group",
      subtitle = "Percentage of composer-passage mentions by framing category within each tradition",
      x        = NULL,
      y        = "% of mentions"
    ) +
    theme_clean() +
    theme(legend.position = "right")

}


# =============================================================================
# N6.  Structural placement — top 30 composers  (fig07)
# =============================================================================

nlp_placement <- read_nlp("fig07_structural_placement.csv")

if (!is.null(nlp_placement)) {

  placement_long <- nlp_placement %>%
    arrange(desc(mention_count)) %>%
    slice_head(n = 30) %>%
    mutate(
      composer_canonical = fct_reorder(composer_canonical, mention_count),
      status             = factor(status, levels = names(status_palette)),
      # Build sub-columns if present; fall back to mention_count as central
      n_central       = if ("n_central"       %in% names(.)) n_central       else mention_count,
      n_supplementary = if ("n_supplementary" %in% names(.)) n_supplementary else 0L,
      n_application   = if ("n_application"   %in% names(.)) n_application   else 0L
    ) %>%
    pivot_longer(c(n_central, n_supplementary, n_application),
                 names_to = "role", values_to = "n_role") %>%
    mutate(role = factor(recode(role,
                                n_central       = "central",
                                n_supplementary = "supplementary",
                                n_application   = "application"),
                         levels = c("central", "supplementary", "application")))

  plot_nlp_placement <- ggplot(
    placement_long,
    aes(x = n_role, y = composer_canonical, fill = status, alpha = role)
  ) +
    geom_col(orientation = "y", width = 0.75) +
    scale_fill_manual(values = status_palette, labels = status_labels, name = NULL) +
    scale_alpha_manual(
      values = c(central = 1.0, supplementary = 0.45, application = 0.25),
      labels = c(central = "Central passages",
                 supplementary = "Supplementary",
                 application   = "Application / exercises"),
      name = "Passage role"
    ) +
    # Square-root x axis: compresses the long tail (top 5 composers > 100 mentions,
    # remaining 25 cluster 5–80) without breaking near-zero values the way log10 does.
    # log10 is incompatible with stacked bars that include zero-valued segments.
    scale_x_continuous(trans  = "sqrt",
                       labels = scales::comma_format(),
                       expand = expansion(mult = c(0, .20))) +
    labs(
      title    = "Top 30 Composers: Structural Placement (√ scale)",
      subtitle = "Dark fill = central  ·  mid = supplementary  ·  light = application/exercises; square-root x-axis",
      x        = "Composer-passage mentions (√ scale)",
      y        = NULL
    ) +
    theme_clean() +
    theme(legend.position = "right",
          axis.text.y = element_text(size = 9))

}


# =============================================================================
# N8.  Co-occurrence network  (fig08)
# =============================================================================
#
# Requires: igraph + ggraph (loaded in Section A)
#
# Data source: r_data/fig08_cooccurrence_edges.csv
#              r_data/fig08_cooccurrence_nodes.csv
#
# Both are written by save_r_data() in analysis_server.py.
# The 'threshold' column in the edges file records the COOC_THRESHOLD value
# used by the Python pipeline; R uses the same value to filter edges.

nlp_edges <- read_nlp("fig08_cooccurrence_edges.csv")
nlp_nodes <- read_nlp("fig08_cooccurrence_nodes.csv")

if (!is.null(nlp_edges) && !is.null(nlp_nodes) && nrow(nlp_edges) > 0) {

  # Read threshold from the data; fall back to 2 if column absent
  COOC_THRESH <- if ("threshold" %in% names(nlp_edges)) nlp_edges$threshold[1] else 2L

  edges_filt <- nlp_edges %>%
    filter(weight >= COOC_THRESH,
           composer_a != composer_b)   # guard against self-loops

  if (nrow(edges_filt) == 0) {
    # No edges above threshold — relax to top-40 pairs
    edges_filt <- nlp_edges %>%
      arrange(desc(weight)) %>%
      slice_head(n = 40) %>%
      filter(composer_a != composer_b)
    message("[N8] No edges above threshold ", COOC_THRESH,
            "; showing top-40 pairs instead.")
  }

  # Nodes present in filtered edge set
  active_nodes <- base::union(edges_filt$composer_a, edges_filt$composer_b)

  node_data <- nlp_nodes %>%
    filter(composer %in% active_nodes) %>%
    mutate(
      status        = replace_na(status, "unclassified"),
      status        = factor(status, levels = names(status_palette)),
      total_mentions = replace_na(total_mentions, 1)
    )

  # Build igraph object
  g <- graph_from_data_frame(
    d        = edges_filt %>% select(from = composer_a, to = composer_b, weight),
    directed = FALSE,
    vertices = node_data %>% select(name = composer, everything())
  )

  plot_nlp_cooccurrence <- ggraph(g, layout = "fr") +
    geom_edge_link(
      aes(width = weight, alpha = weight),
      color       = COL_GRAY,
      show.legend = FALSE
    ) +
    geom_node_point(
      aes(size = total_mentions, fill = status),
      shape  = 21,
      color  = "white",
      stroke = 0.6,
      alpha  = 0.92
    ) +
    geom_node_text(
      aes(label = name,
          # only label nodes above median mentions to reduce clutter
          label = if_else(total_mentions >= median(total_mentions), name, "")),
      size          = 2.6,
      repel         = TRUE,
      max.overlaps  = 20,
      color         = COL_DARK_GRAY,
      box.padding   = 0.3,
      point.padding = 0.2
    ) +
    scale_fill_manual(values  = status_palette,
                      labels  = status_labels,
                      name    = NULL) +
    scale_size_continuous(
      name   = "Total mentions",
      range  = c(2.5, 12),
      labels = scales::comma_format()
    ) +
    scale_edge_width_continuous(range = c(0.3, 3.0)) +
    scale_edge_alpha_continuous(range = c(0.15, 0.80)) +
    labs(
      title    = "Composer Co-occurrence Network",
      subtitle = paste0(
        "Edges \u2265 ", COOC_THRESH, " shared passages; ",
        "node size = total mentions; color = dominant / marginalized / unclassified; ",
        "layout = Fruchterman\u2013Reingold"
      )
    ) +
    theme_graph(base_family = "sans") +
    theme(
      plot.title       = element_text(face = "bold", color = COL_BLUE,      size = 14),
      plot.subtitle    = element_text(             color = COL_DARK_GRAY,   size = 9),
      legend.position  = "bottom",
      legend.text      = element_text(size = 9),
      legend.title     = element_text(size = 9)
    )

}


# =============================================================================
# N8b.  Within-chapter position of composer mentions  (fig10)
# =============================================================================
#
# Two complementary density plots:
#   (a) Within-chapter position: where in a chapter does each composer appear?
#       (0 = first section of chapter, 1 = last section)
#       Research question: are marginalized composers an afterthought, cited
#       only at the end of chapters once the "core" content is done?
#   (b) Whole-book position: which chapters (early vs. late in the textbook)
#       feature marginalized vs. dominant composers?
#       (0 = chapter 1, 1 = final chapter)
#
# Both variables are now in fig10_chapter_position.csv.

nlp_chapter_pos <- read_nlp("fig10_chapter_position.csv")

if (!is.null(nlp_chapter_pos)) {

  # Helper: build the filtered position dataset for either variable
  make_pos_data <- function(data, pos_var) {
    data %>%
      filter(!is.na(.data[[pos_var]])) %>%
      mutate(
        pos_x      = .data[[pos_var]],
        status     = factor(status, levels = names(status_palette),
                            labels = c("Dominant", "Marginalized", "Unclassified")),
        book_label = str_replace_all(book_id, "_", " ") %>% str_to_title()
      ) %>%
      # Count per book × status; drop sparse curves (n < 5 = misleading density)
      add_count(book_label, status, name = "n_obs") %>%
      filter(n_obs >= 5,
             status %in% c("Dominant", "Marginalized"))
  }

  status_colors_named <- c("Dominant" = COL_BLUE, "Marginalized" = COL_AMBER)

  make_pos_plot <- function(filtered_data, x_label, title, subtitle) {
    ggplot(filtered_data,
           aes(x = pos_x, color = status, fill = status)) +
      geom_density(alpha = 0.18, linewidth = 0.9) +
      facet_wrap(~ book_label, ncol = 4) +
      scale_color_manual(values = status_colors_named, name = NULL) +
      scale_fill_manual( values = status_colors_named, name = NULL) +
      scale_x_continuous(labels = percent_format(accuracy = 1),
                         breaks = c(0, 0.5, 1),
                         limits = c(0, 1)) +
      labs(title = title, subtitle = subtitle,
           x = x_label, y = "Density") +
      theme_clean() +
      theme(
        legend.position = "bottom",
        strip.text      = element_text(size = 8, face = "bold"),
        axis.text.x     = element_text(size = 7),
        axis.text.y     = element_text(size = 7),
        panel.spacing   = unit(0.8, "lines")
      )
  }

  # (a) Within-chapter position — primary research question
  # Resolve column names without mutating nlp_chapter_pos, so both plots (a)
  # and (b) can independently query whatever columns are actually present.
  #
  # (a) Within-chapter position: preferred column name is 'within_chapter_position';
  #     fall back to 'chapter_position' if that is what the pipeline exported.
  within_col <- dplyr::case_when(
    "within_chapter_position" %in% names(nlp_chapter_pos) ~ "within_chapter_position",
    "chapter_position"        %in% names(nlp_chapter_pos) ~ "chapter_position",
    TRUE                                                   ~ NA_character_
  )
  if (!is.na(within_col)) {
    if (within_col == "chapter_position")
      message("N08b: 'within_chapter_position' absent; using 'chapter_position' as fallback")
    within_data <- make_pos_data(nlp_chapter_pos, within_col)
    plot_nlp_within_chapter_pos <- make_pos_plot(
      within_data,
      x_label  = "Position within chapter (0 = first section, 1 = last section)",
      title    = "Within-Chapter Placement of Composer Mentions",
      subtitle = paste0(
        "Are marginalized composers cited only at chapter end (as an afterthought)?",
        "\nDensity curves per textbook; dominant (blue) vs. marginalized (amber)"
      )
    )
  }

  # (b) Whole-book position — secondary question (requires a separate column;
  #     skipped gracefully if not present in this pipeline export).
  book_col <- dplyr::case_when(
    "book_position"   %in% names(nlp_chapter_pos) ~ "book_position",
    "passage_position"%in% names(nlp_chapter_pos) ~ "passage_position",
    TRUE                                           ~ NA_character_
  )
  if (!is.na(book_col)) {
    book_data <- make_pos_data(nlp_chapter_pos, book_col)
    plot_nlp_chapter_pos <- make_pos_plot(
      book_data,
      x_label  = "Position across whole textbook (0 = first chapter, 1 = last chapter)",
      title    = "Whole-Book Placement of Composer Mentions",
      subtitle = "In which chapters (early vs. late) do dominant and marginalized composers appear?"
    )
  } else {
    message("N08c: whole-book position column not found in fig10_chapter_position.csv; plot skipped")
  }

}


# =============================================================================
# N9.  Passage role by musical tradition  (fig10b)
# =============================================================================

nlp_role_trad <- read_nlp("fig10b_passage_role_by_tradition.csv")

if (!is.null(nlp_role_trad)) {

  role_trad_data <- nlp_role_trad %>%
    filter(!is.na(tradition_label), tradition_label != "Unknown" | mentions > 5) %>%
    mutate(
      tradition_label = fct_reorder(tradition_label,
                                    ifelse(passage_role == "central", mentions, 0),
                                    .fun = sum),
      passage_role    = factor(passage_role, levels = c("central","supplementary","application"))
    )

  plot_nlp_role_trad <- ggplot(
    role_trad_data,
    aes(x = mentions, y = tradition_label, fill = passage_role)
  ) +
    geom_col(width = 0.75) +
    scale_fill_manual(values = role_palette,
                      labels = c(central       = "Central",
                                 supplementary  = "Supplementary",
                                 application    = "Application / exercises"),
                      name = "Passage role") +
    scale_x_continuous(trans  = "sqrt",
                       labels = scales::comma_format(),
                       expand = expansion(mult = c(0, .10))) +
    labs(
      title    = "Passage Role by Musical Tradition (√ scale)",
      subtitle = "Central = core pedagogical passages; supplementary = sidebars; application = exercises; square-root x-axis",
      x        = "Composer-passage mentions (√ scale)",
      y        = NULL
    ) +
    theme_clean() +
    theme(legend.position = "bottom")

}


# =============================================================================
# N10.  Dominant / marginalized % by textbook  (fig13a)
# =============================================================================

nlp_rep_book <- read_nlp("fig13a_representation_by_book.csv")

if (!is.null(nlp_rep_book)) {

  rep_book_data <- nlp_rep_book %>%
    mutate(
      status     = factor(status, levels = c("dominant","marginalized")),
      book_label = str_replace_all(book_id, "_", " ") %>% str_to_title(),
      book_label = fct_reorder(book_label, -pct, .fun = max)
    )

  plot_nlp_rep_book <- ggplot(
    rep_book_data,
    aes(x = book_label, y = pct, fill = status)
  ) +
    geom_col(position = position_dodge(width = 0.75), width = 0.7) +
    geom_text(
      aes(label = paste0(round(pct, 1), "%")),
      position = position_dodge(width = 0.75),
      vjust = -0.4, size = 3.0, color = COL_DARK_GRAY
    ) +
    scale_fill_manual(values = c(dominant = COL_BLUE, marginalized = COL_AMBER),
                      labels = c(dominant     = "Dominant (Western Canon)",
                                 marginalized = "Marginalized / Non-Dominant"),
                      name = NULL) +
    scale_y_continuous(expand = expansion(mult = c(0, .12))) +
    labs(
      title    = "Dominant vs. Marginalized Composer Representation by Textbook",
      subtitle = "% of composer-passage mentions classified as dominant or marginalized",
      x        = NULL,
      y        = "% of composer-passage mentions"
    ) +
    theme_clean() +
    theme(axis.text.x = element_text(angle = 40, hjust = 1, size = 9),
          legend.position = "bottom")

}


# =============================================================================
# N11.  Framing distribution by textbook  (fig13b)
# =============================================================================

nlp_framing_book <- read_nlp("fig13b_framing_by_book.csv")

if (!is.null(nlp_framing_book)) {

  framing_book_data <- nlp_framing_book %>%
    mutate(
      framing    = factor(framing, levels = c("normative","additive","neutral")),
      book_label = str_replace_all(book_id, "_", " ") %>% str_to_title()
    ) %>%
    filter(!is.na(framing))

  plot_nlp_framing_book <- ggplot(
    framing_book_data,
    aes(x = book_label, y = pct, fill = framing)
  ) +
    geom_col(position = position_dodge(width = 0.75), width = 0.65) +
    geom_text(
      aes(label = paste0(round(pct, 1), "%")),
      position = position_dodge(width = 0.75),
      vjust = -0.4, size = 2.6, color = COL_DARK_GRAY
    ) +
    scale_fill_manual(
      values = c(normative = COL_BLUE, additive = COL_MID_BLUE, neutral = COL_GRAY),
      labels = c(normative = "Normative", additive = "Additive", neutral = "Neutral"),
      name = NULL
    ) +
    scale_y_continuous(labels = percent_format(accuracy = 1, scale = 1),
                       expand = expansion(mult = c(0, .15))) +
    labs(
      title    = "Framing Category Distribution by Textbook",
      subtitle = "% of all composer-passage mentions with each framing label",
      x        = NULL,
      y        = "% of composer-passage mentions"
    ) +
    theme_clean() +
    theme(axis.text.x = element_text(angle = 40, hjust = 1, size = 9),
          legend.position = "bottom")

}


# =============================================================================
# N12.  Concept contrast: dominant vs. marginalized passages  (fig14)
# =============================================================================

nlp_concept <- read_nlp("fig14_concept_contrast.csv")

if (!is.null(nlp_concept)) {

  concept_data <- nlp_concept %>%
    mutate(
      contrast        = as.numeric(contrast),
      n_dom_passages  = as.numeric(n_dom_passages),
      n_marg_passages = as.numeric(n_marg_passages),
      # sign convention: negative contrast = marg_freq > dom_freq (marginalized-associated)
      direction       = if_else(contrast <= 0, "marginalized", "dominant"),
      concept         = fct_reorder(concept, contrast)
    ) %>%
    # Neutral-band threshold calibrated to this data's scale.
    # analysis_11 contrast = dom_freq - marg_freq normalized over ALL passages,
    # so values top out at ~0.07. The 0.02 threshold retains the ~30 most
    # distinctive concepts while discarding the uninformative middle mass.
    filter(abs(contrast) >= 0.02)

  # Height: 0.38 in per row, capped at 22 in (readable on A3 / tabloid).
  concept_plot_height <- min(max(6, nrow(concept_data) * 0.38), 22)

  plot_nlp_concept <- ggplot(concept_data,
                              aes(x = contrast, y = concept, fill = direction)) +
    geom_col(width = 0.75, show.legend = FALSE) +
    geom_vline(xintercept = 0, linewidth = 0.7, color = COL_DARK_GRAY,
               linetype = "dashed") +
    geom_text(
      aes(label = paste0("d=", n_dom_passages, " m=", n_marg_passages)),
      hjust = if_else(concept_data$contrast <= 0, 1.08, -0.08),
      size = 2.6, color = COL_DARK_GRAY
    ) +
    scale_fill_manual(values = c(dominant = COL_BLUE, marginalized = COL_AMBER)) +
    scale_x_continuous(expand = expansion(mult = c(.18, .18))) +
    labs(
      title    = "Concept Association: Dominant vs. Marginalized Composer Passages",
      subtitle = paste0(
        "Concepts with |contrast| \u2265 0.02 (", nrow(concept_data), " of ",
        nrow(nlp_concept), " shown). ",
        "Contrast = dom_freq \u2212 marg_freq (all-passage normalized). ",
        "Blue = dominant-associated, amber = marginalized."
      ),
      x = "Contrast (positive = dominant-associated; negative = marginalized-associated)",
      y = NULL
    ) +
    theme_clean() +
    theme(axis.text.y = element_text(size = 8))

}


# =============================================================================
# N13.  Framing gap by textbook  (fig16)
# =============================================================================

nlp_framing_gap <- read_nlp("fig16_framing_gap.csv")

if (!is.null(nlp_framing_gap)) {

  framing_gap_data <- nlp_framing_gap %>%
    mutate(
      status     = factor(status, levels = c("dominant","marginalized")),
      framing    = factor(framing, levels = c("normative","additive","exceptional")),
      book_label = str_replace_all(book_id, "_", " ") %>% str_to_title()
    )

  # Normative framing gap
  plot_nlp_norm_gap <- framing_gap_data %>%
    filter(framing == "normative") %>%
    ggplot(aes(x = book_label, y = pct, fill = status)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.7) +
    scale_fill_manual(values = c(dominant = COL_BLUE, marginalized = COL_AMBER),
                      labels = c(dominant     = "Dominant",
                                 marginalized = "Marginalized"),
                      name = NULL) +
    scale_y_continuous(labels = percent_format(accuracy = 1, scale = 1),
                       expand = expansion(mult = c(0, .08))) +
    labs(
      title    = "Normative Framing: Dominant vs. Marginalized by Textbook",
      subtitle = "% of each group's mentions carrying normative framing (standard, typical, fundamental)",
      x        = NULL,
      y        = "% with normative framing"
    ) +
    theme_clean() +
    theme(axis.text.x = element_text(angle = 40, hjust = 1, size = 9),
          legend.position = "bottom")

  # Additive framing gap
  plot_nlp_add_gap <- framing_gap_data %>%
    filter(framing == "additive") %>%
    ggplot(aes(x = book_label, y = pct, fill = status)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.7) +
    scale_fill_manual(values = c(dominant = COL_BLUE, marginalized = COL_AMBER),
                      labels = c(dominant     = "Dominant",
                                 marginalized = "Marginalized"),
                      name = NULL) +
    scale_y_continuous(labels = percent_format(accuracy = 1, scale = 1),
                       expand = expansion(mult = c(0, .08))) +
    labs(
      title    = "Additive Framing: Dominant vs. Marginalized by Textbook",
      subtitle = "% of each group's mentions carrying additive framing (also, in addition, another example)",
      x        = NULL,
      y        = "% with additive framing"
    ) +
    theme_clean() +
    theme(axis.text.x = element_text(angle = 40, hjust = 1, size = 9),
          legend.position = "bottom")

}


# =============================================================================
# N14.  Print & save all NLP figures
# =============================================================================

if (exists("plot_nlp_tradition"))
  { print(plot_nlp_tradition);   save_nlp_plot(plot_nlp_tradition,   "N01_tradition.png") }

if (exists("plot_nlp_gender"))
  { print(plot_nlp_gender);      save_nlp_plot(plot_nlp_gender,      "N02_gender.png",
                                               width = 8, height = 5) }

if (exists("plot_nlp_era_totals"))
  { print(plot_nlp_era_totals);  save_nlp_plot(plot_nlp_era_totals,  "N03a_era_totals.png",
                                               width = 10, height = 6) }

if (exists("plot_nlp_era_gender"))
  { print(plot_nlp_era_gender);  save_nlp_plot(plot_nlp_era_gender,  "N03b_era_gender.png",
                                               width = 10, height = 6) }

if (exists("plot_nlp_era_pct"))
  { print(plot_nlp_era_pct);     save_nlp_plot(plot_nlp_era_pct,     "N03c_era_pct.png",
                                               width = 10, height = 6) }

if (exists("plot_nlp_era_status"))
  { print(plot_nlp_era_status);  save_nlp_plot(plot_nlp_era_status,  "N03d_era_status.png",
                                               width = 10, height = 6) }

if (exists("plot_nlp_framing_status"))
  { print(plot_nlp_framing_status); save_nlp_plot(plot_nlp_framing_status, "N06a_framing_status.png",
                                                  width = 10, height = 6) }

if (exists("plot_nlp_framing_pct"))
  { print(plot_nlp_framing_pct);    save_nlp_plot(plot_nlp_framing_pct,    "N06b_framing_pct.png",
                                                  width = 7,  height = 8) }

if (exists("plot_nlp_placement"))
  { print(plot_nlp_placement);   save_nlp_plot(plot_nlp_placement,   "N07_structural_placement.png",
                                               width = 12, height = 10) }

if (exists("plot_nlp_cooccurrence"))
  { print(plot_nlp_cooccurrence); save_nlp_plot(plot_nlp_cooccurrence, "N08_cooccurrence_network.png",
                                                width = 14, height = 11) }

if (exists("plot_nlp_within_chapter_pos"))
  { print(plot_nlp_within_chapter_pos);
    save_nlp_plot(plot_nlp_within_chapter_pos, "N08b_within_chapter_position.png",
                  width = 14, height = 9) }

if (exists("plot_nlp_chapter_pos"))
  { print(plot_nlp_chapter_pos); save_nlp_plot(plot_nlp_chapter_pos, "N08c_whole_book_position.png",
                                               width = 14, height = 9) }

if (exists("plot_nlp_role_trad"))
  { print(plot_nlp_role_trad);   save_nlp_plot(plot_nlp_role_trad,   "N10b_passage_role_tradition.png",
                                               width = 12, height = 7) }

if (exists("plot_nlp_rep_book"))
  { print(plot_nlp_rep_book);    save_nlp_plot(plot_nlp_rep_book,    "N13a_representation_by_book.png") }

if (exists("plot_nlp_framing_book"))
  { print(plot_nlp_framing_book); save_nlp_plot(plot_nlp_framing_book, "N13b_framing_by_book.png") }

if (exists("plot_nlp_concept"))
  { print(plot_nlp_concept);     save_nlp_plot(plot_nlp_concept,     "N14_concept_contrast.png",
                                               width = 13, height = concept_plot_height) }

if (exists("plot_nlp_norm_gap"))
  { print(plot_nlp_norm_gap);    save_nlp_plot(plot_nlp_norm_gap,    "N16a_normative_gap.png") }

if (exists("plot_nlp_add_gap"))
  { print(plot_nlp_add_gap);     save_nlp_plot(plot_nlp_add_gap,     "N16b_additive_gap.png") }

message("=== NLP Analysis plots complete ===")


# =============================================================================
# Statistical-support plots (read directly from textrep.db)
# =============================================================================
#
# These plots support the article's Results section and are kept in lock-step
# with framing_analysis.py by reading the same source tables from textrep.db
# rather than the intermediate r_data CSVs.  Each plot answers one claim in the
# Results prose.
#
# Section map
#   N15   DB connection + shared labels/orders
#   N15a  PRIMARY: framing pattern is NOT uniform across books
#           (per-book marginalized framing profile, with corpus reference)
#   N15b  Framing gap dot-and-line (dominant vs marginalized, faceted by category)
#   N15c  Structural placement by gender and BIPOC (separate axes)
#   N15d  Integration vs breadth scatter (volume is not integration)
#   N16   Print & save statistical-support figures
#
# All of these plots degrade gracefully: if DB_PATH is missing, the section is
# skipped with a warning and the rest of the script is unaffected.


# =============================================================================
# N15.  DB connection + shared labels / orders
# =============================================================================

con_db <- NULL
if (file.exists(DB_PATH)) {
  con_db <- dbConnect(RSQLite::SQLite(), DB_PATH)
} else {
  warning("textrep.db not found at DB_PATH; statistical-support plots skipped: ", DB_PATH)
}

# Pretty labels for book_id values. Both book-ID namespaces are covered:
# server keys (as they appear in analysis_* tables built before BOOK_ID_MAP in
# build_db.py covered every book) and JSONL stems (fully normalised builds).
nlp_book_labels <- c(
  aldwell_2019          = "Aldwell & Schachter 5th (2019)",
  aldwell_schachter_5ed = "Aldwell & Schachter 5th (2019)",
  mount_fff_2020        = "Mount (2020)",
  roig_francoli_2020    = "Roig-Francoli 3rd (2020)",
  roig_francoli_3ed     = "Roig-Francoli 3rd (2020)",
  benward_saker_v1_10ed = "Benward & Saker 10th (2021)",
  gotham_2023           = "Gotham, OMT (2023)",
  gotham_omt            = "Gotham, OMT (2023)",
  laitz_5ed             = "Laitz 5th (2023)",
  kostka_almen_9ed      = "Kostka & Almen 9th (2024)",
  burnstein_2025        = "Burstein & Straus 3rd (2025)",
  burnstein_straus_3ed  = "Burstein & Straus 3rd (2025)",
  hutchinson_mt21c_2025 = "Hutchinson (2025)",
  clendinning_marvin_5ed= "Clendinning & Marvin 5th (2026)"
)

prettify_book <- function(book_id) {
  out <- nlp_book_labels[book_id]
  ifelse(is.na(out), str_replace_all(book_id, "_", " ") %>% str_to_title(), out)
}

# Framing factor levels + named labels (corrective excluded from the headline
# plots: it has <10 instances corpus-wide and is too sparse for inference).
stat_framing_levels <- c("normative", "additive", "exceptional")
stat_framing_named  <- c(normative   = "Normative",
                       additive    = "Additive",
                       exceptional = "Exceptional")


# =============================================================================
# N15a.  PRIMARY PLOT — framing pattern is NOT uniform across books
# =============================================================================
#
# Claim supported (Results §"Framing"): "The corpus-level pattern is not
# uniform across books ... a chi-square test of homogeneity on the
# normative-versus-other split rejects a common rate."
#
# Design: one small-multiple panel per textbook. Within each panel, two bars
# (dominant, marginalized) are split 100% by framing category. A horizontal
# reference line marks the CORPUS-WIDE marginalized normative rate, so a reader
# can see at a glance which books sit above or below the pooled average — i.e.
# the non-uniformity itself. Books are ordered by their marginalized normative
# rate, making the spread the organising visual axis.
#
# Books with < 10 marginalized mentions (Aldwell) are dropped from this plot:
# their proportions are unstable and were excluded from the homogeneity test.

if (!is.null(con_db)) {

  # Pull per-book × status × framing counts straight from the framing table
  stat_framing_raw <- dbGetQuery(con_db, "
    SELECT book_id,
           CASE WHEN dominant='True'     THEN 'dominant'
                WHEN marginalized='True' THEN 'marginalized' END AS status,
           framing_category,
           COUNT(*) AS n
    FROM analysis_06_entity_framing_detail
    WHERE (dominant='True' OR marginalized='True')
      AND framing_category IN ('normative','additive','exceptional')
    GROUP BY book_id, status, framing_category
  ") %>% as_tibble()

  # Within-group percentages (per book × status)
  stat_framing <- stat_framing_raw %>%
    group_by(book_id, status) %>%
    mutate(group_total = sum(n),
           pct         = n / group_total) %>%
    ungroup() %>%
    mutate(
      framing_category = factor(framing_category, levels = stat_framing_levels),
      status           = factor(status, levels = c("dominant", "marginalized"))
    )

  # Books with >= 10 marginalized mentions (matches the homogeneity-test subset)
  stat_books_keep <- stat_framing %>%
    filter(status == "marginalized") %>%
    group_by(book_id) %>%
    summarise(marg_total = sum(n), .groups = "drop") %>%
    filter(marg_total >= 10) %>%
    pull(book_id)

  # Corpus-wide marginalized normative rate (pooled reference line)
  stat_corpus_marg_norm <- stat_framing %>%
    filter(status == "marginalized") %>%
    group_by(framing_category) %>%
    summarise(n = sum(n), .groups = "drop") %>%
    mutate(pct = n / sum(n)) %>%
    filter(framing_category == "normative") %>%
    pull(pct)

  # Order books by their marginalized normative rate (spread = the story)
  stat_book_order <- stat_framing %>%
    filter(status == "marginalized", framing_category == "normative",
           book_id %in% stat_books_keep) %>%
    arrange(pct) %>%
    pull(book_id)

  stat_framing_plot_df <- stat_framing %>%
    filter(book_id %in% stat_books_keep) %>%
    mutate(book_label = factor(prettify_book(book_id),
                               levels = prettify_book(stat_book_order)))

  plot_nlp_framing_nonuniform <- ggplot(
    stat_framing_plot_df,
    aes(x = status, y = pct, fill = framing_category)
  ) +
    geom_col(width = 0.7, colour = "white", linewidth = 0.3) +
    # Corpus-wide marginalized normative rate: the "uniform-world" expectation
    geom_hline(yintercept = stat_corpus_marg_norm,
               linetype = "dashed", colour = COL_DARK_GRAY, linewidth = 0.5) +
    facet_wrap(~ book_label, nrow = 2) +
    scale_fill_manual(values = framing_palette[stat_framing_levels],
                      labels = stat_framing_named,
                      name   = "Framing category",
                      guide  = guide_legend(reverse = TRUE)) +
    scale_x_discrete(labels = c(dominant = "Dom.", marginalized = "Marg.")) +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       expand = expansion(mult = c(0, 0.02))) +
    labs(
      title    = "Framing Is Not Uniform Across Textbooks",
      subtitle = paste0(
        "Within-group framing composition, dominant vs. marginalized, per book. ",
        "Dashed line = corpus-wide marginalized normative rate (",
        scales::percent(stat_corpus_marg_norm, accuracy = 1), "). ",
        "Books ordered by marginalized normative rate; chi-square homogeneity ",
        "p = .022. Books with < 10 marginalized mentions omitted."
      ),
      x = NULL,
      y = "% of group's mentions"
    ) +
    theme_clean() +
    theme(legend.position = "bottom",
          axis.text.x = element_text(size = 8),
          strip.text  = element_text(size = 8, face = "bold"),
          panel.spacing = unit(0.7, "lines"))

}


# =============================================================================
# N15b.  Framing gap — dominant vs marginalized, faceted by category
# =============================================================================
#
# Claim supported: the additive/normative gap exists corpus-wide but varies by
# book; reads from the precomputed analysis_13_framing_gap table.
#
# Design: a "dumbbell"-style plot. For each book and each framing category, a
# segment connects the dominant rate to the marginalized rate; the marginalized
# endpoint is amber, dominant is blue. A long segment = a large within-book gap.
# Faceting by framing category keeps the three contrasts visually separate.

if (!is.null(con_db)) {

  stat_gap <- dbGetQuery(con_db, "
    SELECT book_id,
           dom_normative,  marg_normative,
           dom_additive,   marg_additive,
           dom_exceptional, marg_exceptional,
           n_marg_mentions
    FROM analysis_13_framing_gap
  ") %>% as_tibble()

  stat_gap_long <- stat_gap %>%
    filter(as.numeric(n_marg_mentions) >= 10) %>%
    pivot_longer(
      cols = c(dom_normative, marg_normative,
               dom_additive, marg_additive,
               dom_exceptional, marg_exceptional),
      names_to  = c("status", "framing_category"),
      names_sep = "_",
      values_to = "rate"
    ) %>%
    mutate(
      rate             = as.numeric(rate),
      status           = recode(status, dom = "dominant", marg = "marginalized"),
      framing_category = factor(framing_category, levels = stat_framing_levels,
                                labels = stat_framing_named),
      book_label       = prettify_book(book_id)
    )

  # Order books by additive gap (most editorially interesting contrast)
  stat_gap_order <- stat_gap_long %>%
    filter(framing_category == "Additive") %>%
    select(book_label, status, rate) %>%
    pivot_wider(names_from = status, values_from = rate) %>%
    mutate(gap = marginalized - dominant) %>%
    arrange(gap) %>%
    pull(book_label)

  stat_gap_long <- stat_gap_long %>%
    mutate(book_label = factor(book_label, levels = stat_gap_order))

  plot_nlp_framing_gap_dumbbell <- ggplot(
    stat_gap_long,
    aes(x = rate, y = book_label)
  ) +
    geom_line(aes(group = book_label), colour = COL_GRAY, linewidth = 1) +
    geom_point(aes(colour = status), size = 3) +
    facet_wrap(~ framing_category, nrow = 1) +
    scale_colour_manual(
      values = c(dominant = COL_BLUE, marginalized = COL_AMBER),
      labels = c(dominant = "Dominant", marginalized = "Marginalized"),
      name   = NULL
    ) +
    scale_x_continuous(labels = percent_format(accuracy = 1)) +
    labs(
      title    = "Within-Book Framing Gaps by Category",
      subtitle = paste0(
        "Each segment connects the dominant rate (blue) to the marginalized ",
        "rate (amber) within a book. Longer segment = larger gap. ",
        "Books with < 10 marginalized mentions omitted."
      ),
      x = "% of group's mentions with this framing",
      y = NULL
    ) +
    theme_clean() +
    theme(legend.position = "bottom",
          axis.text.y = element_text(size = 8))

}


# =============================================================================
# N15c.  Structural placement by gender and BIPOC (separate axes)
# =============================================================================
#
# Claim supported (Results §"Gender and race as separate axes"): women and
# BIPOC composers skew supplementary and are near-absent from application
# material; the two axes are analysed separately, not via the collapsed binary.
#
# Source: normalized tables (passage_composers -> passages -> composers),
# matching framing_analysis.py Section 3.

if (!is.null(con_db)) {

  stat_placement <- dbGetQuery(con_db, "
    SELECT c.sex, c.bipoc, p.passage_role
    FROM passage_composers pc
    JOIN passages  p ON p.passage_id  = pc.passage_id
    JOIN composers c ON c.composer_id = pc.composer_id
    WHERE p.passage_role IN ('central','supplementary','application')
  ") %>% as_tibble()

  # Long format: one facet per axis (Gender, BIPOC), bars by category, fill by role
  stat_place_gender <- stat_placement %>%
    filter(sex %in% c("F", "M")) %>%
    mutate(axis = "Gender",
           group = recode(sex, F = "Women", M = "Men"))

  stat_place_bipoc <- stat_placement %>%
    filter(bipoc %in% c("Y", "N")) %>%
    mutate(axis = "Race",
           group = recode(bipoc, Y = "BIPOC", N = "Non-BIPOC"))

  stat_place_long <- bind_rows(stat_place_gender, stat_place_bipoc) %>%
    count(axis, group, passage_role, name = "n") %>%
    group_by(axis, group) %>%
    mutate(pct = n / sum(n)) %>%
    ungroup() %>%
    mutate(
      passage_role = factor(passage_role,
                            levels = c("central", "supplementary", "application")),
      group = factor(group, levels = c("Men", "Women", "Non-BIPOC", "BIPOC"))
    )

  plot_nlp_placement_axes <- ggplot(
    stat_place_long,
    aes(x = group, y = pct, fill = passage_role)
  ) +
    geom_col(width = 0.7, colour = "white", linewidth = 0.3) +
    geom_text(aes(label = ifelse(pct >= 0.04, percent(pct, accuracy = 1), "")),
              position = position_stack(vjust = 0.5),
              size = 3.0, colour = "white", fontface = "bold") +
    facet_wrap(~ axis, scales = "free_x") +
    scale_fill_manual(values = role_palette,
                      labels = c(central       = "Central",
                                 supplementary = "Supplementary",
                                 application   = "Application / exercises"),
                      name = "Passage role") +
    scale_y_continuous(labels = percent_format(accuracy = 1),
                       expand = expansion(mult = c(0, 0.02))) +
    labs(
      title    = "Structural Placement by Gender and Race",
      subtitle = paste0(
        "Each bar = 100% of that group's mentions. Women and BIPOC composers ",
        "skew supplementary and are near-absent from application material. ",
        "Both contrasts significant (chi-square p < .001)."
      ),
      x = NULL,
      y = "% of group's mentions"
    ) +
    theme_clean() +
    theme(legend.position = "bottom")

}


# =============================================================================
# N15d.  Integration vs breadth (volume is not integration)
# =============================================================================
#
# Claim supported (Results §"Integration by textbook"): volume, breadth, and
# integration are three different quantities; the books with the most distinct
# marginalized composers are not the best-integrated.
#
# Design: scatter. x = distinct marginalized composers (breadth),
# y = integration index (analysis_15). Point size = marginalized passage count
# (volume). A reader can see that high-breadth books (Laitz, Burstein, Gotham)
# do not cluster at high integration.

if (!is.null(con_db)) {

  stat_integ <- dbGetQuery(con_db, "
    SELECT book_id,
           CAST(n_marg_passages AS REAL) AS n_marg_passages,
           CAST(n_integrated    AS REAL) AS n_integrated,
           CAST(integration_pct AS REAL) AS integration_pct
    FROM analysis_15_integration_index
  ") %>% as_tibble()

  stat_breadth <- dbGetQuery(con_db, "
    SELECT book_id, COUNT(DISTINCT composer_canonical) AS n_unique_marg
    FROM analysis_06_entity_framing_detail
    WHERE marginalized='True'
    GROUP BY book_id
  ") %>% as_tibble()

  stat_integ_df <- stat_integ %>%
    left_join(stat_breadth, by = "book_id") %>%
    filter(n_marg_passages >= 5) %>%   # drop Aldwell (n=2): uninterpretable
    mutate(book_label = prettify_book(book_id))

  plot_nlp_integration_breadth <- ggplot(
    stat_integ_df,
    aes(x = n_unique_marg, y = integration_pct, size = n_marg_passages,
        label = book_label)
  ) +
    geom_point(colour = COL_BLUE, alpha = 0.8) +
    geom_text_repel(size = 3, max.overlaps = 15, show.legend = FALSE,
                    colour = COL_DARK_GRAY) +
    scale_size_continuous(name = "Marginalized\npassages", range = c(3, 12)) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(
      title    = "Breadth Is Not Integration",
      subtitle = paste0(
        "Distinct marginalized composers (x) vs. integration index (y); ",
        "point size = marginalized passage volume. High-breadth books do not ",
        "cluster at high integration. Aldwell (n = 2) omitted."
      ),
      x = "Distinct marginalized composers",
      y = "Integration index"
    ) +
    theme_clean()

}


# =============================================================================
# N16.  Print & save statistical-support figures
# =============================================================================

if (exists("plot_nlp_framing_nonuniform"))
  { print(plot_nlp_framing_nonuniform);
    save_nlp_plot(plot_nlp_framing_nonuniform, "N15a_framing_nonuniform.png",
                  width = 14, height = 8) }

if (exists("plot_nlp_framing_gap_dumbbell"))
  { print(plot_nlp_framing_gap_dumbbell);
    save_nlp_plot(plot_nlp_framing_gap_dumbbell, "N15b_framing_gap_dumbbell.png",
                  width = 14, height = 6) }

if (exists("plot_nlp_placement_axes"))
  { print(plot_nlp_placement_axes);
    save_nlp_plot(plot_nlp_placement_axes, "N15c_placement_axes.png",
                  width = 11, height = 6) }

if (exists("plot_nlp_integration_breadth"))
  { print(plot_nlp_integration_breadth);
    save_nlp_plot(plot_nlp_integration_breadth, "N15d_integration_breadth.png",
                  width = 11, height = 7) }

# Close the DB connection opened in N15
if (!is.null(con_db)) {
  dbDisconnect(con_db)
}

message("=== Statistical-support plots complete ===")
