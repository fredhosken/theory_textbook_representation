# Data License: CC BY 4.0

The data files in the `data/` directory of this repository — including
the composer demographic table (`bio_data_processed.csv`), the composer
alias table (`composer_aliases.csv`), the concept vocabulary
(`concepts_music_theory.csv`), and the aggregate results tables — are
original curated datasets created by the author(s) and are licensed under
the Creative Commons Attribution 4.0 International License (CC BY 4.0).

You are free to:

- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any
  purpose, even commercially

Under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to
  the license, and indicate if changes were made. You may do so in any
  reasonable manner, but not in any way that suggests the licensor
  endorses you or your use.

To cite this dataset, please cite the accompanying article (see
`CITATION.cff`).

Full license text: https://creativecommons.org/licenses/by/4.0/legalcode

## Scope notes

- These data files contain no text from the analyzed textbooks. They are
  derived statistical summaries and independently curated reference
  tables (composer demographics, name variants, concept patterns).
- The textbooks analyzed in the accompanying article are copyrighted
  works and are **not** included in this repository in any form. The
  passage-level corpus (JSONL) and the assembled database (`textrep.db`)
  contain verbatim textbook text and therefore cannot be distributed;
  see the README for instructions on rebuilding them from independently
  obtained copies of the textbooks.
- The example data in `examples/` are synthetic, created solely to
  demonstrate the pipeline, and are covered by this same license.
