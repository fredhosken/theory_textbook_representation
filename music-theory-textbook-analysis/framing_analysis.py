#!/usr/bin/env python3
"""
framing_analysis.py
───────────────────
Statistical analysis of how dominant and marginalized composers are framed
and placed across the textbook corpus, drawing on textrep.db (built by
build_db.py). Produces the numbers underlying the Results section of the
accompanying article.

Five analytical questions, in order:
  1. Do dominant and marginalized composers receive different framing labels?
  2. Does that pattern vary significantly across textbooks?
  3. Does structural placement differ by gender and BIPOC status separately?
  4. How integrated are marginalized composers within each textbook?
  5. Which music-theory concepts are disproportionately associated with
     each group?

Data architecture
─────────────────
Two ID systems coexist in the database:

  analysis_06_entity_framing_detail — one row per composer mention; carries
      book_id, passage_role, framing_category, and group flags (dominant,
      marginalized, is_bipoc). Primary source for framing and placement.

  passages / passage_composers / composers — the normalized corpus layer.
      passage_composers.passage_id aligns with analysis_06.passage_id,
      enabling the concept-association analysis to bridge the two layers.

  Some books use different book_id strings in analysis_06 than in
  passages/textbooks; BOOK_ID_MAP translates wherever a join is needed.

Operationalization of groups
────────────────────────────
  dominant / marginalized: analysis_06 flags. Mentions classified as
      neither are excluded from all group contrasts.
  women / men: composers.sex = 'F' / 'M' (normalized layer).
  BIPOC / non-BIPOC: composers.bipoc = 'Y' / 'N' (normalized layer).
  The dominant/marginalized binary collapses gender and race into one flag;
  sections 3-4 treat them as separate axes.

Dependencies
────────────
  pip install numpy scipy statsmodels

Usage
─────
  python framing_analysis.py [textrep.db]
"""

import sqlite3
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

DB = sys.argv[1] if len(sys.argv) > 1 else "textrep.db"

# book_id translation: analysis_06 IDs -> passages/textbooks IDs, for
# databases built before BOOK_ID_MAP in build_db.py covered every book.
# Databases rebuilt with the current build_db.py carry JSONL stems throughout
# and need no translation. Must stay consistent with build_db.py.
BOOK_ID_MAP = {
    "aldwell_2019":        "aldwell_schachter_5ed",
    "gotham_2023":         "gotham_omt",
    "roig_francoli_2020":  "roig_francoli_3ed",
    "burnstein_2025":      "burnstein_straus_3ed",  # 'Burstein' is misspelled in the IDs; internally consistent
}

# Human-readable labels, keyed by analysis_06 book_id. Both ID namespaces are
# covered: server keys (pre-translation databases) and JSONL stems
# (databases rebuilt with the current build_db.py).
BOOK_LABELS = {
    "aldwell_2019":          "Aldwell & Schachter 5th (2019)",
    "aldwell_schachter_5ed": "Aldwell & Schachter 5th (2019)",
    "benward_saker_v1_10ed": "Benward & Saker 10th (2021)",
    "gotham_2023":           "Gotham, Open Music Theory (2023)",
    "gotham_omt":            "Gotham, Open Music Theory (2023)",
    "laitz_5ed":             "Laitz 5th (2023)",
    "kostka_almen_9ed":      "Kostka & Almén 9th (2024)",
    "hutchinson_mt21c_2025": "Hutchinson (2025)",
    "clendinning_marvin_5ed":"Clendinning & Marvin 5th (2026)",
    "mount_fff_2020":        "Mount (2020)",
    "roig_francoli_2020":    "Roig-Francolí 3rd (2020)",
    "roig_francoli_3ed":     "Roig-Francolí 3rd (2020)",
    "burnstein_2025":        "Burstein & Straus 3rd (2025)",
    "burnstein_straus_3ed":  "Burstein & Straus 3rd (2025)",
}

con = sqlite3.connect(DB)
cur = con.cursor()
A6 = "analysis_06_entity_framing_detail"


def q(sql, *args):
    return cur.execute(sql, args).fetchall()


def hr(title):
    bar = "─" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def cramers_v(chi2, n, k):
    """k = min(nrows-1, ncols-1)"""
    return np.sqrt(chi2 / (n * k))


# ── Corpus orientation ─────────────────────────────────────────────────────
hr("CORPUS")
n_books    = q(f"SELECT COUNT(DISTINCT book_id) FROM {A6}")[0][0]
n_passages = q("SELECT COUNT(*) FROM passages")[0][0]
n_comp     = q("SELECT COUNT(*) FROM composers")[0][0]
n_dom      = q(f"SELECT COUNT(*) FROM {A6} WHERE dominant='True'")[0][0]
n_marg     = q(f"SELECT COUNT(*) FROM {A6} WHERE marginalized='True'")[0][0]
n_neither  = q(f"SELECT COUNT(*) FROM {A6} WHERE dominant!='True' AND marginalized!='True'")[0][0]
n_total    = q(f"SELECT COUNT(*) FROM {A6}")[0][0]

print(f"Textbooks:          {n_books}")
print(f"Passages:           {n_passages:,}")
print(f"Composers in bio:   {n_comp}")
print(f"\nanalysis_06 mentions: {n_total:,}")
print(f"  dominant:           {n_dom:,}  ({100*n_dom/n_total:.1f}%)")
print(f"  marginalized:       {n_marg:,}  ({100*n_marg/n_total:.1f}%)")
print(f"  neither (excluded): {n_neither:,}  ({100*n_neither/n_total:.1f}%)")
print("\n[Framing and placement analyses below use only dominant and marginalized "
      "mentions; the 'neither' category is excluded from all group contrasts.]")


# ═══════════════════════════════════════════════════════════════════════════
hr("1. FRAMING DISTRIBUTION — dominant vs marginalized")
# ═══════════════════════════════════════════════════════════════════════════

CATS = ["normative", "additive", "exceptional", "corrective"]

contingency = []
for label, cond in [("dominant", "dominant='True'"),
                    ("marginalized", "marginalized='True'")]:
    row_total = cur.execute(
        f"SELECT COUNT(*) FROM {A6} WHERE {cond}").fetchone()[0]
    counts = {c: cur.execute(
        f"SELECT COUNT(*) FROM {A6} WHERE {cond} AND framing_category=?",
        (c,)).fetchone()[0] for c in CATS}
    contingency.append([counts[c] for c in CATS])
    print(f"\n{label}  (n = {row_total:,})")
    for c in CATS:
        print(f"   {c:12s}  {counts[c]:5d}   {100*counts[c]/row_total:5.1f}%")

print("\nMean classifier confidence by framing category:")
for cat, mean, n in q(f"""
        SELECT framing_category,
               ROUND(AVG(CAST(framing_confidence AS REAL)), 3),
               COUNT(*)
        FROM {A6}
        GROUP BY framing_category
        ORDER BY AVG(CAST(framing_confidence AS REAL)) DESC"""):
    print(f"   {cat:12s}  mean={mean}  n={n}")

# Corrective is too sparse for inference: excluded from the omnibus test,
# reported in the distribution above for transparency.
ct = np.array(contingency)
print(f"\n[corrective: {ct[:,3].sum()} total instances — excluded from "
      f"omnibus test, reported for transparency only]")

chi2, p, dof, _ = chi2_contingency(ct[:, :3])
n3 = ct[:, :3].sum()
v  = cramers_v(chi2, n3, 1)
print(f"\nOmnibus chi-square (normative / additive / exceptional × group):")
print(f"  chi2 = {chi2:.2f},  df = {dof},  p = {p:.3e},  Cramér's V = {v:.3f}")

# Targeted 2x2 Fisher tests isolate which categories drive the difference
print("\nTargeted Fisher's exact tests (category vs all others):")
for name, idx in [("normative", 0), ("additive", 1), ("exceptional", 2)]:
    col  = ct[:, idx]
    rest = np.delete(ct[:, :4], idx, axis=1).sum(1)
    odds, pf = fisher_exact([[col[0], rest[0]], [col[1], rest[1]]])
    sig = "**" if pf < 0.001 else ("*" if pf < 0.05 else "n.s.")
    print(f"   {name:12s}  OR(dom/marg) = {odds:.3f},  p = {pf:.3e}  {sig}")
    print(f"               dom: {100*col[0]/ct[0,:4].sum():.1f}%  "
          f"marg: {100*col[1]/ct[1,:4].sum():.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
hr("2. BETWEEN-BOOK FRAMING VARIATION — is the pattern uniform?")
# ═══════════════════════════════════════════════════════════════════════════
# Chi-square across books tests whether the marginalized normative-rate is
# homogeneous. Books with fewer than 10 marginalized mentions are excluded
# (expected-count assumption).

print("\nPer-book marginalized framing profile:")
print(f"{'Textbook':36s}  {'n':>5}  {'norm%':>6}  {'add%':>6}  "
      f"{'exc%':>6}  {'uniq':>5}")

btab_rows, btab_labels = [], []
for (bid,) in q(f"SELECT DISTINCT book_id FROM {A6} ORDER BY book_id"):
    tot = cur.execute(
        f"SELECT COUNT(*) FROM {A6} WHERE marginalized='True' AND book_id=?",
        (bid,)).fetchone()[0]
    if tot == 0:
        continue
    fr = dict(q(
        f"SELECT framing_category, COUNT(*) FROM {A6} "
        f"WHERE marginalized='True' AND book_id=? GROUP BY framing_category", bid))
    uniq = cur.execute(
        f"SELECT COUNT(DISTINCT composer_canonical) FROM {A6} "
        f"WHERE marginalized='True' AND book_id=?", (bid,)).fetchone()[0]
    label = BOOK_LABELS.get(bid, bid)
    n_n, n_a, n_e = (fr.get(c, 0) for c in ["normative","additive","exceptional"])
    print(f"{label:36s}  {tot:>5}  {100*n_n/tot:>6.1f}  {100*n_a/tot:>6.1f}  "
          f"{100*n_e/tot:>6.1f}  {uniq:>5}")
    if tot >= 10:
        btab_rows.append([fr.get("normative", 0), tot - fr.get("normative", 0)])
        btab_labels.append(label)

btab = np.array(btab_rows)
chi2b, pb, dofb, _ = chi2_contingency(btab)
print(f"\nBetween-book homogeneity test (marginalized normative vs other):")
print(f"  Books with ≥10 marginalized mentions: {len(btab_rows)}")
print(f"  chi2 = {chi2b:.1f},  df = {dofb},  p = {pb:.3e}")
print("  " + ("Significant variation across books." if pb < 0.05
               else "No significant between-book variation."))


# ═══════════════════════════════════════════════════════════════════════════
hr("3. STRUCTURAL PLACEMENT — gender and BIPOC as separate axes")
# ═══════════════════════════════════════════════════════════════════════════
# Source: normalized tables (passage_composers -> passages -> composers),
# independent of the analysis_06 pipeline classifications.

BASE = """FROM passage_composers pc
          JOIN passages  p ON p.passage_id  = pc.passage_id
          JOIN composers c ON c.composer_id = pc.composer_id"""
ROLES = ["central", "supplementary", "application"]

for axis, pairs, labels in [
    ("sex",   [("F","M")],  ("women",   "men")),
    ("bipoc", [("Y","N")],  ("BIPOC",   "non-BIPOC")),
]:
    a_val, b_val   = pairs[0]
    a_lab, b_lab   = labels
    print(f"\n── {axis.upper()} ──")
    table = []
    for val, lab in [(a_val, a_lab), (b_val, b_lab)]:
        tot = cur.execute(f"SELECT COUNT(*) {BASE} WHERE c.{axis}=?",
                          (val,)).fetchone()[0]
        row = [cur.execute(
            f"SELECT COUNT(*) {BASE} WHERE c.{axis}=? AND p.passage_role=?",
            (val, r)).fetchone()[0] for r in ROLES]
        table.append(row)
        pcts = "   ".join(f"{r}: {100*row[i]/tot:.1f}%" for i, r in enumerate(ROLES))
        print(f"   {lab:12s}  n={tot:5,}   {pcts}")
    table = np.array(table)
    chi2c, pc, dofc, _ = chi2_contingency(table)
    vc = cramers_v(chi2c, table.sum(), min(table.shape[0]-1, table.shape[1]-1))
    print(f"   chi2 = {chi2c:.1f},  df = {dofc},  p = {pc:.3e},  Cramér's V = {vc:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
hr("4. INTEGRATION AND PLACEMENT BY TEXTBOOK")
# ═══════════════════════════════════════════════════════════════════════════
# Integration = (passages with ≥1 marginalized AND ≥1 dominant mention)
#               / (passages with ≥1 marginalized mention)
# Source: analysis_06 flags, cross-checked against precomputed analysis_15.

print(f"\n{'Textbook':36s}  {'integ%':>7}  {'n_marg':>6}  {'uniq':>5}  "
      f"{'supp%':>6}  {'norm%':>6}  {'add%':>6}  {'exc%':>6}")

a15 = {r[0]: r for r in q(
    "SELECT book_id, n_marg_passages, n_integrated, integration_pct "
    "FROM analysis_15_integration_index")}

for (bid,) in q(f"SELECT DISTINCT book_id FROM {A6} ORDER BY "
                f"CAST((SELECT integration_pct FROM analysis_15_integration_index "
                f"WHERE book_id=analysis_06_entity_framing_detail.book_id) AS REAL) DESC"):
    tot = cur.execute(
        f"SELECT COUNT(*) FROM {A6} WHERE marginalized='True' AND book_id=?",
        (bid,)).fetchone()[0]
    if tot == 0:
        continue
    fr = dict(q(f"SELECT framing_category, COUNT(*) FROM {A6} "
                f"WHERE marginalized='True' AND book_id=? GROUP BY framing_category",
                bid))
    pr = dict(q(f"SELECT passage_role, COUNT(*) FROM {A6} "
                f"WHERE marginalized='True' AND book_id=? GROUP BY passage_role", bid))
    uniq = cur.execute(
        f"SELECT COUNT(DISTINCT composer_canonical) FROM {A6} "
        f"WHERE marginalized='True' AND book_id=?", (bid,)).fetchone()[0]
    integ_pct = float(a15[bid][3]) if bid in a15 else float("nan")
    label = BOOK_LABELS.get(bid, bid)

    g = lambda d, k: 100 * d.get(k, 0) / tot
    supp_pct = g(pr, "supplementary")
    norm_pct = g(fr, "normative")
    add_pct  = g(fr, "additive")
    exc_pct  = g(fr, "exceptional")

    note = ""
    if tot < 5:
        note = "  [n too small]"
    print(f"{label:36s}  {integ_pct:>7.1%}  {tot:>6}  {uniq:>5}  "
          f"{supp_pct:>6.1f}  {norm_pct:>6.1f}  {add_pct:>6.1f}  "
          f"{exc_pct:>6.1f}{note}")

print("\nCross-check: precomputed analysis_15_integration_index")
for bid, npass, ninteg, pct in q(
        "SELECT book_id, n_marg_passages, n_integrated, integration_pct "
        "FROM analysis_15_integration_index ORDER BY CAST(integration_pct AS REAL) DESC"):
    print(f"  {bid:28s}  stored={float(pct):.3f}  "
          f"(marg_passages={npass}, integrated={ninteg})")


# ═══════════════════════════════════════════════════════════════════════════
hr("5. CONCEPT ASSOCIATION — which topics accompany each group?")
# ═══════════════════════════════════════════════════════════════════════════
# For each concept, compare its rate in passages containing only
# dominant-group composers vs passages containing only marginalized-group
# composers ("pure" passages, to avoid confounding co-citation).
# Fisher's exact test per concept; Benjamini-Hochberg FDR correction.
# Concept tags include both 'passage' and 'chapter' level annotations.

pcon = q("SELECT pc.passage_id, co.name "
         "FROM passage_concepts pc JOIN concepts co ON co.concept_id = pc.concept_id")
cbp = defaultdict(set)
for pid, name in pcon:
    cbp[int(pid)].add(name)

prows = q(f"""
    SELECT passage_id,
           MAX(CASE WHEN dominant='True'     THEN 1 ELSE 0 END),
           MAX(CASE WHEN marginalized='True' THEN 1 ELSE 0 END)
    FROM {A6}
    GROUP BY passage_id
""")
dom_p  = {int(r[0]) for r in prows if r[1] == 1 and r[2] == 0}
marg_p = {int(r[0]) for r in prows if r[2] == 1 and r[1] == 0}
mixed  = {int(r[0]) for r in prows if r[1] == 1 and r[2] == 1}
nd, ng = len(dom_p), len(marg_p)
print(f"Pure dominant passages:     {nd}")
print(f"Pure marginalized passages: {ng}")
print(f"Mixed (excluded):           {len(mixed)}")

concepts = {c for s in cbp.values() for c in s}
res = []
for c in concepts:
    d = sum(1 for p in dom_p  if c in cbp.get(p, ()))
    g = sum(1 for p in marg_p if c in cbp.get(p, ()))
    if d + g < 5:       # minimum support to run the test
        continue
    _, pval = fisher_exact([[d, nd - d], [g, ng - g]])
    res.append(dict(concept=c, dom=d, marg=g,
                    dom_share=d/nd, marg_share=g/ng,
                    contrast=g/ng - d/nd, p_raw=pval))

pvals = [r["p_raw"] for r in res]
reject, padj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
for i, r in enumerate(res):
    r["p_fdr"] = padj[i]
    r["sig"]   = reject[i]

n_sig = sum(r["sig"] for r in res)
print(f"\nTestable concepts (support ≥ 5): {len(res)}")
print(f"Significant after BH FDR correction (α=0.05): {n_sig}")

print("\nMost marginalized-associated concepts  [+contrast = more common in marg. passages]")
print(f"  {'concept':26s}  {'dom':>5}  {'marg':>5}  {'contrast':>8}  p_FDR")
for r in sorted(res, key=lambda x: -x["contrast"])[:15]:
    star = "*" if r["sig"] else " "
    print(f"  {star} {r['concept']:24s}  {r['dom_share']:.3f}  {r['marg_share']:.3f}  "
          f"{r['contrast']:+.3f}    {r['p_fdr']:.4f}")

print("\nMost dominant-associated concepts  [-contrast = more common in dom. passages]")
print(f"  {'concept':26s}  {'dom':>5}  {'marg':>5}  {'contrast':>8}  p_FDR")
for r in sorted(res, key=lambda x: x["contrast"])[:15]:
    star = "*" if r["sig"] else " "
    print(f"  {star} {r['concept']:24s}  {r['dom_share']:.3f}  {r['marg_share']:.3f}  "
          f"{r['contrast']:+.3f}    {r['p_fdr']:.4f}")

# BIPOC-specific concept concentration: is any concept cluster specific to
# BIPOC composers rather than the broader marginalized group?
print("\n── BIPOC-specific concept concentration ──")

brows = q(f"""
    SELECT passage_id,
           MAX(CASE WHEN is_bipoc='True'  THEN 1 ELSE 0 END),
           MAX(CASE WHEN is_bipoc='False' THEN 1 ELSE 0 END)
    FROM {A6}
    GROUP BY passage_id
""")
bip_p = {int(r[0]) for r in brows if r[1] == 1 and r[2] == 0}
non_p = {int(r[0]) for r in brows if r[2] == 1 and r[1] == 0}
nb, nn = len(bip_p), len(non_p)
print(f"\n  Pure BIPOC passages:     {nb}")
print(f"  Pure non-BIPOC passages: {nn}")

bres = []
for c in concepts:
    d = sum(1 for p in bip_p if c in cbp.get(p, ()))
    g = sum(1 for p in non_p if c in cbp.get(p, ()))
    if d + g < 5:
        continue
    _, pval = fisher_exact([[d, nb - d], [g, nn - g]])
    bres.append(dict(concept=c, bipoc=d, non=g,
                     bipoc_share=d/nb, non_share=g/nn,
                     contrast=d/nb - g/nn, p_raw=pval))

bpv = [r["p_raw"] for r in bres]
brej, bpadj, _, _ = multipletests(bpv, method="fdr_bh")
for i, r in enumerate(bres):
    r["p_fdr"] = bpadj[i]; r["sig"] = brej[i]

print("\n  Top BIPOC-associated concepts:")
for r in sorted(bres, key=lambda x: -x["contrast"])[:12]:
    star = "*" if r["sig"] else " "
    print(f"  {star} {r['concept']:26s}  bipoc={r['bipoc_share']:.3f}  "
          f"non={r['non_share']:.3f}  contrast={r['contrast']:+.3f}  p_FDR={r['p_fdr']:.4f}")

con.close()
print("\n── Done ──")
