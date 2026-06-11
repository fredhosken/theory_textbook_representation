"""
prepare_bio_csv.py
──────────────────
Build bio_data_processed.csv — the per-composer demographic and geographic
table consumed by analysis_server.py and build_db.py.

Per-field precedence (first non-empty wins):
    1. xlsx index   (composers_pieces_index.xlsx)  — optional; authoritative
                                                     for sex / BIPOC
    2. raw_df.csv                                  — primary source: geography,
                                                     demographics, dates
    3. country lookup CSV (optional)               — country only
    4. "Unknown"

Aliases: composer_aliases.csv (variant -> canonical) is applied to all
sources before matching, so spelling variants collapse to one row.

Output columns:
    composer, sex, bipoc, race, country, continent,
    latitude, longitude, born, died

Usage
─────
    python prepare_bio_csv.py
    python prepare_bio_csv.py --raw-df raw_df.csv --out bio_data_processed.csv
    python prepare_bio_csv.py --xlsx composers_pieces_index.xlsx --raw-df raw_df.csv
"""
from __future__ import annotations
import argparse, csv, re, sys, unicodedata
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────
RAW_DF_CSV   = Path("raw_df.csv")                       # primary source
OUTPUT_CSV   = Path("bio_data_processed.csv")
INPUT_XLSX   = Path("composers_pieces_index.xlsx")      # optional enrichment
ALIASES_CSV  = Path("composer_aliases.csv")
COUNTRY_CSV  = Path("composer_countries_lookup.csv")    # optional supplement

_COUNTRY_CONTINENT: dict[str, str] = {
    "Austria":"Europe","Germany":"Europe","France":"Europe","Italy":"Europe",
    "UK":"Europe","England (UK)":"Europe","England":"Europe","Scotland":"Europe",
    "United Kingdom":"Europe","Poland":"Europe","Russia":"Europe",
    "Ukraine":"Europe","Czech Republic":"Europe","Czechia":"Europe",
    "Hungary":"Europe","Spain":"Europe","Switzerland":"Europe","Norway":"Europe",
    "Iceland":"Europe","Estonia":"Europe","Finland":"Europe","Sweden":"Europe",
    "Denmark":"Europe","Belgium":"Europe","Netherlands":"Europe","Lithuania":"Europe",
    "Portugal":"Europe","Greece":"Europe","Romania":"Europe","Ireland":"Europe",
    "Croatia":"Europe","Slovakia":"Europe","Belarus":"Europe","Israel":"Asia",
    "USA":"North America","United States":"North America","Canada":"North America",
    "Brazil":"Latin America","Cuba":"Latin America","Puerto Rico":"Latin America",
    "Venezuela":"Latin America","Mexico":"Latin America","Argentina":"Latin America",
    "Colombia":"Latin America","Chile":"Latin America","Peru":"Latin America",
    "Bolivia":"Latin America","Uruguay":"Latin America","Panama":"Latin America",
    "Barbados":"Caribbean","Guadeloupe":"Caribbean","Martinique":"Caribbean",
    "Haiti":"Caribbean","Jamaica":"Caribbean","Trinidad and Tobago":"Caribbean",
    "Bahamas":"Caribbean",
    "Tanzania":"Africa","Nigeria":"Africa","Ghana":"Africa","South Africa":"Africa",
    "South Korea":"Asia","Japan":"Asia","China":"Asia","India":"Asia","Taiwan":"Asia",
    "Philippines":"Asia",
    "Australia":"Oceania","New Zealand":"Oceania",
}
_CARIB   = {"Barbados","Guadeloupe","Martinique","Haiti","Jamaica","Trinidad and Tobago","Bahamas"}
_LATIN   = {"Brazil","Cuba","Puerto Rico","Venezuela","Mexico","Argentina","Colombia","Chile","Peru","Bolivia","Uruguay","Panama"}
_AFRICAN = {"Tanzania","Nigeria","Ghana","South Africa"}
_ASIAN   = {"South Korea","Japan","China","India","Taiwan","Philippines"}

# Historical / variant country spellings -> modern canonical
_COUNTRY_CANON = {
    "England":"UK","Scotland":"UK","United Kingdom":"UK",
    "United Kingdom of Great Britain and Ireland":"UK","England (UK)":"UK",
    "United States":"USA","Czechia":"Czech Republic",
    "Holy Roman Empire":"Germany","Electorate of Saxony":"Germany",
    "Margraviate of Brandenburg":"Germany","Kingdom of Prussia":"Germany",
    "Duchy of Moscow":"Russia","Russian Empire":"Russia",
    "Kingdom of Italy":"Italy","Roman Empire":"Italy","Francia":"France",
    "Empire of Japan":"Japan","People's Republic of China":"China",
    "Polish People's Republic":"Poland","Polish People\u2019s Republic":"Poland",
    "France (Guadeloupe)":"Guadeloupe",
}


def _fold(s) -> str:
    if s is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower().strip()


def _clean_year(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in ("na", "nan", "none", "", "-"):
        return ""
    m = re.search(r"\b(1?\d{3})\b", s)
    return m.group(1) if m else ""


def _canon_country(c: str) -> str:
    c = (c or "").strip()
    return _COUNTRY_CANON.get(c, c)


def _continent_for(country: str) -> str:
    return _COUNTRY_CONTINENT.get(_canon_country(country), "Unknown")


def _race(bipoc: str, country: str, race_hint: str = "") -> str:
    rh = (race_hint or "").strip()
    if rh and rh not in ("Unknown", "NA", "nan", ""):
        return rh
    if bipoc != "Y":
        return "White"
    c = _canon_country(country)
    if c in _CARIB:   return "Black"
    if c in _LATIN:   return "Hispanic/Latino"
    if c in _AFRICAN: return "Black"
    if c in _ASIAN:   return "Asian"
    return "Black"


def _load_aliases(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with path.open(encoding=enc, newline="") as fh:
                for row in csv.DictReader(fh):
                    var = (row.get("variant") or "").strip()
                    canon = (row.get("canonical") or "").strip()
                    if var and canon:
                        out[_fold(var)] = canon
            print(f"[aliases] {len(out)} variant mappings from {path.name}")
            return out
        except Exception:
            continue
    return out


def _canonicalize(name: str, aliases: dict[str, str]) -> tuple[str, str]:
    """Return (canonical_display_name, folded_key)."""
    fk = _fold(name)
    canon = aliases.get(fk, name)
    return canon, _fold(canon)


def _load_rawdf(path: Path, aliases: dict[str, str]) -> dict[str, dict]:
    """folded(canonical) -> full bio record. Primary source."""
    out: dict[str, dict] = {}
    if not path.exists():
        sys.exit(f"ERROR: {path} not found — raw_df.csv is required.")
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            raw_name = r.get("composer_clean") or r.get("composer") or ""
            if not raw_name or raw_name.strip().lower() in ("anon", "anonymous"):
                continue
            canon, fk = _canonicalize(raw_name, aliases)
            rec = out.setdefault(fk, {"composer": canon})
            def setif(field, val):
                val = (str(val).strip() if val is not None else "")
                if val and val not in ("Unknown", "NA", "nan") and not rec.get(field):
                    rec[field] = val
            setif("country",   _canon_country(r.get("country", "")))
            setif("continent", r.get("continent", ""))
            setif("latitude",  r.get("latitude", ""))
            setif("longitude", r.get("longitude", ""))
            setif("race",      r.get("race", ""))
            setif("sex",       r.get("sex", ""))
            setif("bipoc",     r.get("bipoc", ""))
            setif("born",      _clean_year(r.get("born")))
            setif("died",      _clean_year(r.get("died")))
    print(f"[raw_df] {len(out)} unique composers from {path.name} (primary source)")
    return out


def _enrich_from_xlsx(records: dict[str, dict], xlsx_path: Path, aliases: dict[str, str]):
    """Add composers present in the xlsx index but missing from raw_df, and
    let the xlsx override sex / BIPOC (its authoritative columns)."""
    if not xlsx_path.exists():
        print(f"[xlsx] {xlsx_path.name} not found — skipping enrichment (raw_df-only)")
        return 0, 0
    try:
        import openpyxl
    except ImportError:
        print("[xlsx] openpyxl not installed — skipping enrichment")
        return 0, 0
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip().lower() if c else "" for c in rows[0]]
    def ci(n):
        try: return header.index(n.lower())
        except ValueError: return -1
    i_comp, i_born, i_died = ci("composer"), ci("born"), ci("died")
    i_sex, i_bipc = ci("sex"), ci("bipoc")
    if i_comp < 0:
        print("[xlsx] no 'Composer' column — skipping")
        return 0, 0
    added = overridden = 0
    for row in rows[1:]:
        name = str(row[i_comp]).strip() if row[i_comp] is not None else ""
        if not name or name.lower() in ("anon","anonymous","na","none","nan",""):
            continue
        canon, fk = _canonicalize(name, aliases)
        sex_x = str(row[i_sex]).strip().upper() if (i_sex>=0 and row[i_sex] is not None) else ""
        bip_x = str(row[i_bipc]).strip().upper() if (i_bipc>=0 and row[i_bipc] is not None) else ""
        if fk in records:
            rec = records[fk]
            if sex_x in ("M","F") and rec.get("sex") not in ("M","F"):
                rec["sex"] = sex_x; overridden += 1
            if bip_x.startswith("Y"):
                rec["bipoc"] = "Y"
            rec.setdefault("born", _clean_year(row[i_born] if i_born>=0 else None))
            rec.setdefault("died", _clean_year(row[i_died] if i_died>=0 else None))
        else:
            records[fk] = {
                "composer": canon,
                "sex": sex_x if sex_x in ("M","F") else "Unknown",
                "bipoc": "Y" if bip_x.startswith("Y") else "N",
                "born": _clean_year(row[i_born] if i_born>=0 else None),
                "died": _clean_year(row[i_died] if i_died>=0 else None),
            }
            added += 1
    print(f"[xlsx] enriched: +{added} new composers, {overridden} sex backfills")
    return added, overridden


def _load_country_sup(path: Path, aliases: dict[str, str]) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for enc in ("utf-8-sig","utf-8","latin-1"):
        try:
            with path.open(encoding=enc, newline="") as fh:
                for row in csv.DictReader(fh):
                    first = (row.get("Composer_First") or "").strip()
                    last  = (row.get("Composer_Last")  or "").strip()
                    country = _canon_country((row.get("Country") or "").strip())
                    if not first or not country: continue
                    canon = f"{first} {last}".strip() if last else first
                    _, fk = _canonicalize(canon, aliases)
                    out.setdefault(fk, {"country": country,
                                        "continent": _continent_for(country)})
            print(f"[country sup] {len(out)} from {path.name}")
            return out
        except Exception:
            continue
    return {}


def finalize(records: dict[str, dict], csup: dict[str, dict]) -> list[dict]:
    out = []
    for fk, rec in records.items():
        bipc = "Y" if (rec.get("bipoc","").upper().startswith("Y")) else "N"
        country = rec.get("country") or csup.get(fk,{}).get("country","") or "Unknown"
        country = _canon_country(country)
        continent = rec.get("continent") or csup.get(fk,{}).get("continent","")
        if not continent or continent == "Unknown":
            continent = _continent_for(country)
        sex = rec.get("sex","Unknown")
        if sex not in ("M","F"): sex = "Unknown"
        out.append({
            "composer": rec["composer"], "sex": sex, "bipoc": bipc,
            "race": _race(bipc, country, rec.get("race","")),
            "country": country, "continent": continent,
            "latitude": rec.get("latitude",""), "longitude": rec.get("longitude",""),
            "born": rec.get("born",""), "died": rec.get("died",""),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-df",  type=Path, default=RAW_DF_CSV)
    ap.add_argument("--out",     type=Path, default=OUTPUT_CSV)
    ap.add_argument("--xlsx",    type=Path, default=INPUT_XLSX)
    ap.add_argument("--aliases", type=Path, default=ALIASES_CSV)
    ap.add_argument("--country", type=Path, default=COUNTRY_CSV)
    a = ap.parse_args()

    aliases = _load_aliases(a.aliases)
    records = _load_rawdf(a.raw_df, aliases)
    _enrich_from_xlsx(records, a.xlsx, aliases)
    csup = _load_country_sup(a.country, aliases)
    rows = finalize(records, csup)

    n = len(rows)
    unk = sum(1 for r in rows if r["country"] == "Unknown")
    print(f"\nUnique composers: {n}")
    print(f"Female: {sum(1 for r in rows if r['sex']=='F')}  "
          f"BIPOC: {sum(1 for r in rows if r['bipoc']=='Y')}  "
          f"Marginalized: {sum(1 for r in rows if r['sex']=='F' or r['bipoc']=='Y')}")
    print(f"Country Unknown: {unk}/{n} ({100*unk/n:.1f}%)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["composer","sex","bipoc","race","country","continent",
              "latitude","longitude","born","died"]
    with a.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["composer"]):
            w.writerow(r)
    print(f"Written → {a.out}  ({n} composers)")


if __name__ == "__main__":
    main()
