# Changelog

## Unreleased

- `H2Result.tau_star` now takes `M_ref`, the number of reference SNPs
  (e.g. the `M_5_50` of baselineLD's all-ones `base` column).  It used
  `M_annot.sum()`, which with overlapping or continuous annotations counts
  SNPs many times over, so tau* came out about 18x too large on
  baselineLD v2.2 designs.
- `io.read_sumstats` drops rows whose SNP id repeats an earlier row, keeping
  the first, as ldsc's `_read_sumstats` does.  A sumstats file with a
  duplicated rsID (e.g. PASS BMI Yengo 2018) previously crashed `TraitData`
  with "cannot reindex on an axis with duplicate labels".
- `io.read_sumstats` splits on any whitespace, as ldsc does.  It split on
  tabs only, so space-delimited files (the `UKB_460K.*` traits of the
  Price-lab sumstats bundle) failed with "Usecols do not match columns".

## 0.1.0b1 (2026-09-29) — initial beta

First packaged release, assembled from research code used for a
perturbation-screen-to-GWAS project (August–September 2026).

- `fastldsc.sldsc`: arithmetic mirror of the partitioned-heritability path
  of `ldsc.py --h2` with all I/O amortised (`RefData` / `TraitData` /
  `fit_h2`).  Numerics unchanged from the validated research code
  (`fast_sldsc.py`); adds the ratio jackknife for `Prop._h2` and the
  overlap-based `Enrichment` table (`overlap_output`).
- `fastldsc.l2`: LD scores for many annotations in one genotype pass, with a
  pure-numpy plink `.bed` reader and an independent implementation of
  ldsc's chunked-window estimator (previously `fast_l2_multi.py`, which
  called ldsc's own library).
- `fastldsc.screen`: bordered / rank-one screening of thousands of candidate
  annotations per trait (`screen_regulons.py`), plus `check_against_refit`.
- `fastldsc.annot`: gene sets + coordinates (GTF or table) + window ->
  binary SNP annotations; BH mask, in-degree ranking and matched controls
  generalised from the `build_ldsc_sets*.py` scripts.
- `fastldsc.overlap`: cached baseline overlap matrix for enrichment across
  many models.
- `fastldsc.simulate`: synthetic reference panel / annotations / GWAS for
  tests and the quickstart.
- CLI: `annot`, `l2`, `overlap`, `h2`, `scan`, `screen`, `validate`,
  `parse-results`, `fdr`, `simulate`.
- All hard-coded paths of the research scripts replaced by arguments.
- License changed from MIT to GNU GPL v3 or later, matching ldsc (the LD-score chunk schedule follows ldsc's traversal).
