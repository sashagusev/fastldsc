# Changelog

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
