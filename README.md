# fastldsc

**Fast stratified LD score regression for scanning many annotations.**
Status: **beta (0.1.0b1)**.

`fastldsc` is an independent Python re-implementation of the
partitioned-heritability path of [ldsc](https://github.com/bulik/ldsc)
(`ldsc.py --h2 --ref-ld-chr ... --overlap-annot`; Bulik-Sullivan et al.
2015, Finucane et al. 2015), built for the situation where one wants to test
*hundreds to tens of thousands* of candidate annotations (gene sets, regulons,
programs, cell-state signatures) against many traits on top of a fixed
baseline model.  Stock ldsc re-reads and re-merges every reference file for
every model and recomputes every pairwise r² for every annotation file; here
that work is done once.

Three pieces:

| module | what it does | exact? |
|---|---|---|
| `fastldsc.sldsc` (`fastldsc h2` / `scan` / `validate`) | the S-LDSC regression with reference LD scores, weights and candidate columns loaded once per process and each trait merged once | **exact** arithmetic mirror of `ldsc.py --h2` for n_annot > 1 (see validation) |
| `fastldsc.l2` (`fastldsc l2`) | LD scores for *many* annotations in *one* genotype pass: the annotation only enters the windowed r² sum as a matrix product | **exact** mirror of `ldsc.py --l2 --ld-wind-cm` (see validation) |
| `fastldsc.screen` (`fastldsc screen`) | thousands of candidate annotations per trait through bordered (rank-one) solves of the fixed baseline design, including all 200 jackknife delete values | **approximate**: the regression weights are held at the baseline-design values (measured error below) |

Plus `fastldsc annot` (gene sets + coordinates + window -> `.annot.gz`),
`fastldsc overlap` (cached baseline overlap matrix for enrichment),
`fastldsc fdr` (empirical FDR from negative-control traits),
`fastldsc parse-results`, and `fastldsc simulate` (synthetic data for the
quickstart and the tests).

## What "exact mirror" means

For a design with more than one annotation, `ldsc.py --h2` does the
following (`ldscore/sumstats.py: estimate_h2`, `ldscore/regressions.py`):

1. inner-merge sumstats -> reference LD scores -> weight LD scores on SNP,
   keeping the reference (chromosome, position) order;
2. drop SNPs with `chisq >= max(0.001 * N.max(), 80)`;
3. compute an aggregate heritability
   `h2_agg = M_tot * (mean(chisq) - 1) / mean(l_tot * N)` and the weights
   `w = 1 / (2 (1 + h2_agg N l_tot / M_tot)^2) / w_ld` with `l_tot` and `w_ld`
   floored at 1;
4. scale the annotation columns by `N / mean(N)` and append an intercept;
5. **one** weighted least squares fit.  For n_annot > 1 ldsc sets
   `old_weights=True`, which routes the fit through `LstsqJackknifeFast`
   with the square root of those initial weights.  There is **no
   iteratively re-weighted least squares** in the partitioned path (ldsc's
   IRWLS is used only when n_annot == 1);
6. a 200-block delete-one-block jackknife over block-wise `X'WX`, `X'Wy`;
   a ratio jackknife for `Prop._h2`; `Enrichment` and `Enrichment_p` from the
   annotation overlap matrix over SNPs with 0.05 < MAF < 0.95.

`fastldsc.sldsc.fit_h2` does exactly that, with the same floating-point
operations in the same order for the regression (the research code it comes
from, `fast_sldsc.py`, was written as a line-by-line arithmetic mirror and
its numerics are unchanged here; only I/O and argument handling were
refactored).  What differs is *when* things are read: `RefData` holds the
baseline, the weights and any number of candidate columns in memory in
reference order; `TraitData` merges a trait once; a model is then a few
matrix products.

For LD scores, `fastldsc.l2` reproduces ldsc's estimator: per-SNP
standardisation with mean-imputed missing genotypes, `r2 - (1 - r2)/(n - 2)`,
ldsc's SNP filter (drop MAF = 0 and all-heterozygous/missing SNPs), `M` and
`M_5_50`, and ldsc's chunked window schedule.  Note that ldsc rounds the
left edge of each 50-SNP chunk's window *up to a whole chunk*, evaluated at
the chunk's first SNP, so its effective window is slightly wider than the
nominal `--ld-wind-cm`; `fastldsc` reproduces this so that its output is
numerically identical, and `--chunk-size 1` gives the strict window.  The
plink `.bed` reader is pure numpy (no `bitarray`).

## Validation (measured)

All numbers below were measured on the machine holding the banked stock-ldsc
runs (ldsc python-3 port reporting itself as version 2.0.0, numpy 2.0.2) on 2026-09-29 with the
1000G Phase 3 EUR / baselineLD v2.2 references, and are produced by
`fastldsc validate` and the commands in the benchmark section.  Nothing here
is extrapolated.

### Regression (`fastldsc h2` vs banked `ldsc.py --h2 --overlap-annot`)

A random sample (seed 3) of **100** banked runs out of 1,883 (each: 97
baselineLD v2.2 annotations + 1 or 2 gene-set annotations, 21 traits,
490k–1.1M regression SNPs).  Every row of every `.results` file was compared
(98–99 categories per run), plus total h2 and intercept from the `.log`:

| quantity (over all 98–99 categories x 100 runs) | max abs Δ | max relative Δ |
|---|---|---|
| Coefficient (τ) | 2.0e-15 | 7.1e-7 (a near-zero coefficient; median run 1.8e-8) |
| Coefficient_std_error | — | 4.6e-9 |
| Coefficient z-score | **1.9e-8** | — |
| Prop._h2 | 2.4e-9 | 1.4e-7 |
| Prop._h2_std_error | 4.1e-11 (worst run) | 2.1e-10 (worst run) |
| Prop._SNPs | 2.2e-16 | — |
| Enrichment | see note | 1.4e-7 |
| Enrichment_std_error | see note | 4.3e-11 (worst run) |
| Enrichment_p | 9.2e-9 | — |
| total h2, h2 SE, intercept, intercept SE (log, 4 decimals) | 5.0e-5 | — |
| number of regression SNPs after the chi-square filter | identical in 100/100 | |

Note on Enrichment: baselineLD v2.2 contains continuous annotations whose
`M_5_50` is tiny, for which ldsc's Enrichment is ~1e13 and meaningless; the
absolute deviations there (up to ~1e4) are 5e-11 relative.  The
Enrichment/Enrichment_SE relative maxima above are with the denominator
floored at 1e-10 of the column's scale (ldsc's SE of the all-SNP base
category is exactly 0 in both tools); the "worst run" entries are from a
direct re-check of the run with the largest deviation.  The 100 runs
comprise 42 two-file models (baseline + 1 set) and 58 three-file models
(baseline + 2 sets, e.g. conditional and joint runs); 21 distinct traits.
Re-running stock ldsc itself on one of the banked models reproduced the
banked coefficients only to 2.6e-18 absolute (BLAS summation order), i.e.
the fastldsc deviations are at the same floor as ldsc's own run-to-run noise
for coefficients and SEs.

The "z" column is the coefficient z-score (`Coefficient / Coefficient_std_error`),
the quantity that drives any downstream decision.  An earlier 30-run
validation of the research code (recorded in its output on the same box)
gave max rel |Δcoef| 3.6e-8, max rel |ΔSE| 3.3e-9, max |Δz| 1.7e-8.
Deviations of this size are floating-point summation-order noise
(`np.linalg.solve` vs. ldsc's, block sums), not method differences.

### LD scores (`fastldsc l2` vs banked `ldsc.py --l2`)

Chromosome 22 of the 1000G EUR panel (489 individuals, 141,123 SNPs,
`--ld-wind-cm 1`, `--print-snps` HapMap3), one binary gene-set annotation
(22,598 SNPs):

* **max |ΔL2| = 0.0000** over the 17,489 printed SNPs (ldsc writes three
  decimals; every value identical), `.l2.M` and `.l2.M_5_50` identical.
* The gene-set -> annotation step (`--sets` + coordinates + ±100 kb) reproduced
  the banked `.annot.gz` SNP-for-SNP (0 of 141,123 differ).
* The first column of a 600-annotation one-pass run is bit-identical to the
  same annotation run alone.

### Screening (`fastldsc screen` vs exact refit)

Two real bundles from the research project, baseline (97 columns) + one
global gene set held fixed, 4 traits (two autoimmune, one blood count, BMI),
every candidate screened and a random subset re-fitted exactly:

| bundle | refits | max abs Δz | 99th pct abs Δz | median abs Δz | max abs Δτ / SE | max rel ΔSE |
|---|---|---|---|---|---|---|
| `mod` (281 module annotations) | 160 | **0.069** | 0.040 | 0.005 | 0.045 | 2.2 % |
| `prog` (18 program annotations) | 72 | **0.053** | 0.050 | 0.008 | 0.059 | 1.1 % |

(Relative Δτ is not a useful metric for a screen: a coefficient that is
0 ± 1 SE has an unbounded relative error; the Δτ/SE column is the one
that matters.)

The screen holds the regression weights at the baseline design; the refit
recomputes them with the candidate included in the total LD score.  The
research code recorded max |Δz| = 0.047 on its regulon scan (thousands of
candidates); on the sets above the deviation is of the same order.  Use
`--check N` on your own data: it refits N random candidates per trait and
writes `<out>.check.csv`.

## Benchmarks (measured, same machine, `OMP_NUM_THREADS=8`)

| task | stock ldsc | fastldsc | notes |
|---|---|---|---|
| one S-LDSC model (97 baselineLD + 1 set, PASS_Celiac, `--overlap-annot`), wall clock including all I/O | **117 s** (`--frqfile-chr`; the banked log of the same job: 118 s) | **21 s** with `--overlap-cache`; 93 s with `--frqfile-chr` (reading 22 annot files dominates) | fastldsc regression itself: **1.8 s**; the rest is reading the references once |
| additional models in the same process (`scan` / `validate`, references already loaded) | 117 s each | **3.8 s mean, 4.9 s max** per model (100-run validation, 98–99 columns, 0.5–1.1M SNPs, 8 BLAS threads) | ~30x |
| LD scores, chr22, 1 binary annotation (`--ld-wind-cm 1`, HapMap3 print SNPs) | **17.3 s** | **10.1 s** | identical output |
| LD scores, chr22, 600 binary annotations in one pass | 600 x 17 s ≈ 2.9 h (one pass per annotation) | **33 s** (27 s compute; 2.2 GB peak) | first column bit-identical to the single-annotation run |
| baseline overlap cache (`fastldsc overlap`, 22 chromosomes, 97 annotations) | — | 73 s once (5.4 GB peak, writes 2.3 GB) | then each candidate's enrichment row is one matrix-vector product |
| screening (`fastldsc screen`), 281 candidates x 4 traits incl. loading a 1.3 GB bundle and 160 check refits | — | 616 s total; the screen itself is ≈ 0.2 s per candidate-trait | |

Machine: 16-vCPU EC2 instance shared with another job; both tools limited
to 8 OpenBLAS threads; stock ldsc = the python-3 port of ldsc 2.0.

## Install

```
pip install .            # from a clone; Python >= 3.9, numpy, scipy, pandas
pip install -e ".[test]" && pytest    # tests: synthetic data only, < 10 s
```

## Quickstart (synthetic, no downloads)

`examples/quickstart.sh` runs the whole pipeline in a few seconds:

```
fastldsc simulate --out qs --n-snp 800 --n-indiv 80 --chr 1-2 --n-traits 3
# plink/sim.<chr>.{bed,bim,fam,frq}, baseline/base.<chr>.l2.ldscore.gz (+ .annot.gz, .l2.M_5_50),
# weights/w.<chr>.l2.ldscore.gz, sets/*.GeneSet, genes.coords, sumstats/*.sumstats.gz

# gene sets -> LD scores for all sets in one pass (ldsc-format files + a .npy bundle)
fastldsc l2 --bfile qs/plink/sim. --sets qs/sets --coords qs/genes.coords --window 20000 \
    --print-snps qs/print_snps.txt --chr 1-2 --out qs/sets_l2/sets. \
    --bundle-dir qs/sets_l2 --tag sets --overlap-with qs/baseline/base.

# one model, ldsc style
fastldsc h2 --h2 qs/sumstats/trait0.sumstats.gz --ref-ld-chr qs/baseline/base.,qs/sets_l2/sets. \
    --w-ld-chr qs/weights/w. --chr 1-2 --overlap-annot --frqfile-chr qs/plink/sim. --out qs/h2_trait0

# every set x every trait (marginal + conditional on set5)
fastldsc scan --ref-ld-chr qs/baseline/base. --w-ld-chr qs/weights/w. --chr 1-2 \
    --traits trait0,trait1,trait2,control0,control1,control2 --sumstats-dir qs/sumstats \
    --bundle-dir qs/sets_l2 --tags sets --cond set5 --out qs/scan.csv

# the bordered-solve screen, with 6 candidates re-checked by exact refit
fastldsc screen --ref-ld-chr qs/baseline/base. --w-ld-chr qs/weights/w. --chr 1-2 \
    --traits trait0,trait1,trait2,control0,control1,control2 --sumstats-dir qs/sumstats \
    --bundle-dir qs/sets_l2 --tags sets --check 6 --out qs/screen.csv

# empirical FDR from the control traits
fastldsc fdr --results qs/screen.csv --control-traits control0,control1,control2 --z-col z \
    --out qs/screen_fdr.csv
```

## Using the standard references (1000G Phase 3 EUR, baselineLD v2.2)

`fastldsc` reads exactly the files ldsc reads.  From the Alkes Price group
data page (<https://alkesgroup.broadinstitute.org/LDSCORE/>, GRCh37; see the
ldsc wiki page *Partitioned Heritability* for the current layout):

| what | file | `fastldsc` argument |
|---|---|---|
| baseline LD scores + annotations | `1000G_Phase3_baselineLD_v2.2_ldscores.tgz` -> `baselineLD.<chr>.l2.ldscore.gz`, `.l2.M_5_50`, `.annot.gz` | `--ref-ld-chr .../baselineLD.` ; `overlap --annot-chr .../baselineLD.` |
| regression weights | `1000G_Phase3_weights_hm3_no_MHC.tgz` -> `weights.hm3_noMHC.<chr>.l2.ldscore.gz` | `--w-ld-chr .../weights.hm3_noMHC.` |
| allele frequencies | `1000G_Phase3_frq.tgz` -> `1000G.EUR.QC.<chr>.frq` | `--frqfile-chr .../1000G.EUR.QC.` |
| genotypes (for `l2`) | `1000G_Phase3_plinkfiles.tgz` -> `1000G.EUR.QC.<chr>.{bed,bim,fam}` | `l2 --bfile .../1000G.EUR.QC.` |
| regression SNP list | `hm3_no_MHC.list.txt` (or `w_hm3.snplist`) | `l2 --print-snps` |
| gene coordinates | any GTF (e.g. GENCODE v19 for GRCh37) | `--gtf` or a `GENE CHR START END` table via `--coords` |
| summary statistics | `munge_sumstats.py` output (`SNP A1 A2 N Z`) | `--h2` / `--traits` + `--sumstats-dir` |

A typical real run:

```
REF=/path/to/ldsc_ref
BASE=$REF/baselineLD_v2.2/baselineLD.
W=$REF/weights_hm3_no_MHC/weights.hm3_noMHC.
FRQ=$REF/1000G_Phase3_frq/1000G.EUR.QC.
PLINK=$REF/1000G_EUR_Phase3_plink/1000G.EUR.QC.

# gene coordinates once
fastldsc annot --gtf gencode.v19.annotation.gtf.gz --write-coords ENSG_coords.txt \
    --sets my_sets/ --bfile $PLINK --out /dev/null --chr 22   # (or just use --coords later)

# LD scores for all candidate sets in one pass over the genome (about the cost of
# ONE stock ldsc annotation per chromosome for hundreds of annotations)
fastldsc l2 --bfile $PLINK --sets my_sets/ --coords ENSG_coords.txt --window 100000 \
    --exclude-mhc --print-snps $REF/hm3_no_MHC.list.txt \
    --bundle-dir l2_bundles --tag mysets --overlap-with $BASE

# baseline overlap once (for Enrichment)
fastldsc overlap --annot-chr $BASE --frqfile-chr $FRQ --out base_overlap

# scan: marginal and conditional models, results in one CSV
fastldsc scan --ref-ld-chr $BASE --w-ld-chr $W --traits PASS_Celiac,PASS_Crohns_Disease,... \
    --sumstats-dir $REF/sumstats --bundle-dir l2_bundles --tags mysets --cond my_main_set \
    --l2-dir ldsc_l2 --out scan.csv

# screen thousands of candidates per trait
fastldsc screen --ref-ld-chr $BASE --w-ld-chr $W --traits ... --sumstats-dir $REF/sumstats \
    --bundle-dir l2_bundles --tags mysets --global-sets my_main_set --l2-dir ldsc_l2 \
    --check 40 --out screen.csv
```

Memory: `RefData` holds the baseline (1.19M SNPs x 97 columns, float64,
~0.9 GB) plus one float32 column per candidate (4.8 MB each); the `l2`
bundle for S annotations is 4.8 MB x S.  The `overlap` cache stores the
common-SNP baseline annotation matrix as float32 (~2.3 GB for baselineLD v2.2)
so that each candidate's enrichment row is one matrix-vector product.

## CLI reference

```
fastldsc simulate --out DIR [--n-snp N --n-indiv N --chr 1-2 --n-traits K --seed S]
fastldsc annot    --sets FILES|DIR|NPZ (--coords TABLE | --gtf GTF [--write-coords F])
                  --bfile PREFIX --out PREFIX [--window BP --chr LIST --full --add-base
                  --exclude-mhc --write-sets-npz F]
fastldsc l2       --bfile PREFIX (--annot PREFIX | --sets ... --coords/--gtf [--window BP]) 
                  [--ld-wind-cm 1.0 --chunk-size 50 --print-snps F --chr LIST]
                  [--out PREFIX] [--bundle-dir DIR --tag TAG] [--overlap-with ANNOT_PREFIX]
                  [--exclude-mhc --add-base --no-annot]
fastldsc overlap  --annot-chr PREFIX [--frqfile-chr PREFIX] --out PREFIX [--chr LIST --no-annot]
fastldsc h2       --h2 SUMSTATS --ref-ld-chr P1[,P2,...] --w-ld-chr P --out PREFIX
                  [--overlap-annot (--frqfile-chr P | --overlap-cache F.npz)]
                  [--chisq-max X --n-blocks 200 --chr LIST --not-M-5-50]
fastldsc scan     --ref-ld-chr P --w-ld-chr P --traits LIST [--sumstats-dir D]
                  (--sets LIST --l2-dir D | --bundle-dir D --tags LIST [--snps-file F])
                  [--cond LIST --set-filter STR --n-blocks 200] --out CSV
fastldsc screen   --ref-ld-chr P --w-ld-chr P --traits LIST [--sumstats-dir D]
                  --bundle-dir D --tags LIST [--global-sets LIST --l2-dir D]
                  [--check N --chunk 250 --set-filter STR] --out CSV
fastldsc validate --ldsc-results DIR|GLOB --ref-ld-chr P --w-ld-chr P --out CSV
                  [--n 40 --seed 3 --pattern REGEX --l2-dir D --sumstats-dir D --overlap-cache F]
fastldsc parse-results --h2-dir DIR --out CSV [--last-only K]
fastldsc fdr      --results CSV --control-traits LIST --out CSV [--z-col z --trait-col trait --two-sided]
```

Per-chromosome prefixes follow ldsc: the chromosome number is appended
(`baselineLD.` -> `baselineLD.22`), or substituted for `@`.

Outputs: `h2` writes `<out>.results` (ldsc's columns) and `<out>.log`;
`scan` writes one row per (set, trait) with `marg_*`, and `cond_*` when
`--cond` is given (plus `z_<cond set>` for the conditioning terms);
`screen` writes `set, trait, coef, se, z, M` and, with `--check`,
`<out>.check.csv`; `validate` writes per-run maxima of every deviation.

The bundle written by `l2 --bundle-dir` is `l2_<tag>.npy` (printed SNPs x
annotations, float32), `M_<tag>.npy` (M_5_50), `names_<tag>.npy`,
`overlap_<tag>.npy` (if `--overlap-with`) and `snps_all.npy` /
`snps_chr<list>.npy` (row order).  `scan`/`screen` align it to the reference
by SNP id and refuse bundles that cover < 90 % of the reference.

## Calibration warning: do not trust nominal p-values for sparse annotations

The most important lesson from the project this code comes from: **nominal
conditional τ\* z-scores (even robustly standardised ones) are not calibrated
when many sparse annotations are screened.**  A gene-set annotation covering
a few thousand SNPs gives a 200-block jackknife standard error that is
unstable — a handful of blocks carry most of the annotation — and the
resulting z distribution has heavy, one-sided tails under the null.  Ranking
thousands of candidates by nominal p, or applying Benjamini–Hochberg to
them, produces "discoveries" that do not replicate.

The remedy built into the workflow is empirical: **always carry
negative-control traits** through the same scan — traits that share nothing
with the biology of the annotations (e.g. height, BMI, tanning,
chronotype for immune-cell gene sets) but have the same sample sizes and
LD structure.  Their z-scores for the same annotations form an empirical
null with the right sparsity and jackknife behaviour.  `fastldsc fdr`
implements the recipe:

1. run `scan` or `screen` with the test traits **and** the control traits;
2. `fastldsc fdr --results scan.csv --control-traits ctrl1,ctrl2,... --z-col cond_z`
   computes, for every (annotation, test trait), the empirical one-sided p
   `(#control z >= z + 1) / (#control z + 1)`, and Benjamini–Hochberg q-values
   from those; it also prints the control-trait mean/SD of z (nominally 0/1)
   and the empirical 95th percentile of |z|;
3. report annotations by empirical q, never by nominal p.

Four control traits x a few hundred annotations already give a usable null;
more controls tighten the smallest attainable p.  If the control z's have
SD well above 1, the nominal SEs are too small for that annotation family
and the empirical threshold is the one to use.

## Relation to ldsc, license, citation

`ldsc` (Bulik-Sullivan, Loh, Finucane et al., *Nat Genet* 2015; Finucane,
Bulik-Sullivan, Gusev et al., *Nat Genet* 2015) is GPL-3.0.  `fastldsc` is an
**independent re-implementation**, released under the same license as ldsc
(GNU GPL v3 or later; see `LICENSE`); it contains no ldsc source code.  The algorithms it mirrors are those described
in the papers and documented in ldsc's own code, which was read to establish
the exact arithmetic (weights, chi-square cap, `old_weights`, jackknife,
overlap output).  The LD-score chunk schedule in `fastldsc.l2` necessarily
follows the same traversal as ldsc's estimator because the numbers depend on
it; it was written from scratch.  Please cite the two ldsc papers for the
method, and this repository for the implementation.

## Beta status and limitations

* Only the partitioned (`n_annot > 1`, free intercept) h2 path is
  implemented: no `--two-step` single-annotation h2, no constrained
  intercept, no `rg`, no `--h2-cts`, no liability-scale conversion.
* Enrichment (`--overlap-annot`) needs the annotation files of every
  column in the design, as in ldsc; for bundles built with `l2 --overlap-with`
  the rows are stored in the bundle.
* `l2` computes `--ld-wind-cm` / `--ld-wind-kb` / `--ld-wind-snps` windows
  and the standard `L2`; no `--pq-exp`, `--per-allele`, `--cts-bin` or
  block-jackknife SEs of LD scores.
* Everything is in-memory per chromosome (`l2`) or per process (`h2`/`scan`):
  the genome-wide baseline needs ~1–1.5 GB; bundles scale with the number of
  annotations.
* The screen's approximation error grows with the candidate's share of the
  total LD score; on a 97-column baseline it is small (table above), on a
  tiny baseline it is not.  Use `--check`.
* The validation covers one reference set (1000G Phase 3 EUR, baselineLD
  v2.2, HapMap3 regression SNPs) and one ldsc version (2.0.0 python-3 port).
