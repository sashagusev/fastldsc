#!/usr/bin/env bash
# End-to-end run of fastldsc on a small synthetic dataset (no downloads).
# Runs in well under a minute on a laptop.
set -euo pipefail
OUT=${1:-quickstart_out}
rm -rf "$OUT"

# 1. synthetic reference panel, baseline LD scores, weights, gene sets, GWAS
fastldsc simulate --out "$OUT" --n-snp 800 --n-indiv 80 --chr 1-2 --n-traits 3

BFILE=$OUT/plink/sim.          # per-chromosome plink prefix (chromosome appended)
BASE=$OUT/baseline/base.       # baseline LD scores (.l2.ldscore.gz / .l2.M_5_50 / .annot.gz)
W=$OUT/weights/w.              # regression-weight LD scores
FRQ=$OUT/plink/sim.            # plink .frq files (for --overlap-annot)
TRAITS=trait0,trait1,trait2
CONTROLS=$(paste -sd, "$OUT/control_traits.txt")

# 2. gene sets -> LD scores for all sets in ONE genotype pass
#    (ldsc-format files under sets_l2/ AND a compact .npy bundle for screening)
fastldsc l2 --bfile "$BFILE" --sets "$OUT/sets" --coords "$OUT/genes.coords" \
    --window 20000 --print-snps "$OUT/print_snps.txt" --chr 1-2 \
    --out "$OUT/sets_l2/sets." --bundle-dir "$OUT/sets_l2" --tag sets \
    --overlap-with "$BASE"

# 3. one model, ldsc style (.results / .log), enrichment via the annot + frq files
fastldsc h2 --h2 "$OUT/sumstats/trait0.sumstats.gz" \
    --ref-ld-chr "$BASE,$OUT/sets_l2/sets." --w-ld-chr "$W" --chr 1-2 \
    --overlap-annot --frqfile-chr "$FRQ" --out "$OUT/h2_trait0"
cat "$OUT/h2_trait0.log"
column -t "$OUT/h2_trait0.results" | cut -c1-110

# 4. every set x every trait, marginal and conditional on set5, one CSV
fastldsc scan --ref-ld-chr "$BASE" --w-ld-chr "$W" --chr 1-2 \
    --traits "$TRAITS,$CONTROLS" --sumstats-dir "$OUT/sumstats" \
    --bundle-dir "$OUT/sets_l2" --tags sets --cond set5 --out "$OUT/scan.csv"

# 5. the same candidates through the bordered-solve screen, checked by refit
fastldsc screen --ref-ld-chr "$BASE" --w-ld-chr "$W" --chr 1-2 \
    --traits "$TRAITS,$CONTROLS" --sumstats-dir "$OUT/sumstats" \
    --bundle-dir "$OUT/sets_l2" --tags sets --check 6 --out "$OUT/screen.csv"

# 6. calibrate against the negative-control traits
fastldsc fdr --results "$OUT/screen.csv" --control-traits "$CONTROLS" --z-col z \
    --out "$OUT/screen_fdr.csv"
head -5 "$OUT/screen_fdr.csv"
echo "quickstart finished; outputs in $OUT"
