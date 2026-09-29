"""One-pass multi-annotation LD scores."""

import numpy as np
import pandas as pd
import pytest

from fastldsc import io as fio
from fastldsc.l2 import (PlinkBed, block_lefts, ld_score_blocks, ld_scores,
                         naive_ld_scores, r2_unbiased)
from fastldsc.simulate import simulate_genotypes, write_bed


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    rng = np.random.default_rng(11)
    n, m = 60, 400
    G = simulate_genotypes(n, m, rng, missing=0.01)
    G[:, 3] = 0                    # monomorphic -> must be dropped
    G[:, 4] = 1                    # all heterozygous -> dropped by ldsc's rule
    bp = np.sort(rng.choice(np.arange(1000, 3_000_000, 10), m, replace=False))
    bim = pd.DataFrame({"CHR": 1, "SNP": ["s%d" % j for j in range(m)],
                        "CM": bp / 1e6, "BP": bp, "A1": "A", "A2": "G"})
    prefix = str(tmp_path_factory.mktemp("panel") / "p")
    write_bed(prefix, G, bim, ["i%d" % i for i in range(n)])
    return prefix, G, bim


def _standardise(G):
    X = np.where(G < 0, np.nan, G).astype(float)
    avg = np.nanmean(X, 0)
    X = np.where(np.isnan(X), avg, X)
    sd = X.std(0)
    sd[sd == 0] = 1
    return (X - avg) / sd


def test_bed_roundtrip_and_maf_filter(panel):
    prefix, G, bim = panel
    geno = PlinkBed(prefix)
    assert geno.n == G.shape[0]
    assert 3 not in set(geno.kept_snps) and 4 not in set(geno.kept_snps)
    assert geno.m == G.shape[1] - 2
    raw = geno._decode(geno.kept_snps)
    exp = np.where(G < 0, 9, G)[:, geno.kept_snps].T
    assert np.array_equal(raw, exp)
    # allele frequency ignores missing genotypes
    Gk = np.where(G < 0, np.nan, G)[:, geno.kept_snps]
    assert geno.freq == pytest.approx(np.nanmean(Gk, 0) / 2)
    X = geno.next_snps(geno.m)
    assert X.shape == (geno.n, geno.m)
    assert X == pytest.approx(_standardise(G[:, geno.kept_snps]))
    assert np.abs(X.mean(0)).max() < 1e-12
    assert X.std(0) == pytest.approx(1.0)


def test_block_lefts():
    coords = np.array([0.0, 0.5, 1.0, 1.6, 3.0, 3.1])
    assert block_lefts(coords, 1.0).tolist() == [0, 0, 0, 2, 4, 4]


# ------------------------------------------------------------ (b) many == one at a time

def test_multi_annotation_equals_per_annotation(panel):
    prefix, G, bim = panel
    rng = np.random.default_rng(2)
    geno = PlinkBed(prefix)
    A = (rng.random((geno.m, 7)) < 0.2).astype(float)
    A[:, 0] = 1
    A[:, 6] = rng.gamma(2, 1, geno.m)     # continuous annotation
    cm = bim["CM"].to_numpy()[geno.kept_snps]
    bl = block_lefts(cm, 0.2)
    L_all = ld_score_blocks(geno, bl, 50, A)
    for k in range(A.shape[1]):
        L_k = ld_score_blocks(geno, bl, 50, A[:, [k]])
        assert L_k[:, 0] == pytest.approx(L_all[:, k], rel=1e-10, abs=1e-10)


def test_chunk1_equals_naive_strict_window(panel):
    prefix, G, bim = panel
    geno = PlinkBed(prefix)
    rng = np.random.default_rng(4)
    A = np.column_stack([np.ones(geno.m), (rng.random(geno.m) < 0.3).astype(float)])
    cm = bim["CM"].to_numpy()[geno.kept_snps]
    bl = block_lefts(cm, 0.15)
    L = ld_score_blocks(geno, bl, 1, A)          # chunk 1: no window rounding
    X = _standardise(G[:, geno.kept_snps])
    ref = naive_ld_scores(X, A, cm, 0.15)
    assert L == pytest.approx(ref, rel=1e-9, abs=1e-9)


def test_chunked_equals_naive_when_window_covers_chromosome(panel):
    prefix, G, bim = panel
    geno = PlinkBed(prefix)
    A = np.ones((geno.m, 1))
    cm = bim["CM"].to_numpy()[geno.kept_snps]
    bl = block_lefts(cm, 1e9)
    L = ld_score_blocks(geno, bl, 50, A)
    X = _standardise(G[:, geno.kept_snps])
    ref = naive_ld_scores(X, A, cm, 1e9)
    assert L[:, 0] == pytest.approx(ref[:, 0], rel=1e-9)
    # a SNP's own contribution is exactly 1 (r2_adj(1) = 1)
    assert r2_unbiased(np.array([1.0]), geno.n)[0] == pytest.approx(1.0)


def _chunk_schedule_reference(R, A, block_left, c):
    """ldsc's chunked schedule written directly on the full r2 matrix.

    SNPs are processed in chunks of c.  The first block covers all SNPs whose
    window reaches SNP 0 (rounded up to a chunk); every later chunk B is
    correlated against the b = ceil((l_B - block_left[l_B]) / c) * c SNPs to
    its left and against itself.
    """
    m = R.shape[0]
    out = np.zeros((m, A.shape[1]))
    nz = np.nonzero(block_left > 0)[0]
    b = int(nz[0]) if nz.size else m
    b = int(np.ceil(b / c) * c)
    if b > m:
        c, b = 1, m
    out[:b] += R[:b, :b] @ A[:b]
    for l_B in range(b, m, c):
        hi = min(l_B + c, m)
        w = int(np.ceil((l_B - block_left[l_B]) / c) * c)
        lo = l_B - w
        out[lo:l_B] += R[lo:l_B, l_B:hi] @ A[l_B:hi]
        out[l_B:hi] += R[l_B:hi, lo:l_B] @ A[lo:l_B]
        out[l_B:hi] += R[l_B:hi, l_B:hi] @ A[l_B:hi]
    return out


def test_chunk_schedule_matches_reference(panel):
    """With chunk > 1 the window's left edge is rounded up to a chunk (as in
    ldsc); the streamed implementation equals a direct evaluation of that
    schedule on the full r2 matrix, and differs from the strict window."""
    prefix, G, bim = panel
    geno = PlinkBed(prefix)
    rng = np.random.default_rng(6)
    A = np.column_stack([np.ones(geno.m), (rng.random(geno.m) < 0.3).astype(float)])
    cm = bim["CM"].to_numpy()[geno.kept_snps]
    X = _standardise(G[:, geno.kept_snps])
    R = r2_unbiased((X.T @ X) / geno.n, geno.n)
    for max_dist, c in ((0.1, 50), (0.05, 7), (0.3, 64)):
        bl = block_lefts(cm, max_dist)
        L = ld_score_blocks(geno, bl, c, A)
        assert L == pytest.approx(_chunk_schedule_reference(R, A, bl, c), rel=1e-9, abs=1e-9)
        assert not np.allclose(L, naive_ld_scores(X, A, cm, max_dist))


def test_ld_scores_driver_M_counts(panel):
    prefix, G, bim = panel
    rng = np.random.default_rng(9)
    A = (rng.random((len(bim), 2)) < 0.4).astype(float)
    geno, L, M, M5 = ld_scores(prefix, A, ld_wind_cm=0.2)
    assert M.tolist() == A[geno.kept_snps].sum(0).tolist()
    assert M5.tolist() == A[geno.kept_snps][geno.maf > 0.05].sum(0).tolist()
    assert L.shape == (geno.m, 2)
    with pytest.raises(ValueError):
        ld_scores(prefix, A[:-1], ld_wind_cm=0.2)
    with pytest.raises(ValueError):
        ld_scores(prefix, A, ld_wind_cm=None)


def test_ldscore_file_roundtrip(tmp_path, panel):
    prefix, G, bim = panel
    geno, L, M, M5 = ld_scores(prefix, None, ld_wind_cm=0.2)
    out = str(tmp_path / "x.")
    fio.write_ldscore(out, 1, geno.kept_bim, L, ["L2"], M, M5)
    df = fio.read_ldscore_chr(out, [1])
    assert list(df.columns) == ["SNP", "L2"]
    assert df["L2"].to_numpy() == pytest.approx(L[:, 0], abs=5e-4)
    assert fio.read_M(out, 1, [1])[0] == M5[0]
    assert fio.read_M(out, 1, [1], common=False)[0] == M[0]
