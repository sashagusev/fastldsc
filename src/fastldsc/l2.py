"""LD scores for many annotations in one genotype pass.

``ldsc.py --l2`` recomputes every pairwise r^2 for each annotation file.  In
the windowed sum the annotation only enters as a matrix product

    L[i, a] = sum_{j in window(i)} r2_adj(i, j) * annot[j, a]

so a wide annotation matrix costs one extra GEMM per chunk instead of a
whole genotype pass.  This module reproduces ldsc's estimator exactly:

* genotypes standardised per SNP (missing set to the SNP mean, population
  standard deviation, zero variance -> divisor 1);
* SNPs with zero minor allele frequency, or that are only heterozygous or
  missing, are dropped (ldsc's default ``mafMin = 0`` filter);
* ``r2_adj = r^2 - (1 - r^2) / (n - 2)`` (unbiased for n > 2);
* windows of ``--ld-wind-cm`` (default 1 cM) processed in chunks of
  ``--chunk-size`` SNPs (default 50).  ldsc rounds the left edge of a
  chunk's window *up to a whole chunk*, based on the first SNP of the chunk,
  so the effective window is slightly wider than the nominal one; that is
  reproduced here so that outputs are numerically identical to ldsc's;
* ``M`` counts all retained SNPs, ``M_5_50`` those with MAF > 0.05.

The plink ``.bed`` decoder is pure numpy (no bitarray dependency).

This is an independent implementation; no ldsc source code is used.  The
chunked window traversal necessarily follows the same schedule as ldsc's
``__corSumVarBlocks__`` because the numbers depend on it.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import io as fio

# 2-bit plink codes -> genotype (count of A2 alleles), 9 = missing.
# bits (low first): 00 hom A1, 01 missing, 10 het, 11 hom A2
_CODE = np.array([0, 9, 1, 2], dtype=np.int8)
_LUT = np.zeros((256, 4), dtype=np.int8)
for _b in range(256):
    for _k in range(4):
        _LUT[_b, _k] = _CODE[(_b >> (2 * _k)) & 3]


class PlinkBed:
    """A plink binary fileset with ldsc's SNP filtering and standardisation.

    Parameters
    ----------
    bfile : path prefix of ``.bed/.bim/.fam``
    maf_min : drop SNPs with MAF <= maf_min (ldsc default 0, i.e. keep every
        polymorphic SNP)
    keep_snps : optional indices (into the .bim) to consider

    Attributes
    ----------
    bim : the .bim table (all SNPs)
    kept_snps : indices of retained SNPs
    freq, maf : A2 allele frequency and minor allele frequency of kept SNPs
    m, n : number of kept SNPs and of individuals
    """

    def __init__(self, bfile: str, maf_min: float = 0.0,
                 keep_snps: Optional[np.ndarray] = None):
        self.bfile = bfile
        self.bim = fio.read_bim(bfile + ".bim")
        fam = fio.read_fam(bfile + ".fam")
        self.n = len(fam)
        self.m_all = len(self.bim)
        self.nbytes = (self.n + 3) // 4
        with open(bfile + ".bed", "rb") as fh:
            head = fh.read(3)
            if head[:2] != b"\x6c\x1b":
                raise IOError("not a plink .bed file: %s" % bfile)
            if head[2] != 1:
                raise IOError("plink .bed must be SNP-major")
            raw = np.frombuffer(fh.read(), dtype=np.uint8)
        if raw.size != self.m_all * self.nbytes:
            raise IOError("plink .bed has %d bytes, expected %d"
                          % (raw.size, self.m_all * self.nbytes))
        self._raw = raw.reshape(self.m_all, self.nbytes)
        idx = np.arange(self.m_all) if keep_snps is None \
            else np.asarray(keep_snps, dtype=int)
        freq = np.empty(idx.size)
        keep = np.zeros(idx.size, bool)
        for lo in range(0, idx.size, 20000):
            sel = idx[lo:lo + 20000]
            G = self._decode(sel)                       # (b, n) int8
            miss = G == 9
            n_miss = miss.sum(axis=1)
            n_het = (G == 1).sum(axis=1)
            n_hom2 = (G == 2).sum(axis=1)
            n_nomiss = self.n - n_miss
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(n_nomiss > 0,
                             (n_het + 2 * n_hom2) / (2.0 * n_nomiss), 0.0)
            ok = (np.minimum(f, 1 - f) > maf_min) & ((n_het + n_miss) < self.n)
            freq[lo:lo + sel.size] = f
            keep[lo:lo + sel.size] = ok
        self.kept_snps = idx[keep]
        self.freq = freq[keep]
        self.maf = np.minimum(self.freq, 1 - self.freq)
        self.m = self.kept_snps.size
        self._cursor = 0

    def _decode(self, snp_idx) -> np.ndarray:
        """Raw genotype codes (b x n) for the given .bim row indices."""
        rows = self._raw[snp_idx]                       # (b, nbytes)
        return _LUT[rows].reshape(rows.shape[0], -1)[:, :self.n]

    def genotypes(self, snp_idx) -> np.ndarray:
        """Standardised genotypes (n x b) for .bim row indices."""
        G = self._decode(snp_idx).astype(np.float64).T   # (n, b)
        miss = G == 9
        G[miss] = np.nan
        avg = np.nanmean(G, axis=0)
        # a SNP with every genotype missing cannot be kept, but guard anyway
        avg = np.where(np.isfinite(avg), avg, 0.0)
        G = np.where(miss, avg[None, :], G)
        sd = G.std(axis=0)
        sd[sd == 0] = 1.0
        return (G - avg[None, :]) / sd[None, :]

    def reset(self) -> None:
        self._cursor = 0

    def next_snps(self, b: int) -> np.ndarray:
        """Standardised genotypes of the next ``b`` kept SNPs (n x b)."""
        if self._cursor + b > self.m:
            raise ValueError("%d SNPs requested, %d remain" % (b, self.m - self._cursor))
        sel = self.kept_snps[self._cursor:self._cursor + b]
        self._cursor += b
        return self.genotypes(sel)

    @property
    def kept_bim(self) -> pd.DataFrame:
        return self.bim.iloc[self.kept_snps].reset_index(drop=True)


def block_lefts(coords: np.ndarray, max_dist: float) -> np.ndarray:
    """For each SNP, the index of the leftmost SNP within ``max_dist``.

    ``coords`` must be sorted.  ``block_left[i] = min{j : coords[i] - coords[j] <= max_dist}``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    m = coords.size
    out = np.zeros(m, dtype=np.int64)
    j = 0
    # two-pointer sweep, evaluated with the same comparison ldsc uses so
    # that floating-point edge cases resolve identically
    for i in range(m):
        ci = coords[i]
        while j < m and abs(coords[j] - ci) > max_dist:
            j += 1
        out[i] = j
    return out


def r2_unbiased(r: np.ndarray, n: int) -> np.ndarray:
    denom = n - 2 if n > 2 else n
    sq = np.square(r)
    return sq - (1 - sq) / denom


def ld_score_blocks(geno: PlinkBed, block_left: np.ndarray, chunk: int,
                    annot: Optional[np.ndarray] = None) -> np.ndarray:
    """Windowed sums of adjusted r^2 times annotation, ldsc's chunk schedule.

    Returns an (m x n_annot) array for the kept SNPs of ``geno``.
    """
    m, n = geno.m, geno.n
    c = int(chunk)
    if annot is None:
        annot = np.ones((m, 1))
    annot = np.asarray(annot, dtype=np.float64)
    if annot.shape[0] != m:
        raise ValueError("annot has %d rows, genotypes have %d kept SNPs"
                         % (annot.shape[0], m))
    n_a = annot.shape[1]
    block_left = np.asarray(block_left)
    # window widths rounded up to whole chunks, evaluated at chunk starts
    widths = (np.ceil((np.arange(m) - block_left) / c) * c).astype(int)
    out = np.zeros((m, n_a))
    geno.reset()
    f = lambda r: r2_unbiased(r, n)

    # first block: every SNP whose window reaches back to SNP 0
    nz = np.nonzero(block_left > 0)[0]
    b = int(nz[0]) if nz.size else m
    b = int(np.ceil(b / c) * c)
    if b > m:
        c, b = 1, m
    l_A = 0
    A = geno.next_snps(b)
    for l_B in range(0, b, c):
        B = A[:, l_B:l_B + c]
        rAB = f(A.T @ (B / n))
        out[l_A:l_A + b] += rAB @ annot[l_B:l_B + c]

    # remaining chunks: correlate chunk B against the window A to its left
    b0 = b
    md = int(c * np.floor(m / c))
    end = md + 1 if md != m else md
    for l_B in range(b0, end, c):
        old_b = b
        b = int(widths[l_B])
        if l_B > b0 and b > 0:
            A = np.hstack((A[:, old_b - b + c:old_b], B))
            l_A += old_b - b + c
        elif l_B == b0 and b > 0:
            A = A[:, b0 - b:b0]
            l_A = b0 - b
        elif b == 0:                     # gap: nothing to the left in window
            A = np.empty((n, 0))
            l_A = l_B
        if l_B == md:                    # last, partial chunk
            c = m - md
        B = geno.next_snps(c)
        if not annot[l_A:l_A + b].any() and not annot[l_B:l_B + c].any():
            continue                     # sparse annotation: nothing to add
        rAB = f(A.T @ (B / n))
        out[l_A:l_A + b] += rAB @ annot[l_B:l_B + c]
        out[l_B:l_B + c] += (annot[l_A:l_A + b].T @ rAB).T
        rBB = f(B.T @ (B / n))
        out[l_B:l_B + c] += rBB @ annot[l_B:l_B + c]
    return out


def ld_scores(bfile: str, annot: Optional[np.ndarray] = None,
              ld_wind_cm: Optional[float] = 1.0, ld_wind_kb: Optional[float] = None,
              ld_wind_snps: Optional[int] = None, chunk: int = 50,
              maf_min: float = 0.0):
    """LD scores for one chromosome.

    ``annot`` has one row per .bim SNP (before MAF filtering), as ldsc's
    ``--annot`` files do; it is restricted to the kept SNPs internally.
    Returns ``(geno, L, M, M_5_50)`` with ``L`` (m_kept x n_annot).
    """
    geno = PlinkBed(bfile, maf_min=maf_min)
    if annot is not None:
        annot = np.asarray(annot)
        if annot.shape[0] != geno.m_all:
            raise ValueError("annot must have one row per .bim SNP")
        annot = annot[geno.kept_snps]
    else:
        annot = np.ones((geno.m, 1))
    if sum(x is not None for x in (ld_wind_cm, ld_wind_kb, ld_wind_snps)) != 1:
        raise ValueError("specify exactly one of ld_wind_cm/kb/snps")
    if ld_wind_snps is not None:
        coords, max_dist = np.arange(geno.m), ld_wind_snps
    elif ld_wind_kb is not None:
        coords = geno.bim["BP"].to_numpy(float)[geno.kept_snps]
        max_dist = ld_wind_kb * 1000
    else:
        coords = geno.bim["CM"].to_numpy(float)[geno.kept_snps]
        max_dist = ld_wind_cm
    bl = block_lefts(coords, max_dist)
    L = ld_score_blocks(geno, bl, chunk, annot)
    M = annot.sum(axis=0)
    M_5_50 = annot[geno.maf > 0.05].sum(axis=0)
    return geno, L, M, M_5_50


def ld_scores_chromosome(bfile: str, chrom: int, annot: Optional[np.ndarray],
                         names: Sequence[str], print_snps: Optional[set],
                         ld_wind_cm: float = 1.0, chunk: int = 50,
                         out_prefix: Optional[str] = None,
                         base_annot: Optional[np.ndarray] = None,
                         verbose: bool = True):
    """One chromosome of the multi-annotation driver.

    Returns ``(snp_ids, L_printed, M, M_5_50, overlap)`` where ``overlap``
    (n_annot x n_base) is ``annot[maf>.05].T @ base_annot[maf>.05]`` when
    ``base_annot`` (one row per .bim SNP) is given, else ``None``.  With
    ``out_prefix`` the ldsc-format files are written as ``<out_prefix><chrom>``.
    """
    t0 = time.time()
    geno, L, M, M_5_50 = ld_scores(bfile, annot, ld_wind_cm=ld_wind_cm, chunk=chunk)
    kept = geno.kept_bim
    ids = kept["SNP"].to_numpy()
    keep = np.ones(ids.size, bool) if print_snps is None \
        else np.fromiter((s in print_snps for s in ids), bool, ids.size)
    overlap = None
    if base_annot is not None:
        m5 = geno.maf > 0.05
        Ak = (annot if annot is not None else np.ones((geno.m_all, 1)))[geno.kept_snps][m5]
        overlap = np.asarray(Ak, np.float64).T @ np.asarray(base_annot[geno.kept_snps][m5], np.float64)
    if out_prefix is not None:
        # ldsc's column convention: '<annot>L2' for several annotations, 'L2' for one
        cols = ["L2"] if len(names) == 1 else ["%sL2" % nm for nm in names]
        fio.write_ldscore(out_prefix, chrom, kept[keep], L[keep], cols, M, M_5_50)
    if verbose:
        print("chr%-3s m=%d kept=%d printed=%d  %.0fs"
              % (chrom, geno.m_all, geno.m, keep.sum(), time.time() - t0), flush=True)
    return ids[keep], L[keep].astype(np.float32), M, M_5_50, overlap


def naive_ld_scores(X: np.ndarray, annot: np.ndarray, coords: np.ndarray,
                    max_dist: float) -> np.ndarray:
    """O(m^2) reference implementation with the *strict* window (testing).

    ``X`` standardised genotypes (n x m).  Equals :func:`ld_score_blocks`
    when ``chunk == 1`` (no window rounding) or when the window covers the
    whole chromosome.
    """
    n, m = X.shape
    R = r2_unbiased((X.T @ X) / n, n)
    coords = np.asarray(coords, float)
    W = np.abs(coords[:, None] - coords[None, :]) <= max_dist
    return (R * W) @ np.asarray(annot, float)
