"""Gene sets -> SNP annotations, and helpers for building gene sets.

The core operation is: a gene set, gene coordinates (from a GTF or a
``GENE CHR START END`` table), a window (default +/-100 kb) and the SNPs of
a reference panel (``.bim``) give a binary annotation over those SNPs.  The
same annotation can be written as an ldsc ``.annot.gz`` (for stock ldsc) or
fed directly to :mod:`fastldsc.l2` as one column of a wide matrix.

The remaining helpers generalise the rank-based gene-set builders used in
the research code: Benjamini-Hochberg masks and in-degree ranking of
(regulator, gene) pairs, MHC exclusion, and a matched control set.
"""

from __future__ import annotations

import gzip
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import io as fio

DEFAULT_WINDOW = 100_000
MHC = ("6", 25_000_000, 35_000_000)   # hg19 extended MHC, as in the research code


# ---------------------------------------------------------------- coordinates

def gene_coords_from_gtf(gtf: str, chroms: Iterable = range(1, 23),
                         feature: str = "gene", strip_version: bool = True,
                         with_symbol: bool = True) -> pd.DataFrame:
    """GENE CHR START END (+ SYMBOL) for every ``feature`` record of a GTF.

    Chromosome names lose a ``chr`` prefix; only ``chroms`` are kept.
    """
    want = {str(c) for c in chroms}
    rows = []
    opener = gzip.open if gtf.endswith(".gz") else open
    with opener(gtf, "rt") as fh:
        for ln in fh:
            if ln.startswith("#"):
                continue
            f = ln.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != feature:
                continue
            chrom = f[0][3:] if f[0].startswith("chr") else f[0]
            if chrom not in want:
                continue
            attrs = f[8]
            try:
                gid = attrs.split('gene_id "')[1].split('"')[0]
            except IndexError:
                continue
            if strip_version:
                gid = gid.split(".")[0]
            sym = ""
            if with_symbol and 'gene_name "' in attrs:
                sym = attrs.split('gene_name "')[1].split('"')[0]
            rows.append((gid, chrom, int(f[3]), int(f[4]), sym))
    df = pd.DataFrame(rows, columns=["GENE", "CHR", "START", "END", "SYMBOL"])
    if not with_symbol:
        df = df.drop(columns=["SYMBOL"])
    return df.drop_duplicates("GENE").reset_index(drop=True)


def mhc_genes(coords: pd.DataFrame, region: Tuple[str, int, int] = MHC) -> set:
    """Genes overlapping the MHC region."""
    c, s, e = region
    sel = (coords["CHR"].astype(str) == str(c)) & (coords["START"] < e) & (coords["END"] > s)
    return set(coords.loc[sel, "GENE"])


# ---------------------------------------------------------------- annotations

def gene_windows(coords: pd.DataFrame, genes: Iterable[str], chrom,
                 window: int = DEFAULT_WINDOW) -> np.ndarray:
    """(k x 2) array of [start - window, end + window] for the genes on ``chrom``."""
    sub = coords[(coords["CHR"].astype(str) == str(chrom)) & coords["GENE"].isin(set(genes))]
    if len(sub) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.column_stack([sub["START"].to_numpy(np.int64) - window,
                            sub["END"].to_numpy(np.int64) + window])


def windows_to_mask(bp: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """Boolean mask over positions ``bp`` (any order) inside any window."""
    bp = np.asarray(bp)
    order = np.argsort(bp, kind="stable")
    bps = bp[order]
    mask = np.zeros(bp.size, bool)
    for lo_bp, hi_bp in windows:
        lo = np.searchsorted(bps, lo_bp, "left")
        hi = np.searchsorted(bps, hi_bp, "right")
        if hi > lo:
            mask[order[lo:hi]] = True
    return mask


def gene_set_annotation(coords: pd.DataFrame, sets: Dict[str, Sequence[str]],
                        bp: np.ndarray, chrom, window: int = DEFAULT_WINDOW,
                        dtype=np.float32) -> np.ndarray:
    """Binary (n_snp x n_set) annotation matrix for one chromosome."""
    A = np.zeros((len(bp), len(sets)), dtype=dtype)
    for k, (name, genes) in enumerate(sets.items()):
        w = gene_windows(coords, genes, chrom, window)
        if len(w):
            A[windows_to_mask(bp, w), k] = 1
    return A


def write_gene_set_annots(coords: pd.DataFrame, sets: Dict[str, Sequence[str]],
                          bfile: str, out_prefix: str, chroms: Iterable[int] = fio.AUTOSOMES,
                          window: int = DEFAULT_WINDOW, thin: bool = True,
                          add_base: bool = False) -> List[str]:
    """Write ``<out_prefix><chr>.annot.gz`` for every chromosome.

    ``bfile`` is a per-chromosome plink prefix (``@`` or trailing number).
    With ``add_base`` a leading all-ones ``base`` column is included.
    """
    paths = []
    names = list(sets)
    for c in chroms:
        bim = fio.read_bim(fio.sub_chr(bfile, c) + ".bim")
        A = gene_set_annotation(coords, sets, bim["BP"].to_numpy(), c, window)
        cols = names
        if add_base:
            A = np.hstack([np.ones((A.shape[0], 1), A.dtype), A])
            cols = ["base"] + names
        path = fio.sub_chr(out_prefix, c) + ".annot.gz"
        fio.write_annot(path, A.astype(int), cols, None if thin else bim)
        paths.append(path)
    return paths


def mappable_genes(coords: pd.DataFrame, genes: Iterable[str],
                   exclude_mhc: bool = True) -> np.ndarray:
    """Boolean: gene has coordinates (and is outside the MHC)."""
    have = set(coords["GENE"])
    bad = mhc_genes(coords) if exclude_mhc else set()
    return np.asarray([g in have and g not in bad for g in genes])


# ---------------------------------------------------------------- set builders

def bh_mask(p: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg discoveries at level ``alpha`` (finite p only)."""
    p = np.asarray(p, float)
    keep = np.zeros(p.shape, bool)
    fin = np.isfinite(p)
    ps = np.sort(p[fin])
    if ps.size == 0:
        return keep
    line = alpha * np.arange(1, ps.size + 1) / ps.size
    ok = ps <= line
    if not ok.any():
        return keep
    keep[fin] = p[fin] <= ps[np.nonzero(ok)[0].max()]
    return keep


def indegree_from_pairs(score: np.ndarray, n_genes: int, topn: int = 100_000) -> np.ndarray:
    """In-degree of each gene among the top-``topn`` (regulator, gene) pairs.

    ``score`` is the flattened (n_regulators x n_genes) score matrix with
    ``-inf``/NaN for pairs that are not eligible.
    """
    score = np.asarray(score, float).ravel()
    score = np.where(np.isnan(score), -np.inf, score)   # NaN would sort as largest
    n = min(topn, int(np.isfinite(score).sum()))
    if n <= 0:
        return np.zeros(n_genes)
    idx = np.argpartition(score, -n)[-n:]
    idx = idx[np.isfinite(score[idx])]
    return np.bincount(idx % n_genes, minlength=n_genes).astype(float)


def top_genes_by_indegree(score: np.ndarray, genes: np.ndarray, sizes: Sequence[int],
                          topn: int = 100_000, eligible: Optional[np.ndarray] = None,
                          prefix: str = "set") -> Dict[str, List[str]]:
    """Rank genes by in-degree and cut the top-M sets for every M in ``sizes``."""
    genes = np.asarray(genes).astype(str)
    deg = indegree_from_pairs(score, genes.size, topn)
    if eligible is not None:
        deg = np.where(np.asarray(eligible, bool), deg, -1.0)
    order = np.argsort(-deg, kind="stable")
    return {"%s_top%d" % (prefix, M): genes[order[:M]].tolist() for M in sizes}


def matched_control(target_idx: Sequence[int], match_score: np.ndarray,
                    eligible: np.ndarray) -> List[int]:
    """Nearest-score matches without replacement, excluding the target set.

    Used to build a control annotation with the same distribution of a
    nuisance quantity (e.g. main-effect in-degree) as a candidate set.
    """
    target = set(int(i) for i in target_idx)
    cand = np.asarray([i for i in range(match_score.size)
                       if i not in target and eligible[i]])
    want = np.sort(match_score[list(target)])[::-1]
    pool = cand[np.argsort(match_score[cand], kind="stable")]
    pdeg = match_score[pool]
    taken = np.zeros(pool.size, bool)
    chosen: List[int] = []
    for w in want:
        j = int(np.searchsorted(pdeg, w))
        lo, hi = j - 1, j
        pick = None
        while lo >= 0 or hi < pool.size:
            cl = abs(pdeg[lo] - w) if lo >= 0 else np.inf
            ch = abs(pdeg[hi] - w) if hi < pool.size else np.inf
            if cl <= ch:
                if lo >= 0 and not taken[lo]:
                    pick = lo
                    break
                lo -= 1
            else:
                if hi < pool.size and not taken[hi]:
                    pick = hi
                    break
                hi += 1
        if pick is not None:
            taken[pick] = True
            chosen.append(int(pool[pick]))
    return chosen


def load_sets(paths_or_npz: Sequence[str]) -> Dict[str, List[str]]:
    """Gene sets from ``.GeneSet`` files, a directory of them, or an ``.npz``."""
    out: Dict[str, List[str]] = {}
    for p in paths_or_npz:
        if os.path.isdir(p):
            files = sorted(os.path.join(p, f) for f in os.listdir(p)
                           if f.endswith(".GeneSet"))
            out.update(fio.read_gene_sets(files))
        elif p.endswith(".npz"):
            out.update(fio.read_sets_npz(p))
        else:
            out.update(fio.read_gene_sets([p]))
    return out
