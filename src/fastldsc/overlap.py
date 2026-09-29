"""Cached annotation overlap for enrichment across many models.

ldsc's ``--overlap-annot`` needs the overlap matrix ``A.T @ A`` of every
annotation in the design over the common SNPs (0.05 < MAF < 0.95).  When
the baseline is fixed and only one or two candidate columns change, the
baseline block of that matrix and the baseline annotation matrix itself can
be computed once (:func:`build_cache`) and each candidate then costs one
matrix-vector product (:meth:`OverlapCache.rows_for_vectors`).
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence

import numpy as np

from . import io as fio


def build_cache(annot_prefix: str, frq_prefix: Optional[str], out: str,
                chroms: Iterable[int] = fio.AUTOSOMES, save_annot: bool = True,
                verbose: bool = True) -> str:
    """Compute the baseline overlap matrix and (optionally) keep the
    common-SNP baseline annotation matrix as a float32 ``.npy`` next to it.

    Writes ``<out>.npz`` (overlap, M_tot, names, mask, chrom_rows, chroms)
    and ``<out>.annot.npy`` (M_tot x n_annot).  ``mask`` is the common-SNP
    mask over the .bim rows of every chromosome, concatenated.
    """
    chroms = list(chroms)
    overlap = None
    names: List[str] = []
    masks, blocks, rows_per_chr = [], [], []
    for c in chroms:
        _, a = fio.read_annot(fio.find_compressed(fio.sub_chr(annot_prefix, c) + ".annot"))
        if not names:
            names = list(a.columns)
        A = a.to_numpy(np.float64)
        if frq_prefix is not None:
            frq = fio.read_frq(fio.find_compressed(fio.sub_chr(frq_prefix, c) + ".frq"))
            if len(frq) != A.shape[0]:
                raise ValueError("frq/annot row mismatch on chr%s" % c)
            mask = fio.common_snp_mask(frq)
        else:
            mask = np.ones(A.shape[0], bool)
        A = A[mask]
        overlap = A.T @ A if overlap is None else overlap + A.T @ A
        masks.append(mask)
        rows_per_chr.append(mask.size)
        if save_annot:
            blocks.append(A.astype(np.float32))
        if verbose:
            print("overlap: chr%s %d common SNPs" % (c, A.shape[0]), flush=True)
    mask_all = np.concatenate(masks)
    np.savez(out + ".npz", overlap=overlap, M_tot=int(mask_all.sum()),
             names=np.array(names, dtype=object), mask=mask_all,
             chrom_rows=np.array(rows_per_chr), chroms=np.array(chroms))
    if save_annot:
        np.save(out + ".annot.npy", np.vstack(blocks))
    return out + ".npz"


class OverlapCache:
    """Baseline overlap block plus (memory-mapped) common-SNP annotations."""

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=True)
        self.overlap = z["overlap"]
        self.M_tot = int(z["M_tot"])
        self.names = list(z["names"].astype(str))
        self.mask = z["mask"].astype(bool)
        self.chrom_rows = z["chrom_rows"]
        self.chroms = [int(c) for c in z["chroms"]]
        annot_path = path[:-4] + ".annot.npy" if path.endswith(".npz") else path + ".annot.npy"
        self.annot = np.load(annot_path, mmap_mode="r") if os.path.exists(annot_path) else None
        self._offsets = np.concatenate([[0], np.cumsum(self.chrom_rows)])

    def vector_for_prefix(self, annot_prefix: str) -> np.ndarray:
        """Common-SNP annotation column(s) of a per-chromosome annot prefix."""
        parts = []
        for k, c in enumerate(self.chroms):
            _, a = fio.read_annot(fio.find_compressed(fio.sub_chr(annot_prefix, c) + ".annot"))
            m = self.mask[self._offsets[k]:self._offsets[k + 1]]
            if len(a) != m.size:
                raise ValueError("annot rows differ from cache on chr%s" % c)
            parts.append(a.to_numpy(np.float64)[m])
        return np.vstack(parts)

    def rows_for_vectors(self, S: np.ndarray, block: int = 500_000) -> np.ndarray:
        """``S.T @ annot`` for common-SNP annotation vectors ``S`` (M_tot x k)."""
        if self.annot is None:
            raise RuntimeError("cache was built without the annotation matrix")
        S = np.asarray(S, np.float64)
        out = np.zeros((S.shape[1], self.annot.shape[1]))
        for lo in range(0, self.M_tot, block):
            out += S[lo:lo + block].T @ np.asarray(self.annot[lo:lo + block], np.float64)
        return out

    def full_matrix(self, S: np.ndarray) -> np.ndarray:
        """Overlap matrix of ``[baseline | S]``."""
        nb = self.overlap.shape[0]
        k = S.shape[1]
        full = np.zeros((nb + k, nb + k))
        full[:nb, :nb] = self.overlap
        cross = self.rows_for_vectors(S)                # (k, nb)
        full[nb:, :nb] = cross
        full[:nb, nb:] = cross.T
        full[nb:, nb:] = np.asarray(S, np.float64).T @ np.asarray(S, np.float64)
        return full

    def full_matrix_from_rows(self, rows: np.ndarray, SS: np.ndarray) -> np.ndarray:
        """Overlap matrix from precomputed candidate rows (k x nb) and the
        candidate-candidate block (k x k), e.g. from an l2 bundle."""
        nb = self.overlap.shape[0]
        k = rows.shape[0]
        full = np.zeros((nb + k, nb + k))
        full[:nb, :nb] = self.overlap
        full[nb:, :nb] = rows
        full[:nb, nb:] = rows.T
        full[nb:, nb:] = SS
        return full
