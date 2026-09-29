"""Screen thousands of candidate annotations per trait via bordered solves.

Within a trait the design ``[base annotations | intercept]`` is fixed and
only the candidate column changes, so the augmented WLS coefficient is

    a_s = (c_s - g_s' b0*) / (d_s - g_s' G0^-1 g_s),   b0* = G0^-1 b0

with ``g_s = Xb' W x_s``, ``d_s = x_s' W x_s``, ``c_s = x_s' W y``.  Per
jackknife block the same identity holds on leave-one-block-out cross
products, so all 200 delete values for every candidate come from grouped
matrix products.  Coefficient, jackknife SE and z match the full refit
exactly *given the weights*; the weights are held at their base-design
values (the candidate's contribution to the total LD score is small), which
is the only approximation.  :func:`check_against_refit` measures it.

The numerics are unchanged from the validated research code
(``screen_regulons.screen_trait``); only argument handling changed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .sldsc import N_BLOCKS, TraitData, block_separators, hsq_weights, run_model


def screen_candidates(td: TraitData, base_cols: Sequence[np.ndarray],
                      base_M: Sequence[float], L: np.ndarray,
                      chunk: int = 250, n_blocks: int = N_BLOCKS):
    """Coefficient, jackknife SE and z for every column of ``L``.

    Parameters
    ----------
    td : merged trait
    base_cols : reference-order LD score arrays of the fixed design
        (baseline annotations plus any conditioning sets)
    base_M : their ``M_5_50`` counts
    L : (n_reference_snps x S) candidate LD scores in reference order
    chunk : candidates per matrix-product batch
    """
    n = td.n
    Xb_raw = np.column_stack([c[td.rows] for c in base_cols]).astype(np.float64)
    M_tot_base = float(np.sum(base_M))
    y = td.chisq
    N = td.N
    Nbar = N.mean()
    x_tot = Xb_raw.sum(axis=1)
    tot_agg = M_tot_base * (y.mean() - 1.0) / np.mean(x_tot * N)
    w = hsq_weights(x_tot, td.w_ld, N, M_tot_base, tot_agg)
    wn = np.sqrt(w)
    wn /= wn.sum()
    Xb = np.column_stack([Xb_raw * (N / Nbar)[:, None], np.ones(n)]) * wn[:, None]
    yw = y * wn
    p = Xb.shape[1]

    n_blocks = min(n_blocks, n)
    seps = block_separators(n, n_blocks)
    Gb = np.zeros((n_blocks, p, p))
    bb = np.zeros((n_blocks, p))
    for b in range(n_blocks):
        s, e = seps[b], seps[b + 1]
        xb = Xb[s:e]
        Gb[b] = xb.T @ xb
        bb[b] = xb.T @ yw[s:e]
    G0 = Gb.sum(0)
    b0 = bb.sum(0)
    # leave-one-block-out inverses (p is only ~100)
    Gm = G0[None] - Gb
    bm = b0[None] - bb
    Ginv_all = np.linalg.inv(Gm)                       # (B,p,p)
    beta_all = np.einsum("bij,bj->bi", Ginv_all, bm)   # (B,p)
    G0inv = np.linalg.inv(G0)
    beta0 = G0inv @ b0

    S = L.shape[1]
    out_coef = np.empty(S)
    out_se = np.empty(S)
    scal = (N / Nbar) * wn                              # candidate col scaling
    for lo in range(0, S, chunk):
        hi = min(lo + chunk, S)
        Xs = np.asarray(L[td.rows, lo:hi], dtype=np.float64) * scal[:, None]
        g_tot = Xb.T @ Xs                               # (p, k)
        d_tot = np.einsum("ij,ij->j", Xs, Xs)
        c_tot = Xs.T @ yw
        gb = np.empty((n_blocks, p, hi - lo))
        db = np.empty((n_blocks, hi - lo))
        cb = np.empty((n_blocks, hi - lo))
        for b in range(n_blocks):
            s, e = seps[b], seps[b + 1]
            xs = Xs[s:e]
            gb[b] = Xb[s:e].T @ xs
            db[b] = np.einsum("ij,ij->j", xs, xs)
            cb[b] = xs.T @ yw[s:e]
        gm = g_tot[None] - gb
        dm = d_tot[None] - db
        cm = c_tot[None] - cb
        # full-data estimate
        u0 = G0inv @ g_tot
        a0 = (c_tot - g_tot.T @ beta0) / (d_tot - np.einsum("ij,ij->j", g_tot, u0))
        # delete values
        um = np.einsum("bij,bjk->bik", Ginv_all, gm)
        denom = dm - np.einsum("bjk,bjk->bk", gm, um)
        num = cm - np.einsum("bjk,bj->bk", gm, beta_all)
        adel = num / denom                               # (B,k)
        pseudo = n_blocks * a0[None] - (n_blocks - 1) * adel
        var = pseudo.var(axis=0, ddof=1) / n_blocks
        out_coef[lo:hi] = a0 / Nbar
        out_se[lo:hi] = np.sqrt(var) / Nbar
    return out_coef, out_se, out_coef / out_se


def check_against_refit(td: TraitData, base_cols: Sequence[np.ndarray],
                        base_M: Sequence[float], L: np.ndarray,
                        M_cand: Sequence[float], which: Sequence[int],
                        n_blocks: int = N_BLOCKS):
    """Exact refits for candidates ``which``; returns (coef, se, z) arrays.

    Compare with :func:`screen_candidates` to measure the weight
    approximation on your own data.
    """
    coef = np.empty(len(which))
    se = np.empty(len(which))
    for k, j in enumerate(which):
        cols = list(base_cols) + [np.asarray(L[:, j], np.float64)]
        M = np.concatenate([np.asarray(base_M, float), [M_cand[j]]])
        c, s, _, _ = run_model(td, cols, M, n_blocks)
        coef[k], se[k] = c[-1], s[-1]
    return coef, se, coef / se
