"""Bordered-solve screening versus full refits."""

import numpy as np
import pytest

from conftest import design
from fastldsc.screen import check_against_refit, screen_candidates
from fastldsc.sldsc import block_jackknife_wls, hsq_weights, run_model


def _base_weighted(td, base_cols, base_M):
    X = np.column_stack(base_cols)[td.rows]
    M_tot = float(np.sum(base_M))
    x_tot = X.sum(1)
    tot_agg = M_tot * (td.chisq.mean() - 1.0) / np.mean(x_tot * td.N)
    w = hsq_weights(x_tot, td.w_ld, td.N, M_tot, tot_agg)
    wn = np.sqrt(w)
    wn = wn / wn.sum()
    Nbar = td.N.mean()
    return wn, Nbar


def test_screen_is_exact_given_the_weights(ref, trait):
    """Holding the weights at the base design, the bordered solve reproduces
    the full-design jackknife (coef and SE) to floating precision."""
    base_cols = ref.base_columns()
    base_M = list(ref.Mbase)
    keys = ["sets:set%dL2" % k for k in range(6)]
    L = np.column_stack([ref.sets[k] for k in keys]).astype(np.float32)
    coef, se, z = screen_candidates(trait, base_cols, base_M, L, chunk=4, n_blocks=50)
    wn, Nbar = _base_weighted(trait, base_cols, base_M)
    for j, k in enumerate(keys):
        cols = base_cols + [L[:, j].astype(np.float64)]
        X = np.column_stack(cols)[trait.rows]
        Xw = np.column_stack([X * (trait.N / Nbar)[:, None], np.ones(trait.n)]) * wn[:, None]
        est, delete, jk_cov, _, _ = block_jackknife_wls(Xw, trait.chisq * wn, 50)
        assert coef[j] == pytest.approx(est[-2] / Nbar, rel=1e-8)
        assert se[j] == pytest.approx(np.sqrt(jk_cov[-2, -2]) / Nbar, rel=1e-6)


def test_screen_matches_refit_within_tolerance(ref, trait):
    """Against the exact refit (weights recomputed with the candidate in the
    total LD score) the screen agrees to within a small fraction of a z."""
    base_cols = ref.base_columns()
    base_M = list(ref.Mbase)
    keys = ["sets:set%dL2" % k for k in range(6)]
    L = np.column_stack([ref.sets[k] for k in keys]).astype(np.float32)
    Mc = np.array([ref.Mset[k] for k in keys])
    coef, se, z = screen_candidates(trait, base_cols, base_M, L, n_blocks=50)
    c2, s2, z2 = check_against_refit(trait, base_cols, base_M, L, Mc, range(6), n_blocks=50)
    assert np.abs(z - z2).max() < 0.05
    assert (np.abs(coef - c2) / s2).max() < 0.1        # coefficient error in SE units
    assert np.abs(se / s2 - 1).max() < 0.05


def test_screen_with_conditioning_set(ref, trait):
    base_cols = ref.base_columns() + [ref.sets["sets:set5L2"]]
    base_M = list(ref.Mbase) + [ref.Mset["sets:set5L2"]]
    L = np.column_stack([ref.sets["sets:set0L2"], ref.sets["sets:set1L2"]]).astype(np.float32)
    coef, se, z = screen_candidates(trait, base_cols, base_M, L, n_blocks=40)
    cols, M = design(ref, ["sets:set5L2", "sets:set0L2"])
    c, s, _, _ = run_model(trait, cols, M, n_blocks=40)
    assert coef[0] == pytest.approx(c[-1], rel=0.02)
    assert np.isfinite(z).all()
