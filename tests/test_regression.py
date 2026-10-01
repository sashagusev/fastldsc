"""The WLS / block-jackknife regression."""

import os

import numpy as np
import pandas as pd
import pytest

from conftest import design
from fastldsc.sldsc import (H2Result, TraitData, block_jackknife_wls, fit_h2,
                            hsq_weights, overlap_output, ratio_jackknife, run_model,
                            wls_coef)


class FakeTrait:
    """Minimal stand-in for TraitData built from arrays."""

    def __init__(self, chisq, N, w_ld):
        self.chisq = np.asarray(chisq, float)
        self.N = np.asarray(N, float)
        self.w_ld = np.asarray(w_ld, float)
        self.rows = np.arange(self.chisq.size)
        self.n = self.chisq.size
        self.name = "fake"


def _weighted_design(td, cols, M_annot):
    """Rebuild the weighted design exactly as fit_h2 does (for reference fits)."""
    X = np.column_stack(cols)[td.rows]
    M_tot = float(np.sum(M_annot))
    x_tot = X.sum(1)
    tot_agg = M_tot * (td.chisq.mean() - 1.0) / np.mean(x_tot * td.N)
    w = hsq_weights(x_tot, td.w_ld, td.N, M_tot, tot_agg)
    Nbar = td.N.mean()
    Xs = np.column_stack([X * (td.N / Nbar)[:, None], np.ones(td.n)])
    wn = np.sqrt(w)
    wn = wn / wn.sum()
    return Xs * wn[:, None], td.chisq * wn, Nbar


# ------------------------------------------------------------ (d) hand-checkable

def test_hand_checkable_exact_linear_model():
    """chisq exactly 1 + N (tau1 l1 + tau2 l2): coefficients are recovered to
    machine precision, the intercept is 1 and every jackknife SE is ~0."""
    rng = np.random.default_rng(0)
    m = 4000
    l1 = rng.gamma(4, 3, m)
    l2 = rng.gamma(2, 2, m)
    N = np.full(m, 20000.0)
    tau1, tau2 = 3e-6, -5e-7
    chisq = 1 + N * (tau1 * l1 + tau2 * l2)
    td = FakeTrait(chisq, N, w_ld=np.fmax(l1, 1))
    M = np.array([1e5, 3e4])
    res = fit_h2(td, [l1, l2], M, n_blocks=20)
    assert res.coef == pytest.approx([tau1, tau2], rel=1e-9)
    assert res.intercept == pytest.approx(1.0, abs=1e-9)
    assert np.all(res.coef_se < 1e-12)
    assert res.tot == pytest.approx(M[0] * tau1 + M[1] * tau2, rel=1e-9)
    assert res.mean_chisq == pytest.approx(chisq.mean())
    # non-overlap enrichment (cat / M) / (tot / M_tot)
    M_tot = M.sum()
    assert res.enrichment[0] == pytest.approx(tau1 / (res.tot / M_tot), rel=1e-9)


def test_tau_star_uses_reference_snp_count():
    """tau* scales by the number of reference SNPs, not by the sum of M over
    annotations (which overcounts SNPs when annotations overlap)."""
    rng = np.random.default_rng(0)
    m = 4000
    l1 = rng.gamma(4, 3, m)
    l2 = rng.gamma(2, 2, m)
    N = np.full(m, 20000.0)
    chisq = 1 + N * (3e-6 * l1 - 5e-7 * l2)
    td = FakeTrait(chisq, N, w_ld=np.fmax(l1, 1))
    M = np.array([1e5, 3e4])          # an all-SNP base column and a 30% subset of it
    res = fit_h2(td, [l1, l2], M, n_blocks=20)
    sd = np.array([0.0, np.sqrt(0.3 * 0.7)])
    ts = res.tau_star(sd, M_ref=1e5)
    assert ts[0] == 0.0
    assert ts[1] == pytest.approx(res.coef[1] * sd[1] * 1e5 / res.tot, rel=1e-12)
    assert ts[1] != pytest.approx(res.coef[1] * sd[1] * M.sum() / res.tot, rel=1e-3)


def test_duplicate_snp_ids_keep_first_row(synth, ref, trait, tmp_path):
    """A repeated SNP id is dropped after its first row, as in ldsc's
    _read_sumstats, instead of breaking the merge with the reference."""
    ss = pd.read_csv(os.path.join(synth["sumstats_dir"], "trait0.sumstats.gz"), sep="\t")
    first = ss[ss["SNP"].isin(ref.snp[trait.rows])].iloc[[0]]
    plain, dup = str(tmp_path / "plain.sumstats.gz"), str(tmp_path / "dup.sumstats.gz")
    ss.to_csv(plain, sep="\t", index=False, compression="gzip")
    pd.concat([ss, first.assign(Z=first["Z"] + 3.0)]).to_csv(dup, sep="\t", index=False,
                                                          compression="gzip")
    td, td0 = TraitData(ref, dup), TraitData(ref, plain)
    assert td.n == td0.n == trait.n
    assert np.array_equal(td.rows, td0.rows)
    assert np.array_equal(td.chisq, td0.chisq)


def test_space_delimited_sumstats(synth, trait, ref, tmp_path):
    """Space-delimited sumstats (e.g. UKB_460K files) read like tab-delimited
    ones, as ldsc reads both."""
    ss = pd.read_csv(os.path.join(synth["sumstats_dir"], "trait0.sumstats.gz"), sep="\t")
    tab, space = str(tmp_path / "tab.sumstats.gz"), str(tmp_path / "space.sumstats.gz")
    ss.to_csv(tab, sep="\t", index=False, compression="gzip")
    ss.to_csv(space, sep=" ", index=False, compression="gzip")
    td_tab, td_space = TraitData(ref, tab), TraitData(ref, space)
    assert td_space.n == td_tab.n == trait.n
    assert np.array_equal(td_space.chisq, td_tab.chisq)


def test_two_point_hand_solution():
    """Two annotations, few SNPs: matches an explicit normal-equation solve."""
    rng = np.random.default_rng(3)
    m = 300
    l1 = rng.gamma(3, 2, m)
    l2 = (rng.random(m) < 0.3) * rng.gamma(2, 1, m)
    N = rng.integers(10000, 20000, m).astype(float)
    chisq = rng.chisquare(1, m) * (1 + N * (2e-6 * l1 + 1e-5 * l2))
    td = FakeTrait(chisq, N, w_ld=np.fmax(l1, 1))
    M = np.array([5e4, 8e3])
    res = fit_h2(td, [l1, l2], M, n_blocks=10)
    Xw, yw, Nbar = _weighted_design(td, [l1, l2], M)
    beta = np.linalg.solve(Xw.T @ Xw, Xw.T @ yw)
    assert res.coef == pytest.approx(beta[:2] / Nbar, rel=1e-12)
    assert res.intercept == pytest.approx(beta[2], rel=1e-12)


# ------------------------------------------------------------ WLS equivalence

def test_normal_equations_match_svd_wls(ref, trait):
    """The block-sufficient-statistic solve equals an SVD least-squares fit
    of the same weighted design (to the precision of the conditioning)."""
    cols, M = design(ref, ["sets:set0L2", "sets:set1L2"])
    res = fit_h2(trait, cols, M)
    Xw, yw, Nbar = _weighted_design(trait, cols, M)
    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    assert res.coef == pytest.approx(beta[:-1] / Nbar, rel=1e-8, abs=1e-14)
    assert res.intercept == pytest.approx(beta[-1], rel=1e-8)
    # and the same through the public helper
    X = np.column_stack(cols)[trait.rows]
    x_tot = X.sum(1)
    tot_agg = M.sum() * (trait.chisq.mean() - 1) / np.mean(x_tot * trait.N)
    w = hsq_weights(x_tot, trait.w_ld, trait.N, M.sum(), tot_agg)
    Xs = np.column_stack([X * (trait.N / Nbar)[:, None], np.ones(trait.n)])
    b2 = wls_coef(Xs, trait.chisq, np.sqrt(w))
    assert b2[:-1] / Nbar == pytest.approx(res.coef, rel=1e-8, abs=1e-14)


# ------------------------------------------------------------ (a) batched == one at a time

def test_block_jackknife_equals_explicit_leave_one_out():
    """Delete values from the batched X'WX / X'Wy block sums equal literal
    refits with each block removed, and the covariance follows."""
    rng = np.random.default_rng(5)
    n, p, B = 1000, 4, 25
    Xw = rng.standard_normal((n, p))
    yw = Xw @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.standard_normal(n)
    est, delete, jk_cov, xtx_b, xty_b = block_jackknife_wls(Xw, yw, B)
    seps = np.floor(np.linspace(0, n, B + 1)).astype(int)
    for b in range(B):
        keep = np.ones(n, bool)
        keep[seps[b]:seps[b + 1]] = False
        ref_b = np.linalg.lstsq(Xw[keep], yw[keep], rcond=None)[0]
        assert delete[b] == pytest.approx(ref_b, rel=1e-9, abs=1e-12)
    full = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    assert est == pytest.approx(full, rel=1e-10)
    pseudo = B * full - (B - 1) * delete
    assert jk_cov == pytest.approx(np.cov(pseudo.T, ddof=1) / B, rel=1e-9)


def test_fit_is_independent_of_batching_and_order(ref, synth):
    """A model fitted alone, inside a scan over other sets, with a fresh
    RefData holding fewer columns, or from float32 views gives bit-identical
    numbers: nothing about batching leaks into the arithmetic."""
    td = TraitData(ref, f"{synth['sumstats_dir']}/trait1.sumstats.gz")
    keys = ["sets:set%dL2" % k for k in range(6)]
    alone = {}
    for k in keys:
        cols, M = design(ref, [k])
        alone[k] = run_model(td, cols, M)
    # scan in reverse order, interleaved with two-column models
    for k in reversed(keys):
        if k != "sets:set3L2":
            cols, M = design(ref, [k, "sets:set3L2"])
            run_model(td, cols, M)
        cols, M = design(ref, [k])
        c, s, i, t = run_model(td, cols, M)
        assert np.array_equal(c, alone[k][0]) and np.array_equal(s, alone[k][1])
        assert i == alone[k][2] and t == alone[k][3]
    # a RefData that never loaded the other sets
    from fastldsc.sldsc import RefData
    small = RefData(synth["base"], synth["w"], chroms=[1, 2], verbose=False)
    small.add_column("only", ref.sets["sets:set2L2"], ref.Mset["sets:set2L2"])
    td2 = TraitData(small, f"{synth['sumstats_dir']}/trait1.sumstats.gz")
    cols, M = design(small, ["only"])
    c, s, i, t = run_model(td2, cols, M)
    assert np.array_equal(c, alone["sets:set2L2"][0])
    assert np.array_equal(s, alone["sets:set2L2"][1])


def test_fit_h2_and_run_model_agree(ref, trait):
    cols, M = design(ref, ["sets:set4L2"])
    res = fit_h2(trait, cols, M)
    c, s, i, t = run_model(trait, cols, M)
    assert np.array_equal(res.coef, c) and np.array_equal(res.coef_se, s)
    assert res.intercept == i and res.tot == t
    assert isinstance(res, H2Result)
    assert res.coef_delete.shape == (res.n_blocks, M.size)


# ------------------------------------------------------------ ratio jackknife / enrichment

def test_ratio_jackknife_matches_naive():
    rng = np.random.default_rng(1)
    B, p = 30, 3
    numer = rng.random((B, p)) + 1
    denom = (rng.random((B, 1)) + 2) * np.ones((1, p))
    est = numer.mean(0) / denom.mean(0)
    cov, se = ratio_jackknife(est, numer, denom)
    pseudo = np.array([B * est - (B - 1) * numer[j] / denom[j] for j in range(B)])
    assert cov == pytest.approx(np.cov(pseudo.T, ddof=1) / B)
    assert se == pytest.approx(np.sqrt(np.diag(cov)))


def test_overlap_output_reduces_to_plain_enrichment_when_disjoint(ref, trait):
    """With a diagonal overlap matrix Prop._h2 is cat/tot and Enrichment is
    the non-overlap enrichment; SEs follow the ratio jackknife."""
    cols, M = design(ref, ["sets:set0L2"])
    res = fit_h2(trait, cols, M)
    overlap = np.diag(M)
    df = overlap_output(res, overlap, float(M.sum()))
    assert df["Prop._h2"].to_numpy() == pytest.approx(res.prop)
    assert df["Enrichment"].to_numpy() == pytest.approx(res.enrichment)
    assert df["Prop._h2_std_error"].to_numpy() == pytest.approx(res.prop_se)
    assert df["Coefficient_z-score"].to_numpy() == pytest.approx(res.z)
    assert df["Prop._SNPs"].to_numpy() == pytest.approx(M / M.sum())


def test_overlap_output_general_matrix_formula(ref, trait):
    """Prop._h2 = (overlap / M) @ prop, Enrichment = Prop._h2 / Prop._SNPs."""
    cols, M = design(ref, ["sets:set0L2", "sets:set1L2"])
    res = fit_h2(trait, cols, M)
    rng = np.random.default_rng(2)
    k = M.size
    R = rng.random((k, k))
    overlap = np.minimum(np.outer(M, M) / M.max(), R * M[None, :])
    overlap = np.triu(overlap) + np.triu(overlap, 1).T
    np.fill_diagonal(overlap, M)
    M_tot = float(M[0])
    df = overlap_output(res, overlap, M_tot)
    prop_h2 = (overlap / M[None, :]) @ res.prop
    assert df["Prop._h2"].to_numpy() == pytest.approx(prop_h2)
    assert df["Enrichment"].to_numpy() == pytest.approx(prop_h2 / (M / M_tot))
    var = np.diag((overlap / M[None, :]) @ res.prop_cov @ (overlap / M[None, :]).T)
    assert df["Prop._h2_std_error"].to_numpy() == pytest.approx(np.sqrt(np.maximum(0, var)))
    # the base annotation covers every SNP: its Enrichment_p is undefined (NA in ldsc)
    assert np.isnan(df["Enrichment_p"].iloc[0])
    assert np.isfinite(df["Enrichment_p"].iloc[1:]).all()


def test_chisq_filter_and_merge(ref, synth, tmp_path):
    """SNPs with chisq >= max(0.001 Nmax, 80) are dropped; merge order is the
    reference order."""
    import gzip
    import pandas as pd
    p = f"{synth['sumstats_dir']}/trait0.sumstats.gz"
    ss = pd.read_csv(p, sep="\t")
    ss.loc[5, "Z"] = 10.0              # chisq 100 > 80
    ss.loc[7, "N"] = np.nan            # dropped by dropna
    q = tmp_path / "t.sumstats.gz"
    ss.to_csv(q, sep="\t", index=False, compression="gzip")
    td = TraitData(ref, str(q))
    assert td.chisq_max == 80.0
    assert td.chisq.max() < 80
    assert np.all(np.diff(td.rows) > 0)
    assert td.n == td.n_merged - 1
    assert ss.loc[5, "SNP"] not in set(ref.snp[td.rows])
