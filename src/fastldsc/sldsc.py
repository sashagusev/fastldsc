"""Stratified LD score regression with all I/O amortised.

This module is an arithmetic mirror of the partitioned-heritability path of
``ldsc.py --h2`` (free intercept, more than one annotation).  It reproduces,
in order:

1. inner merge sumstats -> reference LD scores -> regression-weight LD
   scores, preserving the reference (chromosome, position) order that the
   block jackknife is defined over;
2. the chi-square filter ``chisq < max(0.001 * N.max(), 80)``;
3. the aggregate-heritability initial weights
   ``1 / (2 (1 + h2_agg N l / M)^2) / w_ld`` with ``l`` and ``w_ld`` floored
   at 1;
4. the ``N / mean(N)`` column scaling and the appended intercept column;
5. ONE weighted least squares fit with those weights.  For n_annot > 1 ldsc
   sets ``old_weights=True`` (``ldscore/sumstats.py``: ``estimate_h2``),
   which bypasses the iteratively re-weighted fit used for single-annotation
   h2 -- there is no IRWLS in the partitioned path;
6. the 200-block delete-one-block jackknife on the block-wise ``X'WX`` and
   ``X'Wy`` sufficient statistics, and the ratio jackknife for the
   proportion of h2 (used for ``Prop._h2`` / ``Enrichment`` SEs).

The reference LD scores, weights and any number of candidate annotation
columns are loaded once (:class:`RefData`); each trait is merged once
(:class:`TraitData`); each model is then a few matrix products.

The numerics of :func:`run_model` are unchanged from the research code that
was validated against stock ldsc (see the README for the measured
deviations).  Only argument handling and I/O were refactored.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist

from . import io as fio

N_BLOCKS = 200


# ---------------------------------------------------------------- data


class RefData:
    """Baseline LD scores, regression weights and candidate columns.

    Parameters
    ----------
    base_prefix : per-chromosome prefix of the baseline (e.g. ``.../baselineLD.``)
    w_prefix : per-chromosome prefix of the regression-weight LD scores
    set_prefixes : ``name -> prefix`` of extra single-column LD score files
        (each ``<prefix><chr>.l2.ldscore.gz`` + ``.l2.M_5_50``)
    chroms : chromosomes to read (default autosomes)
    common : use ``.l2.M_5_50`` (ldsc default) rather than ``.l2.M``
    """

    def __init__(self, base_prefix: str, w_prefix: str,
                 set_prefixes: Optional[Dict[str, str]] = None,
                 chroms: Iterable[int] = fio.AUTOSOMES, common: bool = True,
                 verbose: bool = True):
        t0 = time.time()
        self.chroms = list(chroms)
        self.base_prefix = base_prefix
        base = fio.read_ldscore_chr(base_prefix, self.chroms)
        self.snp = base["SNP"].to_numpy()
        self.base_cols = [c for c in base.columns if c != "SNP"]
        self.Xbase = base[self.base_cols].to_numpy(np.float64)
        self.Mbase = fio.read_M(base_prefix, len(self.base_cols), self.chroms,
                                common=common)
        self._idx = pd.Series(np.arange(self.snp.size), index=self.snp)
        w = fio.read_ldscore_chr(w_prefix, self.chroms)
        if w.shape[1] != 2:
            raise ValueError("--w-ld-chr may only have one LD score column")
        self.w_ld = np.full(self.snp.size, np.nan)
        ii = self._idx.reindex(w["SNP"]).to_numpy()
        ok = np.isfinite(ii)
        self.w_ld[ii[ok].astype(int)] = w[w.columns[-1]].to_numpy(np.float64)[ok]
        self.sets: Dict[str, np.ndarray] = {}
        self.Mset: Dict[str, float] = {}
        self.set_names: Dict[str, List[str]] = {}
        for name, prefix in (set_prefixes or {}).items():
            self.add_prefix(name, prefix, common=common)
        if verbose:
            print("refdata: %d snps, %d baseline columns, %d sets, %.0fs"
                  % (self.snp.size, len(self.base_cols), len(self.sets),
                     time.time() - t0), flush=True)

    # -- adding columns ------------------------------------------------
    def add_prefix(self, name: str, prefix: str, common: bool = True) -> List[str]:
        """Add every column of an LD score file set; returns the keys added.

        A single-column file is stored under ``name``; a multi-column file
        under ``name:<column>``.
        """
        df = fio.read_ldscore_chr(prefix, self.chroms)
        cols = [c for c in df.columns if c != "SNP"]
        M = fio.read_M(prefix, len(cols), self.chroms, common=common)
        jj = self._idx.reindex(df["SNP"]).to_numpy()
        ok = np.isfinite(jj)
        rows = jj[ok].astype(int)
        keys = []
        for k, col in enumerate(cols):
            key = name if len(cols) == 1 else "%s:%s" % (name, col)
            v = np.full(self.snp.size, np.nan, np.float32)
            v[rows] = df[col].to_numpy(np.float32)[ok]
            self.sets[key] = v
            self.Mset[key] = float(M[k])
            keys.append(key)
        self.set_names[name] = keys
        return keys

    def add_column(self, name: str, values_ref_order: np.ndarray, M: float) -> None:
        """Add an LD score column already aligned to ``self.snp``."""
        v = np.asarray(values_ref_order, np.float32)
        if v.shape != (self.snp.size,):
            raise ValueError("column must have one value per reference SNP")
        self.sets[name] = v
        self.Mset[name] = float(M)

    def align(self, snps: np.ndarray):
        """Row positions of ``snps`` in reference order (NaN where absent)."""
        return self._idx.reindex(np.asarray(snps).astype(str)).to_numpy()

    def add_bundle(self, out_dir: str, tag: str, snps_file: str = "snps_all.npy",
                   min_coverage: float = 0.9, prefix: str = ""):
        """Add every annotation of a one-pass LD score bundle.

        Returns ``(names, overlap)`` with ``overlap`` the (S x n_baseline)
        overlap rows if the bundle carries them, else ``None``.
        """
        names, M, L, snps, overlap = fio.read_bundle(out_dir, tag, snps_file)
        row = self.align(snps)
        have = np.isfinite(row)
        if have.mean() < min_coverage:
            raise SystemExit("bundle SNP order covers only %.1f%% of the "
                             "reference (wrong %s?)" % (100 * have.mean(), snps_file))
        Lfull = np.zeros((self.snp.size, L.shape[1]), np.float32)
        Lfull[row[have].astype(int)] = np.asarray(L)[have]
        # SNPs of the reference that the bundle lacks stay 0 (never NaN)
        keys = []
        for j, nm in enumerate(names):
            key = prefix + str(nm)
            self.sets[key] = Lfull[:, j]
            self.Mset[key] = float(M[j])
            keys.append(key)
        return keys, overlap

    # -- convenience -----------------------------------------------------
    def base_columns(self) -> List[np.ndarray]:
        return [self.Xbase[:, i] for i in range(self.Xbase.shape[1])]

    def columns(self, names: Sequence[str]) -> List[np.ndarray]:
        return [self.sets[n] for n in names]

    def M_for(self, names: Sequence[str]) -> np.ndarray:
        return np.asarray([self.Mset[n] for n in names], np.float64)


class TraitData:
    """Merged per-trait arrays in reference order.

    Mirrors ldsc's ``_read_ld_sumstats`` (inner merges on SNP) followed by the
    chi-square filter of ``estimate_h2`` for n_annot > 1.
    """

    def __init__(self, ref: RefData, sumstats_path: str,
                 chisq_max: Optional[float] = None):
        ss = fio.read_sumstats(sumstats_path)
        idx = pd.Series(np.arange(len(ss)), index=ss["SNP"].to_numpy())
        rows = idx.reindex(ref.snp).to_numpy()
        keep = np.isfinite(rows) & np.isfinite(ref.w_ld)
        self.rows = np.nonzero(keep)[0]           # ref-order positions
        srows = rows[keep].astype(int)
        Z = ss["Z"].to_numpy(np.float64)[srows]
        self.N = ss["N"].to_numpy(np.float64)[srows]
        self.n_merged = self.rows.size
        if chisq_max is None:
            chisq_max = max(0.001 * self.N.max(), 80.0)
        self.chisq_max = chisq_max
        ii = Z ** 2 < chisq_max
        self.rows = self.rows[ii]
        self.N = self.N[ii]
        self.chisq = (Z ** 2)[ii]
        self.w_ld = ref.w_ld[self.rows]
        self.n = self.rows.size
        self.name = os.path.basename(sumstats_path)
        for suf in (".sumstats.gz", ".sumstats"):
            if self.name.endswith(suf):
                self.name = self.name[:-len(suf)]


# ---------------------------------------------------------------- weights


def hsq_weights(ld, w_ld, N, M, hsq, intercept=1.0):
    """ldsc's heritability regression weights (``Hsq.weights``)."""
    hsq = min(max(hsq, 0.0), 1.0)
    ld = np.fmax(ld, 1.0)
    w_ld = np.fmax(w_ld, 1.0)
    c = hsq * N / M
    return 1.0 / (2 * np.square(intercept + c * ld)) / w_ld


def wls_coef(X, y, w_sqrt):
    """Plain weighted least squares via SVD (``np.linalg.lstsq``).

    Kept for reference/testing: the jackknife path solves the normal
    equations instead (as ldsc's ``LstsqJackknifeFast`` does).
    """
    wn = w_sqrt / w_sqrt.sum()
    Xw = X * wn[:, None]
    yw = y * wn
    return np.linalg.lstsq(Xw, yw, rcond=None)[0]


# ---------------------------------------------------------------- jackknife


def block_separators(n: int, n_blocks: int) -> np.ndarray:
    """Evenly spaced block boundaries, ``floor(linspace(0, n, n_blocks+1))``."""
    return np.floor(np.linspace(0, n, n_blocks + 1)).astype(int)


def block_jackknife_wls(Xw: np.ndarray, yw: np.ndarray, n_blocks: int = N_BLOCKS):
    """Delete-one-block jackknife of the normal-equation WLS estimate.

    ``Xw`` and ``yw`` are already multiplied by the (sum-normalised) square
    root weights.  Returns ``(est, delete, jk_cov, xtx_b, xty_b)`` where
    ``delete`` (n_blocks x p) are the leave-one-block-out estimates and
    ``jk_cov`` the jackknife covariance ``cov(pseudovalues) / n_blocks``.
    """
    n, p = Xw.shape
    seps = block_separators(n, n_blocks)
    xtx_b = np.zeros((n_blocks, p, p))
    xty_b = np.zeros((n_blocks, p))
    for b in range(n_blocks):
        s, e = seps[b], seps[b + 1]
        xb = Xw[s:e]
        xtx_b[b] = xb.T @ xb
        xty_b[b] = xb.T @ yw[s:e]
    xtx = xtx_b.sum(0)
    xty = xty_b.sum(0)
    est = np.linalg.solve(xtx, xty)
    delete = np.stack([np.linalg.solve(xtx - xtx_b[b], xty - xty_b[b])
                       for b in range(n_blocks)])
    pseudo = n_blocks * est[None, :] - (n_blocks - 1) * delete
    jk_cov = np.cov(pseudo.T, ddof=1) / n_blocks
    return est, delete, jk_cov, xtx_b, xty_b


def ratio_jackknife(est: np.ndarray, numer_delete: np.ndarray,
                    denom_delete: np.ndarray):
    """Block jackknife of a ratio from numerator/denominator delete values."""
    n_blocks = numer_delete.shape[0]
    pseudo = n_blocks * est[None, :] - (n_blocks - 1) * numer_delete / denom_delete
    cov = np.atleast_2d(np.cov(pseudo.T, ddof=1) / n_blocks)
    return cov, np.sqrt(np.diag(cov))


# ---------------------------------------------------------------- the model


@dataclass
class H2Result:
    """Everything ldsc reports for one partitioned-h2 regression."""
    coef: np.ndarray            # per-annotation tau (per-SNP h2)
    coef_se: np.ndarray
    coef_cov: np.ndarray
    intercept: float
    intercept_se: float
    tot: float                  # total observed-scale h2
    tot_se: float
    cat: np.ndarray             # M * coef
    prop: np.ndarray            # cat / tot
    prop_cov: np.ndarray
    prop_se: np.ndarray
    M_annot: np.ndarray
    Nbar: float
    n_snp: int
    n_blocks: int
    mean_chisq: float
    lambda_gc: float
    coef_delete: np.ndarray = field(repr=False)   # (n_blocks, n_annot), /Nbar

    @property
    def z(self) -> np.ndarray:
        return self.coef / self.coef_se

    @property
    def enrichment(self) -> np.ndarray:
        """Non-overlap enrichment ``(cat / M) / (tot / M_tot)``."""
        M_tot = float(self.M_annot.sum())
        return (self.cat / self.M_annot) / (self.tot / M_tot)

    @property
    def ratio(self):
        if self.mean_chisq > 1:
            return (self.intercept - 1) / (self.mean_chisq - 1)
        return np.nan

    def tau_star(self, sd_annot: np.ndarray, M_ref: float) -> np.ndarray:
        """Standardised effect ``tau * sd(annot) * M_ref / h2`` (Gazal et al. 2017).

        ``M_ref`` is the number of reference SNPs that h2 is defined over:
        with the default ``.l2.M_5_50`` counts, the number of common SNPs,
        i.e. the ``M_5_50`` of an all-ones ``base`` column (5,961,159 for
        1000G Phase 3 EUR).  It is not ``M_annot.sum()``: with overlapping
        or continuous annotations (any baselineLD design) that sum counts
        SNPs many times over (about 18x for baselineLD v2.2).
        ``sd_annot`` is the per-SNP standard deviation of each annotation
        over the same ``M_ref`` SNPs (``sqrt(p (1 - p))`` for a binary
        annotation covering a proportion ``p``).
        """
        return self.coef * np.asarray(sd_annot) * float(M_ref) / self.tot


def fit_h2(td: TraitData, cols: Sequence[np.ndarray], M_annot,
           n_blocks: int = N_BLOCKS) -> H2Result:
    """Fit ldsc's partitioned h2 regression for one trait and one design.

    ``cols`` are reference-order LD score arrays (raw, unfloored); ``M_annot``
    the matching ``M_5_50`` counts.  The arithmetic is exactly that of
    ``run_model`` in the validated research code; this function additionally
    keeps the jackknife delete values so that overlap-based enrichment can
    be reported.
    """
    n = td.n
    X = np.column_stack([c[td.rows] for c in cols]).astype(np.float64)
    if not np.isfinite(X).all():
        raise ValueError("NaN in the LD score design: an annotation lacks "
                         "scores for some regression SNPs")
    n_annot = X.shape[1]
    M_annot = np.asarray(M_annot, np.float64).reshape(n_annot)
    M_tot = float(M_annot.sum())
    x_tot = X.sum(axis=1)
    y = td.chisq
    N = td.N
    tot_agg = M_tot * (y.mean() - 1.0) / np.mean(x_tot * N)
    w = hsq_weights(x_tot, td.w_ld, N, M_tot, tot_agg)
    Nbar = N.mean()
    Xs = np.column_stack([X * (N / Nbar)[:, None], np.ones(n)])
    # partitioned h2 path: old_weights=True -- single WLS with the
    # aggregate-hsq initial weights, NO IRWLS iterations (sumstats.py:340)
    w_sqrt = np.sqrt(w)            # multiplier = sqrt(raw w), sum-normalized
    wn = w_sqrt / w_sqrt.sum()
    Xw = Xs * wn[:, None]
    yw = y * wn
    n_blocks = min(n_blocks, n)
    est, delete, jk_cov, _, _ = block_jackknife_wls(Xw, yw, n_blocks)
    coef = est[:n_annot] / Nbar
    coef_se = np.sqrt(np.diag(jk_cov)[:n_annot]) / Nbar
    coef_cov = jk_cov[:n_annot, :n_annot] / Nbar ** 2
    tot = float((M_annot * coef).sum())
    cat = M_annot * coef
    cat_cov = np.outer(M_annot, M_annot) * coef_cov
    tot_se = float(np.sqrt(cat_cov.sum()))
    # proportion of h2 via the ratio jackknife (ldsc ``_prop``)
    numer_delete = M_annot[None, :] * delete[:, :n_annot] / Nbar
    denom_delete = numer_delete.sum(axis=1)[:, None] * np.ones((1, n_annot))
    prop = cat / tot
    prop_cov, prop_se = ratio_jackknife(prop, numer_delete, denom_delete)
    return H2Result(coef=coef, coef_se=coef_se, coef_cov=coef_cov,
                    intercept=float(est[n_annot]),
                    intercept_se=float(np.sqrt(jk_cov[n_annot, n_annot])),
                    tot=tot, tot_se=tot_se, cat=cat, prop=prop,
                    prop_cov=prop_cov, prop_se=prop_se, M_annot=M_annot,
                    Nbar=float(Nbar), n_snp=n, n_blocks=n_blocks,
                    mean_chisq=float(y.mean()),
                    lambda_gc=float(np.median(y) / 0.4549),
                    coef_delete=delete[:, :n_annot] / Nbar)


def run_model(td: TraitData, cols: Sequence[np.ndarray], M_annot,
              n_blocks: int = N_BLOCKS):
    """The validated research entry point: ``(coef, coef_se, intercept, tot)``."""
    r = fit_h2(td, cols, M_annot, n_blocks)
    return r.coef, r.coef_se, r.intercept, r.tot


# ---------------------------------------------------------------- enrichment


def overlap_output(res: H2Result, overlap_matrix: np.ndarray, M_tot: float,
                   names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """ldsc's ``--overlap-annot --print-coefficients`` results table.

    ``overlap_matrix[i, j]`` is the number of (common) SNPs in both annotation
    i and j; ``M_tot`` the number of common SNPs.  Enrichment_p is the
    t-test (df = n_blocks) of the difference between per-SNP h2 inside and
    outside the annotation, as in ldsc.
    """
    n_annot = res.coef.size
    M_annot = res.M_annot
    overlap_prop = overlap_matrix / M_annot[None, :]
    prop_h2 = overlap_prop @ res.prop
    prop_h2_var = np.diag(overlap_prop @ res.prop_cov @ overlap_prop.T)
    prop_h2_se = np.sqrt(np.maximum(0, prop_h2_var))
    prop_M = M_annot / M_tot
    enrichment = prop_h2 / prop_M
    enrichment_se = prop_h2_se / prop_M
    diff = np.zeros((n_annot, n_annot))
    for i in range(n_annot):
        if M_tot != M_annot[i]:
            diff[i] = overlap_matrix[i] / M_annot[i] - \
                (M_annot - overlap_matrix[i]) / (M_tot - M_annot[i])
    diff_est = diff @ res.coef
    diff_se = np.sqrt(np.diag(diff @ res.coef_cov @ diff.T))
    with np.errstate(divide="ignore", invalid="ignore"):
        diff_p = np.where(diff_se == 0, np.nan,
                          2 * tdist.sf(np.abs(diff_est / diff_se), res.n_blocks))
    if names is None:
        names = ["CAT_%d" % i for i in range(n_annot)]
    return pd.DataFrame({
        "Category": list(names),
        "Prop._SNPs": prop_M, "Prop._h2": prop_h2, "Prop._h2_std_error": prop_h2_se,
        "Enrichment": enrichment, "Enrichment_std_error": enrichment_se,
        "Enrichment_p": diff_p,
        "Coefficient": res.coef, "Coefficient_std_error": res.coef_se,
        "Coefficient_z-score": res.coef / res.coef_se,
    })


def summary_lines(res: H2Result, names: Optional[Sequence[str]] = None) -> List[str]:
    """The headline block of an ldsc h2 log."""
    out = ["Total Observed scale h2: %.4g (%.4g)" % (res.tot, res.tot_se)]
    if names is not None:
        out.append("Categories: " + " ".join(names))
    out.append("Lambda GC: %.4g" % res.lambda_gc)
    out.append("Mean Chi^2: %.4g" % res.mean_chisq)
    out.append("Intercept: %.4g (%.4g)" % (res.intercept, res.intercept_se))
    if res.mean_chisq > 1:
        r = res.ratio
        if r < 0:
            out.append("Ratio < 0 (usually indicates GC correction).")
        else:
            out.append("Ratio: %.4g (%.4g)" % (r, res.intercept_se / (res.mean_chisq - 1)))
    else:
        out.append("Ratio: NA (mean chi^2 < 1)")
    return out


def coef_p(z) -> np.ndarray:
    """Two-sided normal p-value of a coefficient z-score."""
    return 2 * norm.sf(np.abs(z))
