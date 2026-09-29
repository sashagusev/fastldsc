"""Command line interface: ``fastldsc <subcommand>``.

Subcommands
-----------
annot          gene sets + coordinates + window -> per-chromosome .annot.gz
l2             LD scores for many annotations in one genotype pass
overlap        cache the baseline annotation overlap (for enrichment)
h2             one partitioned-h2 model (ldsc --h2 --overlap-annot mirror)
scan           many (candidate set x trait) models with cached references
screen         thousands of candidates per trait via bordered solves
validate       recompute banked ldsc runs and report the deviations
parse-results  collect ldsc .results files into one CSV
fdr            empirical FDR from negative-control traits
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import __version__
from . import annot as fannot
from . import io as fio
from . import l2 as fl2
from .overlap import OverlapCache, build_cache
from .screen import check_against_refit, screen_candidates
from .sldsc import (N_BLOCKS, RefData, TraitData, fit_h2, overlap_output,
                    summary_lines)


def _chroms(arg: str) -> List[int]:
    if arg in (None, "", "all", "0"):
        return list(fio.AUTOSOMES)
    out: List[int] = []
    for part in arg.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _split(arg: Optional[str]) -> List[str]:
    return [x for x in (arg or "").split(",") if x]


def _add_ref_args(ap: argparse.ArgumentParser, need_w: bool = True) -> None:
    ap.add_argument("--ref-ld-chr", required=True,
                    help="per-chromosome prefix of the baseline LD scores, "
                         "e.g. .../baselineLD.  (first entry if comma list)")
    ap.add_argument("--w-ld-chr", required=need_w,
                    help="per-chromosome prefix of the regression-weight LD scores")
    ap.add_argument("--chr", default="all", help="chromosomes, e.g. 1-22 (default all)")
    ap.add_argument("--not-M-5-50", action="store_true",
                    help="use .l2.M instead of .l2.M_5_50")


def _sumstats_path(trait: str, sumstats_dir: Optional[str]) -> str:
    if os.path.exists(trait):
        return trait
    if sumstats_dir is None:
        raise FileNotFoundError(trait)
    for suf in (".sumstats.gz", ".sumstats", ""):
        p = os.path.join(sumstats_dir, trait + suf)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no sumstats for %s in %s" % (trait, sumstats_dir))


def _set_prefix(name: str, l2_dir: Optional[str]) -> str:
    """``<l2_dir>/<name>/<name>.`` (research layout) or ``<l2_dir>/<name>.``."""
    if l2_dir is None:
        return name if name.endswith(".") else name + "."
    for cand in (os.path.join(l2_dir, name, name + "."), os.path.join(l2_dir, name + ".")):
        if fio.present_chroms(cand):
            return cand
    raise FileNotFoundError("no LD scores for set %s under %s" % (name, l2_dir))


# ================================================================ annot


def cmd_annot(a) -> None:
    coords = _load_coords(a)
    sets = fannot.load_sets(a.sets)
    if a.exclude_mhc:
        bad = fannot.mhc_genes(coords)
        sets = {k: [g for g in v if g not in bad] for k, v in sets.items()}
    if a.write_sets_npz:
        fio.write_sets_npz(a.write_sets_npz, sets)
    paths = fannot.write_gene_set_annots(coords, sets, a.bfile, a.out, _chroms(a.chr),
                                         window=a.window, thin=not a.full,
                                         add_base=a.add_base)
    print("wrote %d annot files (%d sets)" % (len(paths), len(sets)))


def _load_coords(a) -> pd.DataFrame:
    if getattr(a, "gtf", None):
        coords = fannot.gene_coords_from_gtf(a.gtf)
        if getattr(a, "write_coords", None):
            fio.write_gene_coords(a.write_coords, coords)
        return coords
    if getattr(a, "coords", None):
        return fio.read_gene_coords(a.coords)
    raise SystemExit("need --coords or --gtf")


# ================================================================ l2


def cmd_l2(a) -> None:
    chroms = _chroms(a.chr)
    print_snps = set(fio.read_snp_list(a.print_snps).tolist()) if a.print_snps else None
    if a.sets:
        coords = _load_coords(a)
        sets = fannot.load_sets(a.sets)
        if a.exclude_mhc:
            bad = fannot.mhc_genes(coords)
            sets = {k: [g for g in v if g not in bad] for k, v in sets.items()}
        names = list(sets)
        print("annotations: %d" % len(names), flush=True)
    elif a.annot:
        sets, names = None, None
    else:
        sets, names = None, ["L2"]
    if a.bundle_dir:
        os.makedirs(a.bundle_dir, exist_ok=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    L_parts, snp_parts = [], []
    M_tot = None
    M5_tot = None
    ov_tot = None
    for c in chroms:
        bfile = fio.sub_chr(a.bfile, c)
        if sets is not None:
            bim = fio.read_bim(bfile + ".bim")
            A = fannot.gene_set_annotation(coords, sets, bim["BP"].to_numpy(), c, a.window)
            if a.add_base:
                A = np.hstack([np.ones((A.shape[0], 1), A.dtype), A])
                names = ["base"] + list(sets)
        elif a.annot:
            _, adf = fio.read_annot(fio.find_compressed(fio.sub_chr(a.annot, c) + ".annot"))
            A = adf.to_numpy(np.float32)
            names = list(adf.columns)
        else:
            A = None
        if a.out and A is not None and not a.no_annot:
            # keep the annotation next to the LD scores so that
            # `fastldsc h2 --overlap-annot` (and stock ldsc) can use it
            fio.write_annot(fio.sub_chr(a.out, c) + ".annot.gz", A, names)
        base_A = None
        if a.overlap_with:
            _, bdf = fio.read_annot(fio.find_compressed(fio.sub_chr(a.overlap_with, c) + ".annot"))
            base_A = bdf.to_numpy(np.float32)
        ids, L, M, M5, ov = fl2.ld_scores_chromosome(
            bfile, c, A, names, print_snps, ld_wind_cm=a.ld_wind_cm,
            chunk=a.chunk_size, out_prefix=a.out, base_annot=base_A)
        M_tot = M if M_tot is None else M_tot + M
        M5_tot = M5 if M5_tot is None else M5_tot + M5
        if ov is not None:
            ov_tot = ov if ov_tot is None else ov_tot + ov
        if a.bundle_dir:
            L_parts.append(L)
            snp_parts.append(ids)
    if a.bundle_dir:
        p = fio.bundle_paths(a.bundle_dir, a.tag)
        np.save(p["l2"], np.vstack(L_parts))
        np.save(p["M"], np.asarray(M5_tot, float))
        np.save(p["names"], np.array(names, dtype=object))
        if ov_tot is not None:
            np.save(p["overlap"], ov_tot)
        # name the SNP-order file after the chromosome span it covers, so a
        # single-chromosome run can never be mistaken for the genome-wide one
        sp = "snps_all.npy" if chroms == list(fio.AUTOSOMES) else \
            "snps_chr%s.npy" % "_".join(str(c) for c in chroms)
        np.save(os.path.join(a.bundle_dir, sp), np.concatenate(snp_parts))
        print("bundle %s written to %s (%d annotations)" % (a.tag, a.bundle_dir, len(names)))
    print("done", flush=True)


# ================================================================ overlap


def cmd_overlap(a) -> None:
    path = build_cache(a.annot_chr, a.frqfile_chr, a.out, _chroms(a.chr),
                       save_annot=not a.no_annot)
    print("wrote", path)


# ================================================================ h2


def _load_ref(a, extra: Optional[Dict[str, str]] = None, verbose: bool = True) -> RefData:
    prefixes = _split(a.ref_ld_chr)
    base = prefixes[0]
    sets = dict(extra or {})
    for i, p in enumerate(prefixes[1:], 1):
        sets["ref%d" % i] = p
    return RefData(base, a.w_ld_chr, sets, _chroms(a.chr),
                   common=not a.not_M_5_50, verbose=verbose)


def cmd_h2(a) -> None:
    ref = _load_ref(a)
    td = TraitData(ref, a.h2, chisq_max=a.chisq_max)
    keys, names = [], ["%s_0" % c for c in ref.base_cols]
    for i, name in enumerate(ref.set_names, 1):        # one entry per file
        for k in ref.set_names[name]:
            keys.append(k)
            names.append("%s_%d" % (k.split(":", 1)[1] if ":" in k else "L2", i))
    cols = ref.base_columns() + ref.columns(keys)
    M = np.concatenate([ref.Mbase, ref.M_for(keys)])
    t0 = time.time()
    res = fit_h2(td, cols, M, n_blocks=a.n_blocks)
    lines = ["fastldsc %s h2" % __version__,
             "Read summary statistics for %d SNPs (after merging)." % td.n_merged,
             "Removed %d SNPs with chi^2 > %g (%d SNPs remain)"
             % (td.n_merged - td.n, td.chisq_max, td.n)]
    lines += summary_lines(res, names)
    lines.append("Regression time: %.2fs" % (time.time() - t0))
    if a.overlap_annot:
        prefixes = _split(a.ref_ld_chr)
        if a.overlap_cache:
            cache = OverlapCache(a.overlap_cache)
            S = np.hstack([cache.vector_for_prefix(p) for p in prefixes[1:]]) \
                if len(prefixes) > 1 else np.zeros((cache.M_tot, 0))
            overlap, M_tot = cache.full_matrix(S), cache.M_tot
        else:
            overlap, M_tot, _ = fio.read_overlap_matrix(prefixes, a.frqfile_chr, _chroms(a.chr))
        df = overlap_output(res, overlap, M_tot, names)
        df.to_csv(a.out + ".results", sep="\t", index=False)
        lines.append("Results printed to %s.results" % a.out)
    else:
        df = pd.DataFrame({"Category": names, "Coefficient": res.coef,
                           "Coefficient_std_error": res.coef_se,
                           "Coefficient_z-score": res.z,
                           "Prop._SNPs": M / M.sum(), "Enrichment": res.enrichment})
        df.to_csv(a.out + ".results", sep="\t", index=False)
    with open(a.out + ".log", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ================================================================ scan


def cmd_scan(a) -> None:
    """Marginal (and conditional) models for many candidate sets x traits."""
    ref = _load_ref(a)
    cand: List[str] = []
    if a.bundle_dir:
        for tag in _split(a.tags):
            keys, _ = ref.add_bundle(a.bundle_dir, tag, a.snps_file)
            cand.extend(keys)
    for s in _split(a.sets):
        ref.add_prefix(s, _set_prefix(s, a.l2_dir))
        cand.extend(ref.set_names[s])
    if a.set_filter:
        cand = [c for c in cand if a.set_filter in c]
    cond: List[str] = []
    for s in _split(a.cond):
        if s not in ref.sets:
            ref.add_prefix(s, _set_prefix(s, a.l2_dir))
        cond.append(s)
    cand = [c for c in cand if c not in cond]
    print("candidates: %d, conditioning sets: %d" % (len(cand), len(cond)), flush=True)
    base_cols = ref.base_columns()
    Mbase = list(ref.Mbase)
    nb = len(Mbase)
    rows = []
    for trait in _split(a.traits):
        td = TraitData(ref, _sumstats_path(trait, a.sumstats_dir))
        t0 = time.time()
        for nm in cand:
            x = ref.sets[nm].astype(np.float64)
            r = fit_h2(td, base_cols + [x], np.asarray(Mbase + [ref.Mset[nm]]), a.n_blocks)
            row = dict(set=nm, trait=td.name, M=ref.Mset[nm], n_snp=td.n,
                       marg_coef=r.coef[-1], marg_se=r.coef_se[-1], marg_z=r.z[-1],
                       marg_enrichment=r.enrichment[-1], h2=r.tot, h2_se=r.tot_se,
                       intercept=r.intercept)
            if cond:
                ccols = [ref.sets[c].astype(np.float64) for c in cond]
                cM = [ref.Mset[c] for c in cond]
                r2 = fit_h2(td, base_cols + ccols + [x],
                            np.asarray(Mbase + cM + [ref.Mset[nm]]), a.n_blocks)
                row.update(cond_coef=r2.coef[-1], cond_se=r2.coef_se[-1], cond_z=r2.z[-1],
                           cond_enrichment=r2.enrichment[-1], cond_h2=r2.tot)
                for j, c in enumerate(cond):
                    row["z_%s" % c] = r2.z[nb + j]
            rows.append(row)
        print("%s done (%d sets, %.0fs)" % (td.name, len(cand), time.time() - t0), flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    print("wrote %s (%d rows)" % (a.out, len(rows)))


# ================================================================ screen


def cmd_screen(a) -> None:
    ref = _load_ref(a)
    for g in _split(a.global_sets):
        if g not in ref.sets:
            ref.add_prefix(g, _set_prefix(g, a.l2_dir))
    gsets = _split(a.global_sets)
    base_cols = ref.base_columns() + ref.columns(gsets)
    base_M = list(ref.Mbase) + [ref.Mset[g] for g in gsets]
    rows = []
    checks = []
    rng = np.random.default_rng(0)
    for tag in _split(a.tags):
        names, Mc, Lc, snps, _ = fio.read_bundle(a.bundle_dir, tag, a.snps_file)
        row_of_ref = ref.align(snps)
        have = np.isfinite(row_of_ref)
        if have.mean() < 0.9:
            raise SystemExit("SNP-order file covers only %.1f%% of reference SNPs "
                             "- wrong %s?" % (100 * have.mean(), a.snps_file))
        L = np.zeros((ref.snp.size, Lc.shape[1]), np.float32)
        L[row_of_ref[have].astype(int)] = np.asarray(Lc)[have]
        sel = np.arange(names.size)
        if a.set_filter:
            sel = np.asarray([i for i in sel if a.set_filter in names[i]])
        for trait in _split(a.traits):
            td = TraitData(ref, _sumstats_path(trait, a.sumstats_dir))
            t0 = time.time()
            coef, se, zz = screen_candidates(td, base_cols, base_M, L[:, sel],
                                             chunk=a.chunk, n_blocks=a.n_blocks)
            for k, i in enumerate(sel):
                rows.append(dict(set=names[i], trait=td.name, coef=coef[k], se=se[k],
                                 z=zz[k], M=Mc[i]))
            if a.check:
                which = rng.choice(sel.size, min(a.check, sel.size), replace=False)
                c2, s2, z2 = check_against_refit(td, base_cols, base_M, L[:, sel], Mc[sel],
                                                 which, a.n_blocks)
                for k, w in enumerate(which):
                    checks.append(dict(set=names[sel[w]], trait=td.name,
                                       screen_z=zz[w], refit_z=z2[k],
                                       screen_coef=coef[w], refit_coef=c2[k],
                                       screen_se=se[w], refit_se=s2[k]))
            print("%s %s %d candidates %.1fs" % (tag, td.name, sel.size, time.time() - t0),
                  flush=True)
        del L
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print("wrote", a.out, len(df), flush=True)
    if checks:
        cdf = pd.DataFrame(checks)
        cdf["dz"] = cdf.screen_z - cdf.refit_z
        cdf.to_csv(a.out + ".check.csv", index=False)
        print("CHECK %d refits: max |dz| = %.4f, max rel |dcoef| = %.2e, max rel |dse| = %.2e"
              % (len(cdf), cdf.dz.abs().max(),
                 (abs(cdf.screen_coef - cdf.refit_coef) / abs(cdf.refit_coef)).max(),
                 (abs(cdf.screen_se - cdf.refit_se) / cdf.refit_se).max()))


# ================================================================ validate


def _resolve(path: str, fallback_dir: Optional[str], kind: str) -> str:
    """Use the path recorded in a log if it exists, else look it up by name."""
    if kind == "prefix":
        if fio.present_chroms(path):
            return path
        name = os.path.basename(path.rstrip("."))
        if fallback_dir:
            return _set_prefix(name, fallback_dir)
    else:
        if os.path.exists(path):
            return path
        if fallback_dir:
            return _sumstats_path(os.path.basename(path), fallback_dir)
    raise FileNotFoundError(path)


def cmd_validate(a) -> None:
    results = sorted(glob.glob(os.path.join(a.ldsc_results, "*.results"))) \
        if os.path.isdir(a.ldsc_results) else sorted(glob.glob(a.ldsc_results))
    if a.pattern:
        results = [r for r in results if re.search(a.pattern, os.path.basename(r))]
    if not results:
        raise SystemExit("no .results files matched")
    rng = np.random.default_rng(a.seed)
    picks = list(rng.choice(results, min(a.n, len(results)), replace=False)) \
        if a.n and a.n < len(results) else results
    jobs = []
    for r in picks:
        log = fio.parse_ldsc_log(r[:-len(".results")] + ".log")
        if not log["ref_ld_chr"] or not log["h2"]:
            print("skip (no call in log):", r)
            continue
        jobs.append((r, log))
    need = sorted({p for _, log in jobs for p in log["ref_ld_chr"][1:]})
    ref = _load_ref(a, extra={}, verbose=True)
    prefix_of: Dict[str, str] = {}
    for p in need:
        rp = _resolve(p, a.l2_dir, "prefix")
        prefix_of[p] = rp
        ref.add_prefix(p, rp)
    cache = OverlapCache(a.overlap_cache) if a.overlap_cache else None
    svec: Dict[str, np.ndarray] = {}
    if cache is not None:
        for p in need:
            svec[p] = cache.vector_for_prefix(prefix_of[p])
    tcache: Dict[str, TraitData] = {}
    rows = []
    for rpath, log in jobs:
        ss = _resolve(log["h2"], a.sumstats_dir, "sumstats")
        if ss not in tcache:
            tcache[ss] = TraitData(ref, ss)
        td = tcache[ss]
        extra = log["ref_ld_chr"][1:]
        keys = [k for p in extra for k in ref.set_names[p]]
        cols = ref.base_columns() + ref.columns(keys)
        M = np.concatenate([ref.Mbase, ref.M_for(keys)])
        t0 = time.time()
        res = fit_h2(td, cols, M)
        dt = time.time() - t0
        got = fio.read_results(rpath)
        row = dict(results=os.path.basename(rpath), n_annot=len(M), secs=dt,
                   n_snp_fast=td.n, n_snp_ldsc=log["n_snps"],
                   h2_fast=res.tot, h2_ldsc=log["tot_h2"],
                   h2_se_fast=res.tot_se, h2_se_ldsc=log["tot_h2_se"],
                   int_fast=res.intercept, int_ldsc=log["intercept"],
                   int_se_fast=res.intercept_se, int_se_ldsc=log["intercept_se"])
        want = got["Coefficient"].to_numpy(float)
        want_se = got["Coefficient_std_error"].to_numpy(float)
        want_z = got["Coefficient_z-score"].to_numpy(float)
        row["max_abs_dcoef"] = np.abs(res.coef - want).max()
        row["max_rel_dcoef"] = (np.abs(res.coef - want)
                                / np.maximum(np.abs(want), 1e-10 * np.abs(want).max())).max()
        row["max_dcoef_in_se"] = (np.abs(res.coef - want) / want_se).max()
        row["max_rel_dse"] = (np.abs(res.coef_se - want_se) / want_se).max()
        row["max_abs_dz"] = np.abs(res.z - want_z).max()
        row["last_dz"] = res.z[-1] - want_z[-1]
        if cache is not None and "Enrichment" in got.columns:
            S = np.hstack([svec[p] for p in extra]) if extra else np.zeros((cache.M_tot, 0))
            df = overlap_output(res, cache.full_matrix(S), cache.M_tot)
            for col, key in (("Enrichment", "enr"), ("Enrichment_std_error", "enr_se"),
                             ("Prop._h2", "prop_h2"), ("Prop._h2_std_error", "prop_h2_se"),
                             ("Prop._SNPs", "prop_snps")):
                w = got[col].to_numpy(float)
                row["max_abs_d%s" % key] = np.abs(df[col].to_numpy() - w).max()
                # relative to the ldsc value, floored at 1e-10 x the column's
                # scale so that an exact 0 (the base category's SE) or a huge
                # Enrichment of a near-empty continuous annotation cannot
                # dominate
                floor = 1e-10 * np.abs(w).max()
                row["max_rel_d%s" % key] = (np.abs(df[col].to_numpy() - w)
                                            / np.maximum(np.abs(w), floor)).max()
            wp = pd.to_numeric(got["Enrichment_p"], errors="coerce").to_numpy(float)
            fp = df["Enrichment_p"].to_numpy(float)
            ok = np.isfinite(wp) & np.isfinite(fp)
            row["max_abs_denr_p"] = np.abs(fp[ok] - wp[ok]).max() if ok.any() else np.nan
            row["last_enr_fast"], row["last_enr_ldsc"] = df["Enrichment"].iloc[-1], got["Enrichment"].iloc[-1]
        rows.append(row)
        print("%-55s annots=%3d dcoef=%.1e dse=%.1e |dz|=%.1e%s (%.1fs)"
              % (row["results"][:55], len(M), row["max_rel_dcoef"], row["max_rel_dse"],
                 row["max_abs_dz"],
                 (" denr=%.1e" % row["max_rel_denr"]) if "max_rel_denr" in row else "",
                 dt), flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    df = pd.DataFrame(rows)
    print("VALIDATION over %d runs (%d distinct traits):" % (len(df), len(tcache)))
    print("  max rel |dcoef| %.3e   max |dcoef|/SE %.3e   max rel |dse| %.3e   max |dz| %.3e"
          % (df.max_rel_dcoef.max(), df.max_dcoef_in_se.max(), df.max_rel_dse.max(),
             df.max_abs_dz.max()))
    print("  max |dh2| %.3e   max |dintercept| %.3e   (log values have 4 decimals)"
          % ((df.h2_fast - df.h2_ldsc).abs().max(), (df.int_fast - df.int_ldsc).abs().max()))
    if "max_rel_denr" in df:
        print("  max rel |dEnrichment| %.3e   max rel |dEnrichment_se| %.3e   "
              "max rel |dProp_h2| %.3e   max |dEnrichment_p| %.3e"
              % (df.max_rel_denr.max(), df.max_rel_denr_se.max(),
                 df.max_rel_dprop_h2.max(), df.max_abs_denr_p.max()))
    print("  seconds per model: mean %.2f, median %.2f, max %.2f"
          % (df.secs.mean(), df.secs.median(), df.secs.max()))


# ================================================================ parse-results


def _results_rows(path: str, trait_class=None) -> List[dict]:
    base = os.path.basename(path)[:-len(".results")]
    if "__" in base:
        job, trait = base.split("__", 1)
    else:
        job, trait = base, ""
    df = fio.read_results(path)
    log = fio.parse_ldsc_log(path[:-len(".results")] + ".log")
    out = []
    for i, r in df.iterrows():
        out.append(dict(run=job, trait=trait, category=r["Category"], row=i,
                        prop_snps=r.get("Prop._SNPs"), prop_h2=r.get("Prop._h2"),
                        prop_h2_se=r.get("Prop._h2_std_error"),
                        enrichment=r.get("Enrichment"), enrichment_se=r.get("Enrichment_std_error"),
                        enrichment_p=r.get("Enrichment_p"), coef=r.get("Coefficient"),
                        coef_se=r.get("Coefficient_std_error"), coef_z=r.get("Coefficient_z-score"),
                        h2_obs=log["tot_h2"], h2_int=log["intercept"]))
    return out


def cmd_parse_results(a) -> None:
    rows = []
    for p in sorted(glob.glob(os.path.join(a.h2_dir, "*.results"))):
        try:
            rr = _results_rows(p)
            if a.last_only:
                rr = rr[-a.last_only:]
            rows.extend(rr)
        except Exception as e:  # fail loud, keep going
            print("PARSE_FAIL %s: %s" % (p, e))
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("wrote %s (%d rows)" % (a.out, len(rows)))


# ================================================================ fdr


def empirical_fdr(df: pd.DataFrame, control_traits: List[str], z_col: str,
                  trait_col: str = "trait", one_sided: bool = True) -> pd.DataFrame:
    """Empirical p and BH q from the pooled control-trait z distribution.

    Nominal conditional tau* z-scores are not calibrated for sparse
    annotations; the z-scores the same annotations obtain on traits that
    should carry no signal give an empirical null with the same LD, sparsity
    and jackknife instability.  For each test-trait row the empirical p is
    the fraction of control z's at least as extreme (with +1 smoothing).
    """
    is_ctrl = df[trait_col].isin(control_traits)
    null = df.loc[is_ctrl, z_col].to_numpy(float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        raise SystemExit("no control-trait rows found")
    test = df.loc[~is_ctrl].copy()
    z = test[z_col].to_numpy(float)
    if one_sided:
        null_sorted = np.sort(null)
        n_ge = null.size - np.searchsorted(null_sorted, z, side="left")
    else:
        null_sorted = np.sort(np.abs(null))
        n_ge = null.size - np.searchsorted(null_sorted, np.abs(z), side="left")
    test["p_empirical"] = (n_ge + 1) / (null.size + 1)
    p = test["p_empirical"].to_numpy()
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    qv = np.empty_like(q)
    qv[order] = np.minimum(q, 1.0)
    test["q_empirical"] = qv
    test["control_z_mean"] = null.mean()
    test["control_z_sd"] = null.std(ddof=1) if null.size > 1 else np.nan
    return test


def cmd_fdr(a) -> None:
    df = pd.read_csv(a.results)
    ctrl = _split(a.control_traits)
    out = empirical_fdr(df, ctrl, a.z_col, a.trait_col, one_sided=not a.two_sided)
    out.to_csv(a.out, index=False)
    null = df[df[a.trait_col].isin(ctrl)][a.z_col].dropna()
    thr = np.percentile(np.abs(null), 95)
    print("control rows: %d, mean z %.2f, sd %.2f (nominal 0, 1); "
          "empirical 5%% two-sided threshold |z| > %.2f"
          % (len(null), null.mean(), null.std(), thr))
    print("test rows: %d, q < 0.05: %d, q < 0.10: %d"
          % (len(out), (out.q_empirical < 0.05).sum(), (out.q_empirical < 0.10).sum()))
    print("wrote", a.out)


# ================================================================ simulate


def cmd_simulate(a) -> None:
    from .simulate import simulate_all
    paths = simulate_all(a.out, seed=a.seed, n_indiv=a.n_indiv, n_snp=a.n_snp,
                         chroms=_chroms(a.chr), n_traits=a.n_traits)
    for k in ("bfile", "frq", "coords", "sets_dir", "print_snps", "sumstats_dir",
              "base", "w", "traits", "controls", "window"):
        print("%-12s %s" % (k, paths[k]))


# ================================================================ parser


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fastldsc",
                                 description="fast stratified LD score regression "
                                             "for scanning many annotations")
    ap.add_argument("--version", action="version", version="fastldsc " + __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("annot", help="gene sets -> per-chromosome .annot.gz")
    p.add_argument("--sets", nargs="+", required=True,
                   help=".GeneSet files, a directory of them, or a sets .npz")
    p.add_argument("--coords", help="GENE CHR START END table")
    p.add_argument("--gtf", help="GTF (gene records) instead of --coords")
    p.add_argument("--write-coords", help="save the coordinates parsed from --gtf")
    p.add_argument("--bfile", required=True, help="per-chromosome plink prefix")
    p.add_argument("--out", required=True, help="output prefix (chromosome appended)")
    p.add_argument("--window", type=int, default=fannot.DEFAULT_WINDOW)
    p.add_argument("--chr", default="all")
    p.add_argument("--full", action="store_true", help="write CHR BP SNP CM columns too")
    p.add_argument("--add-base", action="store_true", help="prepend an all-ones base column")
    p.add_argument("--exclude-mhc", action="store_true")
    p.add_argument("--write-sets-npz")
    p.set_defaults(func=cmd_annot)

    p = sub.add_parser("l2", help="LD scores for many annotations in one pass")
    p.add_argument("--bfile", required=True, help="per-chromosome plink prefix")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--annot", help="per-chromosome .annot[.gz] prefix (thin or full)")
    g.add_argument("--sets", nargs="+", help="gene sets (.GeneSet files/dir/.npz)")
    p.add_argument("--coords")
    p.add_argument("--gtf")
    p.add_argument("--window", type=int, default=fannot.DEFAULT_WINDOW)
    p.add_argument("--exclude-mhc", action="store_true")
    p.add_argument("--add-base", action="store_true")
    p.add_argument("--ld-wind-cm", type=float, default=1.0)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--print-snps", help="one-column SNP list to restrict output rows")
    p.add_argument("--chr", default="all")
    p.add_argument("--out", help="ldsc-format output prefix (chromosome appended)")
    p.add_argument("--bundle-dir", help="write a one-pass .npy bundle here")
    p.add_argument("--tag", default="sets", help="bundle tag")
    p.add_argument("--overlap-with", help="baseline .annot prefix: also record overlap rows")
    p.add_argument("--no-annot", action="store_true",
                   help="do not write <out><chr>.annot.gz next to the LD scores")
    p.set_defaults(func=cmd_l2)

    p = sub.add_parser("overlap", help="cache the baseline annotation overlap matrix")
    p.add_argument("--annot-chr", required=True)
    p.add_argument("--frqfile-chr")
    p.add_argument("--chr", default="all")
    p.add_argument("--no-annot", action="store_true", help="do not save the annotation matrix")
    p.add_argument("--out", required=True, help="output prefix (.npz, .annot.npy)")
    p.set_defaults(func=cmd_overlap)

    p = sub.add_parser("h2", help="one partitioned-h2 model")
    p.add_argument("--h2", required=True, help="munged .sumstats.gz")
    _add_ref_args(p)
    p.add_argument("--overlap-annot", action="store_true")
    p.add_argument("--frqfile-chr")
    p.add_argument("--overlap-cache", help="from `fastldsc overlap` (faster than --frqfile-chr)")
    p.add_argument("--chisq-max", type=float)
    p.add_argument("--n-blocks", type=int, default=N_BLOCKS)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_h2)

    p = sub.add_parser("scan", help="many candidate sets x traits")
    _add_ref_args(p)
    p.add_argument("--traits", required=True, help="comma list of traits or sumstats paths")
    p.add_argument("--sumstats-dir")
    p.add_argument("--sets", help="comma list of sets with .l2.ldscore.gz files")
    p.add_argument("--l2-dir", help="directory holding <set>/<set>.<chr>.l2.ldscore.gz")
    p.add_argument("--bundle-dir")
    p.add_argument("--tags", help="comma list of bundle tags")
    p.add_argument("--snps-file", default="snps_all.npy")
    p.add_argument("--set-filter")
    p.add_argument("--cond", help="comma list of conditioning sets (added to the design)")
    p.add_argument("--n-blocks", type=int, default=N_BLOCKS)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("screen", help="bordered-solve screening of a bundle")
    _add_ref_args(p)
    p.add_argument("--traits", required=True)
    p.add_argument("--sumstats-dir")
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--tags", required=True)
    p.add_argument("--snps-file", default="snps_all.npy")
    p.add_argument("--global-sets", help="comma list of sets held in the fixed design")
    p.add_argument("--l2-dir")
    p.add_argument("--set-filter")
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--check", type=int, default=0, help="verify N random candidates by refit")
    p.add_argument("--n-blocks", type=int, default=N_BLOCKS)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("validate", help="recompute banked ldsc runs")
    p.add_argument("--ldsc-results", required=True, help="directory (or glob) of .results")
    p.add_argument("--pattern", help="regex on the .results basename")
    p.add_argument("--n", type=int, default=40, help="sample size (0 = all)")
    p.add_argument("--seed", type=int, default=3)
    _add_ref_args(p)
    p.add_argument("--l2-dir", help="where to find set LD scores if the logged path moved")
    p.add_argument("--sumstats-dir")
    p.add_argument("--overlap-cache", help="also compare Enrichment (from `fastldsc overlap`)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("parse-results", help="collect .results files into a CSV")
    p.add_argument("--h2-dir", required=True)
    p.add_argument("--last-only", type=int, default=0, help="keep only the last k rows")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_parse_results)

    p = sub.add_parser("simulate", help="write a small synthetic dataset (no external data)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-indiv", type=int, default=80)
    p.add_argument("--n-snp", type=int, default=600, help="SNPs per chromosome")
    p.add_argument("--chr", default="1-2")
    p.add_argument("--n-traits", type=int, default=3)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("fdr", help="empirical FDR from negative-control traits")
    p.add_argument("--results", required=True, help="CSV from scan/screen")
    p.add_argument("--control-traits", required=True, help="comma list")
    p.add_argument("--z-col", default="z")
    p.add_argument("--trait-col", default="trait")
    p.add_argument("--two-sided", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_fdr)
    return ap


def main(argv=None) -> None:
    ap = build_parser()
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
