"""End-to-end run of the command line on the synthetic dataset."""

import os

import numpy as np
import pandas as pd
import pytest

from fastldsc import io as fio
from fastldsc.cli import main


def test_annot_command(synth, tmp_path):
    out = str(tmp_path / "ann.")
    main(["annot", "--sets", synth["sets_dir"], "--coords", synth["coords"],
          "--bfile", synth["bfile"], "--out", out, "--chr", "1-2",
          "--window", synth["window"], "--write-sets-npz", str(tmp_path / "sets.npz")])
    meta, ann = fio.read_annot(out + "1.annot.gz")
    bim = fio.read_bim(synth["bfile"] + "1.bim")
    assert meta is None and ann.shape == (len(bim), 6)
    assert set(ann.columns) == set(synth["sets"])
    assert os.path.exists(tmp_path / "sets.npz")


def test_l2_from_annot_equals_l2_from_sets(synth, set_ldscores, tmp_path):
    ann = str(tmp_path / "ann.")
    main(["annot", "--sets", synth["sets_dir"], "--coords", synth["coords"],
          "--bfile", synth["bfile"], "--out", ann, "--chr", "1-2", "--window", synth["window"]])
    out = str(tmp_path / "viaannot.")
    main(["l2", "--bfile", synth["bfile"], "--annot", ann, "--print-snps", synth["print_snps"],
          "--chr", "1-2", "--out", out])
    a = fio.read_ldscore_chr(out, [1, 2])
    b = fio.read_ldscore_chr(set_ldscores, [1, 2])
    assert a["SNP"].tolist() == b["SNP"].tolist()
    for c in a.columns[1:]:
        assert np.array_equal(a[c].to_numpy(), b[c].to_numpy())
    assert np.array_equal(fio.read_M(out, 6, [1, 2]), fio.read_M(set_ldscores, 6, [1, 2]))


def test_bundle_matches_ldscore_files(synth, set_ldscores, ref):
    names, M, L, snps, overlap = fio.read_bundle(os.path.dirname(set_ldscores), "sets")
    assert list(names) == list(synth["sets"])
    df = fio.read_ldscore_chr(set_ldscores, [1, 2])
    assert df["SNP"].tolist() == list(snps)
    assert np.asarray(L) == pytest.approx(df.iloc[:, 1:].to_numpy(), abs=5e-4)
    assert M.tolist() == fio.read_M(set_ldscores, 6, [1, 2]).tolist()
    assert overlap.shape == (6, 6)          # overlap with the 6 baseline annotations
    assert overlap[:, 0].tolist() == M.tolist()   # overlap with 'base' = M_5_50


def test_h2_overlap_via_frq_and_cache_agree(synth, set_ldscores, tmp_path):
    ss = os.path.join(synth["sumstats_dir"], "trait0.sumstats.gz")
    ref_ld = synth["base"] + "," + set_ldscores
    o1 = str(tmp_path / "frq")
    main(["h2", "--h2", ss, "--ref-ld-chr", ref_ld, "--w-ld-chr", synth["w"], "--chr", "1-2",
          "--overlap-annot", "--frqfile-chr", synth["frq"], "--out", o1])
    r1 = fio.read_results(o1 + ".results")
    assert len(r1) == 12 and r1["Category"].iloc[0] == "baseL2_0"
    assert r1["Category"].iloc[6] == "set0L2_1"
    cache = str(tmp_path / "cache")
    main(["overlap", "--annot-chr", synth["base"], "--frqfile-chr", synth["frq"],
          "--chr", "1-2", "--out", cache])
    o2 = str(tmp_path / "cache_h2")
    main(["h2", "--h2", ss, "--ref-ld-chr", ref_ld, "--w-ld-chr", synth["w"], "--chr", "1-2",
          "--overlap-annot", "--overlap-cache", cache + ".npz", "--out", o2])
    r2 = fio.read_results(o2 + ".results")
    for col in ("Coefficient", "Coefficient_std_error", "Prop._h2", "Enrichment",
                "Enrichment_std_error", "Prop._SNPs"):
        # the cache stores the annotation matrix as float32 (continuous columns)
        assert r1[col].to_numpy() == pytest.approx(r2[col].to_numpy(), rel=1e-6)
    log = fio.parse_ldsc_log(o1 + ".log")
    assert log["tot_h2"] is not None and log["intercept"] is not None
    # Prop._SNPs of the base annotation is 1 by construction
    assert r1["Prop._SNPs"].iloc[0] == pytest.approx(1.0)


def test_scan_screen_fdr_validate(synth, set_ldscores, tmp_path):
    bdir = os.path.dirname(set_ldscores)
    traits = synth["traits"] + "," + synth["controls"]
    scan = str(tmp_path / "scan.csv")
    main(["scan", "--ref-ld-chr", synth["base"], "--w-ld-chr", synth["w"], "--chr", "1-2",
          "--traits", traits, "--sumstats-dir", synth["sumstats_dir"],
          "--bundle-dir", bdir, "--tags", "sets", "--snps-file", "snps_chr1_2.npy",
          "--cond", "set5", "--out", scan])
    d = pd.read_csv(scan)
    assert len(d) == 5 * 4                    # 5 candidates (set5 is conditioning) x 4 traits
    assert {"marg_z", "cond_z", "z_set5"} <= set(d.columns)
    # the simulated signal is on set0 for the 'trait' GWAS
    assert d[(d.set == "set0") & d.trait.str.startswith("trait")].marg_z.mean() > \
        d[(d.set == "set0") & d.trait.str.startswith("control")].marg_z.mean()

    scr = str(tmp_path / "screen.csv")
    main(["screen", "--ref-ld-chr", synth["base"], "--w-ld-chr", synth["w"], "--chr", "1-2",
          "--traits", traits, "--sumstats-dir", synth["sumstats_dir"],
          "--bundle-dir", bdir, "--tags", "sets", "--snps-file", "snps_chr1_2.npy",
          "--check", "3", "--out", scr])
    s = pd.read_csv(scr)
    assert len(s) == 6 * 4
    chk = pd.read_csv(scr + ".check.csv")
    assert chk.dz.abs().max() < 0.1
    m = d.merge(s, on=["set", "trait"])
    assert np.abs(m.marg_z - m.z).max() < 0.1

    fdr = str(tmp_path / "fdr.csv")
    main(["fdr", "--results", scr, "--control-traits", synth["controls"], "--z-col", "z",
          "--out", fdr])
    f = pd.read_csv(fdr)
    assert len(f) == 6 * 2 and (f.q_empirical <= 1).all() and (f.p_empirical > 0).all()

    # validate: use our own h2 output as if it were a banked ldsc run
    bank = tmp_path / "bank"
    bank.mkdir()
    ss = os.path.join(synth["sumstats_dir"], "trait0.sumstats.gz")
    main(["h2", "--h2", ss, "--ref-ld-chr", synth["base"] + "," + set_ldscores,
          "--w-ld-chr", synth["w"], "--chr", "1-2", "--out", str(bank / "sets__trait0")])
    with open(bank / "sets__trait0.log", "a") as fh:
        fh.write("Call:\n./ldsc.py \\\n--h2 %s \\\n--ref-ld-chr %s,%s \\\n"
                 % (ss, synth["base"], set_ldscores))
    main(["validate", "--ldsc-results", str(bank), "--ref-ld-chr", synth["base"],
          "--w-ld-chr", synth["w"], "--chr", "1-2", "--n", "0",
          "--out", str(tmp_path / "val.csv")])
    v = pd.read_csv(tmp_path / "val.csv")
    assert len(v) == 1 and v.max_abs_dz.iloc[0] < 1e-8

    main(["parse-results", "--h2-dir", str(bank), "--out", str(tmp_path / "parsed.csv")])
    p = pd.read_csv(tmp_path / "parsed.csv")
    assert len(p) == 12 and p.trait.iloc[0] == "trait0"


def test_simulate_command(tmp_path):
    main(["simulate", "--out", str(tmp_path / "s"), "--n-snp", "120", "--n-indiv", "30",
          "--chr", "1", "--n-traits", "1"])
    assert os.path.exists(tmp_path / "s" / "baseline" / "base.1.l2.ldscore.gz")
    assert os.path.exists(tmp_path / "s" / "sumstats" / "control0.sumstats.gz")
