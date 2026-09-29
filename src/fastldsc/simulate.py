"""Synthetic reference panel, annotations and summary statistics.

Used by the test-suite and by the README quickstart so that the whole
pipeline can be exercised without downloading any reference data.  The
genotypes carry local LD (a latent AR(1) process thresholded at random
allele frequencies), positions get 1 cM per Mb, and chi-square statistics
are simulated from the true LD-score model
``E[chisq] = 1 + N * sum_a tau_a * l_a``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import io as fio

_BED_CODE = np.array([0, 2, 3, 1], dtype=np.uint8)   # genotype 0,1,2,missing(3) -> 2-bit


def write_bed(prefix: str, G: np.ndarray, bim: pd.DataFrame, fam_ids: Sequence[str]) -> None:
    """Write plink binary files.  ``G`` is (n_indiv x m_snp) with 0/1/2 and
    -1 for missing (count of A2 alleles)."""
    n, m = G.shape
    codes = np.where(G < 0, 3, G).astype(np.int64)
    codes = _BED_CODE[codes]                            # (n, m)
    nbytes = (n + 3) // 4
    padded = np.zeros((m, nbytes * 4), dtype=np.uint8)
    padded[:, :n] = codes.T
    packed = (padded[:, 0::4] | (padded[:, 1::4] << 2)
              | (padded[:, 2::4] << 4) | (padded[:, 3::4] << 6)).astype(np.uint8)
    with open(prefix + ".bed", "wb") as fh:
        fh.write(b"\x6c\x1b\x01")
        fh.write(packed.tobytes())
    bim.to_csv(prefix + ".bim", sep="\t", header=False, index=False)
    with open(prefix + ".fam", "w") as fh:
        for i in fam_ids:
            fh.write("%s %s 0 0 0 -9\n" % (i, i))


def simulate_genotypes(n: int, m: int, rng: np.random.Generator,
                       rho: float = 0.9, missing: float = 0.005) -> np.ndarray:
    """(n x m) genotypes with AR(1)-style LD; a few entries missing (-1)."""
    thr = rng.uniform(0.05, 0.95, m)
    G = np.zeros((n, m), dtype=np.int8)
    for _ in range(2):                                  # two haplotypes
        z = rng.standard_normal((n, m))
        for j in range(1, m):
            z[:, j] = rho * z[:, j - 1] + np.sqrt(1 - rho ** 2) * z[:, j]
        u = 0.5 * (1 + _erf(z / np.sqrt(2)))           # uniform marginals
        G += (u < thr[None, :]).astype(np.int8)
    if missing > 0:
        G[rng.random((n, m)) < missing] = -1
    return G


def _erf(x):
    from scipy.special import erf
    return erf(x)


def simulate_panel(out_dir: str, n_indiv: int = 80, n_snp: int = 600,
                   chroms: Sequence[int] = (1, 2), n_genes: int = 40,
                   n_sets: int = 6, genes_per_set: int = 6, seed: int = 1,
                   n_traits: int = 3, n_gwas: float = 50_000.0,
                   window: int = 20_000) -> Dict[str, str]:
    """Write a complete synthetic dataset and return the paths.

    Layout (all under ``out_dir``)::

        plink/sim.<chr>.{bed,bim,fam,frq}    reference genotypes
        genes.coords                          GENE CHR START END
        sets/<name>.GeneSet                   gene sets (annotations to test)
        sumstats/<trait>.sumstats.gz          simulated GWAS
        print_snps.txt                        the "regression SNPs"
        control_traits.txt                    traits simulated with no signal
    """
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out_dir, "plink"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "sets"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "sumstats"), exist_ok=True)
    fam = ["IND%03d" % i for i in range(n_indiv)]
    coords_rows = []
    all_snps: List[str] = []
    for c in chroms:
        G = simulate_genotypes(n_indiv, n_snp, rng)
        bp = np.sort(rng.choice(np.arange(10_000, 5_000_000, 50), n_snp, replace=False))
        snps = ["rs%d_%d" % (c, j) for j in range(n_snp)]
        bim = pd.DataFrame({"CHR": c, "SNP": snps, "CM": bp / 1e6, "BP": bp,
                            "A1": "A", "A2": "G"})
        prefix = os.path.join(out_dir, "plink", "sim.%d" % c)
        write_bed(prefix, G, bim, fam)
        Gf = np.where(G < 0, np.nan, G).astype(float)
        f = np.nanmean(Gf, axis=0) / 2
        maf = np.minimum(f, 1 - f)
        nchrobs = 2 * np.isfinite(Gf).sum(axis=0)
        with open(prefix + ".frq", "w") as fh:
            fh.write(" CHR           SNP   A1   A2          MAF  NCHROBS\n")
            for j in range(n_snp):
                fh.write("%4d %13s %4s %4s %12.5f %8d\n"
                         % (c, snps[j], "A", "G", maf[j], nchrobs[j]))
        all_snps.extend(snps)
        for g in range(n_genes):
            s = int(rng.integers(bp.min(), bp.max() - 60_000))
            coords_rows.append(("GENE%d_%d" % (c, g), str(c), s, s + int(rng.integers(2_000, 50_000))))
    coords = pd.DataFrame(coords_rows, columns=["GENE", "CHR", "START", "END"])
    fio.write_gene_coords(os.path.join(out_dir, "genes.coords"), coords)
    genes = coords["GENE"].to_numpy()
    sets = {}
    for k in range(n_sets):
        sets["set%d" % k] = rng.choice(genes, genes_per_set, replace=False).tolist()
    for name, members in sets.items():
        with open(os.path.join(out_dir, "sets", name + ".GeneSet"), "w") as fh:
            fh.write("\n".join(members) + "\n")
    # regression SNPs: a random 70% subset (mimics the HapMap3 restriction)
    print_snps = np.array(all_snps)[rng.random(len(all_snps)) < 0.7]
    with open(os.path.join(out_dir, "print_snps.txt"), "w") as fh:
        fh.write("\n".join(print_snps) + "\n")
    paths = dict(bfile=os.path.join(out_dir, "plink", "sim."),
                 frq=os.path.join(out_dir, "plink", "sim."),
                 coords=os.path.join(out_dir, "genes.coords"),
                 sets_dir=os.path.join(out_dir, "sets"),
                 print_snps=os.path.join(out_dir, "print_snps.txt"),
                 sumstats_dir=os.path.join(out_dir, "sumstats"),
                 out_dir=out_dir)
    paths["sets"] = sets
    paths["window"] = str(window)
    # traits are simulated once the LD scores exist (see simulate_traits)
    paths["_rng_state"] = rng
    paths["n_traits"] = n_traits
    paths["n_gwas"] = n_gwas
    return paths


def simulate_traits(paths: Dict, base_L: pd.DataFrame, set_L: Optional[pd.DataFrame],
                    tau_base: float = 2e-6, tau_set: float = 3e-5,
                    signal_set: str = "set0", seed: int = 7) -> List[str]:
    """Simulate GWAS chi-squares from LD scores and write ``.sumstats.gz``.

    ``base_L`` / ``set_L`` are LD score frames (SNP + columns) as read by
    :func:`fastldsc.io.read_ldscore_chr`.  Traits ``trait0..`` carry
    ``tau_set`` on ``signal_set``; ``control0..`` carry no annotation signal.
    Returns the list of control trait names.
    """
    rng = np.random.default_rng(seed)
    n_traits = int(paths["n_traits"])
    N = float(paths["n_gwas"])
    snp = base_L["SNP"].to_numpy()
    l_base = base_L.iloc[:, 1].to_numpy(float)
    l_set = set_L.set_index("SNP").reindex(snp)[signal_set].to_numpy(float) \
        if set_L is not None else np.zeros(snp.size)
    l_set = np.nan_to_num(l_set)
    controls = []
    for k in range(n_traits):
        for kind in ("trait", "control"):
            mu = 1 + N * (tau_base * l_base + (tau_set * l_set if kind == "trait" else 0))
            # chisq with the right mean: scale a chi2(1) draw by mu
            z = rng.standard_normal(snp.size) * np.sqrt(mu)
            name = "%s%d" % (kind, k)
            fio.write_sumstats(os.path.join(paths["sumstats_dir"], name + ".sumstats.gz"),
                               snp, z, np.full(snp.size, N), A1="A", A2="G")
            if kind == "control":
                controls.append(name)
    with open(os.path.join(paths["out_dir"], "control_traits.txt"), "w") as fh:
        fh.write("\n".join(controls) + "\n")
    return controls


def simulate_all(out_dir: str, seed: int = 1, **kw) -> Dict[str, str]:
    """Synthetic panel + baseline LD scores + weights + GWAS in one call.

    Writes, in addition to :func:`simulate_panel`::

        baseline/base.<chr>.l2.ldscore.gz (+ .l2.M, .l2.M_5_50)   base + 5 random annotations
        baseline/base.<chr>.annot.gz                              their annotation files
        weights/w.<chr>.l2.ldscore.gz                             LD among regression SNPs only
    """
    from . import annot as fannot
    from . import l2 as fl2

    paths = simulate_panel(out_dir, seed=seed, **kw)
    rng = np.random.default_rng(seed + 100)
    chroms = sorted({int(c) for c in fio.read_gene_coords(paths["coords"])["CHR"]})
    os.makedirs(os.path.join(out_dir, "baseline"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "weights"), exist_ok=True)
    base_prefix = os.path.join(out_dir, "baseline", "base.")
    w_prefix = os.path.join(out_dir, "weights", "w.")
    print_snps = set(fio.read_snp_list(paths["print_snps"]).tolist())
    coords = fio.read_gene_coords(paths["coords"])
    base_frames, set_frames = [], []
    for c in chroms:
        bfile = fio.sub_chr(paths["bfile"], c)
        bim = fio.read_bim(bfile + ".bim")
        m = len(bim)
        # a small "baseline": all SNPs, four random binary annotations of
        # decreasing density and one continuous one (like baselineLD's
        # MAF/LD-related columns)
        A = np.column_stack([np.ones(m)] + [(rng.random(m) < d).astype(float)
                                            for d in (0.5, 0.3, 0.2, 0.1)]
                            + [rng.gamma(2.0, 1.0, m)])
        base_names = ["base", "rand50", "rand30", "rand20", "rand10", "cont"]
        fio.write_annot(fio.sub_chr(base_prefix, c) + ".annot.gz",
                        pd.DataFrame(A).round(4).to_numpy(), base_names, bim)
        ids, L, M, M5, _ = fl2.ld_scores_chromosome(
            bfile, c, A, base_names, print_snps, out_prefix=base_prefix, verbose=False)
        base_frames.append(pd.DataFrame({"SNP": ids, "baseL2": L[:, 0]}))
        # regression-weight LD: sum of r^2 over regression SNPs only
        w_annot = np.isin(bim["SNP"].to_numpy(), list(print_snps)).astype(float)[:, None]
        fl2.ld_scores_chromosome(bfile, c, w_annot, ["L2"], print_snps,
                                 out_prefix=w_prefix, verbose=False)
        S = fannot.gene_set_annotation(coords, paths["sets"], bim["BP"].to_numpy(), c,
                                       int(paths["window"]))
        ids2, L2, _, _, _ = fl2.ld_scores_chromosome(bfile, c, S, list(paths["sets"]),
                                                     print_snps, verbose=False)
        set_frames.append(pd.DataFrame(L2, columns=list(paths["sets"])).assign(SNP=ids2))
    base_L = pd.concat(base_frames, ignore_index=True)
    set_L = pd.concat(set_frames, ignore_index=True)
    controls = simulate_traits(paths, base_L, set_L, seed=seed + 200)
    paths.update(base=base_prefix, w=w_prefix, controls=",".join(controls),
                 traits=",".join("trait%d" % k for k in range(int(paths["n_traits"]))))
    paths.pop("_rng_state", None)
    return paths
