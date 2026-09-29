"""File readers and writers for the ldsc file formats.

Everything here is a thin layer over pandas/numpy.  The formats are those
documented by ldsc (``docs/file_formats_ld.txt`` and
``docs/file_formats_sumstats.txt`` in the ldsc repository):

* ``<prefix><chr>.l2.ldscore.gz``   tab-separated, CHR SNP BP <one column per annotation>
* ``<prefix><chr>.l2.M`` / ``.l2.M_5_50``  one line of per-annotation SNP counts
* ``<prefix><chr>.annot.gz``        CHR BP SNP CM <annotations>  (or "thin": annotations only)
* ``<prefix><chr>.frq``             plink --freq output (CHR SNP A1 A2 MAF NCHROBS)
* ``<trait>.sumstats.gz``           SNP A1 A2 N Z (munge_sumstats.py output)
* ``.results``                      ldsc --h2 --overlap-annot output table
"""

from __future__ import annotations

import gzip
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

AUTOSOMES = tuple(range(1, 23))


# ---------------------------------------------------------------- paths

def sub_chr(prefix: str, chrom) -> str:
    """Insert a chromosome number into a per-chromosome file prefix.

    Mirrors ldsc's convention: if the prefix contains ``@`` it is replaced,
    otherwise the chromosome is appended (``baselineLD.`` -> ``baselineLD.22``).
    """
    if "@" in prefix:
        return prefix.replace("@", str(chrom))
    return prefix + str(chrom)


def find_compressed(path: str) -> str:
    """Return ``path``, ``path.gz`` or ``path.bz2`` -- whichever exists."""
    for suffix in ("", ".gz", ".bz2"):
        if os.path.exists(path + suffix):
            return path + suffix
    raise FileNotFoundError("could not open %s[.gz/.bz2]" % path)


def present_chroms(prefix: str, suffix: str = ".l2.ldscore",
                   chroms: Iterable[int] = AUTOSOMES) -> List[int]:
    """Chromosomes for which ``<prefix><chr><suffix>[.gz]`` exists."""
    out = []
    for c in chroms:
        try:
            find_compressed(sub_chr(prefix, c) + suffix)
            out.append(c)
        except FileNotFoundError:
            pass
    return out


# ---------------------------------------------------------------- LD scores

def read_ldscore_chr(prefix: str, chroms: Iterable[int] = AUTOSOMES,
                     sort: bool = False) -> pd.DataFrame:
    """Concatenate ``<prefix><chr>.l2.ldscore[.gz]`` over chromosomes.

    Returns a frame with a ``SNP`` column followed by one column per
    annotation (CHR/BP are dropped; legacy MAF/CM columns are dropped).  Rows
    are kept in file order (chromosomes in the given order), which is the
    order the block jackknife is defined over.  With ``sort=True`` the rows
    are ordered by (CHR, BP) first, as ldsc does.
    """
    dfs = []
    for c in chroms:
        path = find_compressed(sub_chr(prefix, c) + ".l2.ldscore")
        dfs.append(pd.read_csv(path, sep="\t"))
    df = pd.concat(dfs, ignore_index=True)
    if sort and "CHR" in df.columns and "BP" in df.columns:
        df = df.sort_values(["CHR", "BP"], kind="stable").reset_index(drop=True)
    drop = [c for c in ("CHR", "BP", "MAF", "CM") if c in df.columns]
    return df.drop(columns=drop)


def read_M(prefix: str, n_annot: int, chroms: Iterable[int] = AUTOSOMES,
           common: bool = True) -> np.ndarray:
    """Sum ``<prefix><chr>.l2.M_5_50`` (or ``.l2.M``) over chromosomes."""
    suffix = ".l2.M_5_50" if common else ".l2.M"
    tot = np.zeros(n_annot)
    for c in chroms:
        tot += np.loadtxt(sub_chr(prefix, c) + suffix).reshape(n_annot)
    return tot


def write_ldscore(prefix: str, chrom: int, snp_df: pd.DataFrame,
                  L: np.ndarray, colnames: Sequence[str],
                  M: np.ndarray, M_5_50: np.ndarray,
                  compress: bool = True) -> str:
    """Write one chromosome of LD scores in ldsc's format.

    ``snp_df`` must have CHR, SNP, BP for the rows of ``L`` (rows x annots).
    Values are written with three decimals, as ldsc does.
    """
    out = sub_chr(prefix, chrom)
    df = pd.DataFrame(L, columns=list(colnames))
    df.insert(0, "BP", snp_df["BP"].to_numpy())
    df.insert(0, "SNP", snp_df["SNP"].to_numpy())
    df.insert(0, "CHR", snp_df["CHR"].to_numpy())
    path = out + ".l2.ldscore" + (".gz" if compress else "")
    df.to_csv(path, sep="\t", index=False, float_format="%.3f",
              compression="gzip" if compress else None)
    with open(out + ".l2.M", "w") as fh:
        fh.write("\t".join("%d" % round(x) for x in np.asarray(M).ravel()) + "\n")
    with open(out + ".l2.M_5_50", "w") as fh:
        fh.write("\t".join("%d" % round(x) for x in np.asarray(M_5_50).ravel()) + "\n")
    return path


# ---------------------------------------------------------------- annotations

ANNOT_META = ("CHR", "BP", "SNP", "CM")


def read_annot(path: str) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
    """Read a ``.annot[.gz]`` file.

    Returns ``(meta, annots)`` where ``meta`` holds CHR/BP/SNP/CM (``None``
    for a thin annot file) and ``annots`` the annotation columns as floats.
    """
    df = pd.read_csv(path, sep="\t")
    meta_cols = [c for c in ANNOT_META if c in df.columns]
    meta = df[meta_cols] if meta_cols else None
    annots = df.drop(columns=meta_cols).astype(float)
    return meta, annots


def write_annot(path: str, A: np.ndarray, names: Sequence[str],
                bim: Optional[pd.DataFrame] = None) -> None:
    """Write an annotation matrix as ``.annot[.gz]``.

    With ``bim`` (CHR, SNP, CM, BP columns) the full format is written,
    otherwise the thin format (annotation columns only; ldsc ``--thin-annot``).
    """
    df = pd.DataFrame(np.asarray(A), columns=list(names))
    if bim is not None:
        df.insert(0, "CM", bim["CM"].to_numpy())
        df.insert(0, "SNP", bim["SNP"].to_numpy())
        df.insert(0, "BP", bim["BP"].to_numpy())
        df.insert(0, "CHR", bim["CHR"].to_numpy())
    comp = "gzip" if path.endswith(".gz") else None
    df.to_csv(path, sep="\t", index=False, compression=comp)


def read_frq(path: str) -> pd.DataFrame:
    """Read a plink ``.frq`` file (whitespace separated).  MAF -> FRQ."""
    df = pd.read_csv(path, sep=r"\s+")
    if "MAF" in df.columns:
        df = df.rename(columns={"MAF": "FRQ"})
    return df[["SNP", "FRQ"]]


def common_snp_mask(frq: pd.DataFrame) -> np.ndarray:
    """ldsc's ``--overlap-annot`` SNP filter: 0.05 < FRQ < 0.95."""
    f = frq["FRQ"].to_numpy(float)
    return (f > 0.05) & (f < 0.95)


def read_overlap_matrix(prefixes: Sequence[str], frq_prefix: Optional[str],
                        chroms: Iterable[int] = AUTOSOMES,
                        keep_annot: bool = False):
    """Overlap matrix of the annotations in a list of ``.annot`` prefixes.

    Mirrors ldsc's ``parse.annot``: the annot files of every prefix are
    concatenated sideways per chromosome, restricted to SNPs with
    0.05 < FRQ < 0.95 when ``frq_prefix`` is given, and ``A.T @ A`` is
    accumulated.  Returns ``(overlap, M_tot, names)``; with ``keep_annot``
    also the concatenated (filtered) annotation matrix as float32 so callers
    can compute overlap rows for further annotations without re-reading.
    """
    overlap = None
    M_tot = 0
    names: List[str] = []
    kept = []
    for c in chroms:
        blocks = []
        for p in prefixes:
            _, a = read_annot(find_compressed(sub_chr(p, c) + ".annot"))
            blocks.append(a)
        if not names:
            names = [n for a in blocks for n in a.columns]
        A = np.hstack([b.to_numpy(np.float64) for b in blocks])
        if frq_prefix is not None:
            frq = read_frq(find_compressed(sub_chr(frq_prefix, c) + ".frq"))
            if len(frq) != A.shape[0]:
                raise ValueError("frq/annot row mismatch on chr%s" % c)
            A = A[common_snp_mask(frq)]
        M_tot += A.shape[0]
        overlap = A.T @ A if overlap is None else overlap + A.T @ A
        if keep_annot:
            kept.append(A.astype(np.float32))
    if keep_annot:
        return overlap, M_tot, names, np.vstack(kept)
    return overlap, M_tot, names


# ---------------------------------------------------------------- sumstats

def read_sumstats(path: str) -> pd.DataFrame:
    """Read a munged ``.sumstats[.gz]`` file: SNP, Z, N with NAs dropped."""
    ss = pd.read_csv(path, sep="\t", usecols=["SNP", "Z", "N"],
                     dtype={"SNP": str, "Z": float, "N": float})
    return ss.dropna(how="any")


def write_sumstats(path: str, snp, Z, N, A1=None, A2=None) -> None:
    df = pd.DataFrame({"SNP": snp})
    if A1 is not None:
        df["A1"] = A1
        df["A2"] = A2
    df["N"] = N
    df["Z"] = Z
    comp = "gzip" if path.endswith(".gz") else None
    df.to_csv(path, sep="\t", index=False, compression=comp)


# ---------------------------------------------------------------- plink text

def read_bim(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
                     dtype={"CHR": str, "SNP": str, "A1": str, "A2": str})
    return df


def read_fam(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1],
                       names=["FID", "IID"], dtype=str)


def read_snp_list(path: str) -> np.ndarray:
    """One-column SNP list (e.g. the HapMap3 ``--print-snps`` list)."""
    df = pd.read_csv(path, header=None, sep=r"\s+")
    if df.shape[1] > 1:
        # a header line such as 'SNP' or a w_hm3.snplist with alleles
        if str(df.iloc[0, 0]).upper() == "SNP":
            df = df.iloc[1:]
        return df.iloc[:, 0].astype(str).to_numpy()
    return df.iloc[:, 0].astype(str).to_numpy()


# ---------------------------------------------------------------- gene sets

def read_gene_coords(path: str) -> pd.DataFrame:
    """GENE CHR START END table (tab separated, header)."""
    df = pd.read_csv(path, sep="\t", dtype={"GENE": str, "CHR": str})
    return df


def write_gene_coords(path: str, coords: pd.DataFrame) -> None:
    coords.to_csv(path, sep="\t", index=False)


def read_geneset(path: str) -> List[str]:
    with open(path) as fh:
        return [ln.strip().split()[0] for ln in fh if ln.strip()]


def read_gene_sets(paths: Sequence[str]) -> Dict[str, List[str]]:
    """``name -> members`` from a list of ``.GeneSet`` files (name = stem)."""
    out = {}
    for p in paths:
        name = os.path.basename(p)
        for suf in (".GeneSet", ".geneset", ".txt"):
            if name.endswith(suf):
                name = name[:-len(suf)]
                break
        out[name] = read_geneset(p)
    return out


def read_sets_npz(path: str) -> Dict[str, List[str]]:
    """``npz`` with ``names`` (S,) and ``members`` (object array of lists)."""
    z = np.load(path, allow_pickle=True)
    names = z["names"].astype(str)
    return {n: [str(g) for g in m] for n, m in zip(names, z["members"])}


def write_sets_npz(path: str, sets: Dict[str, Sequence[str]]) -> None:
    np.savez_compressed(path, names=np.array(list(sets), dtype=object),
                        members=np.array([list(v) for v in sets.values()],
                                         dtype=object))


# ---------------------------------------------------------------- l2 bundles

def bundle_paths(out_dir: str, tag: str) -> Dict[str, str]:
    return dict(l2=os.path.join(out_dir, "l2_%s.npy" % tag),
                M=os.path.join(out_dir, "M_%s.npy" % tag),
                names=os.path.join(out_dir, "names_%s.npy" % tag),
                overlap=os.path.join(out_dir, "overlap_%s.npy" % tag))


def read_bundle(out_dir: str, tag: str, snps_file: str = "snps_all.npy",
                mmap: bool = True):
    """Read a one-pass LD score bundle written by ``fastldsc l2 --bundle``.

    Returns ``(names, M_5_50, L, snps)`` with ``L`` (n_print_snps x S,
    float32) and ``snps`` the row order.  ``overlap_<tag>.npy`` (S x
    n_baseline) is returned as a fifth element when present, else ``None``.
    """
    p = bundle_paths(out_dir, tag)
    names = np.load(p["names"], allow_pickle=True).astype(str)
    M = np.load(p["M"])
    L = np.load(p["l2"], mmap_mode="r" if mmap else None)
    sp = os.path.join(out_dir, snps_file)
    if not os.path.exists(sp):
        cands = sorted(f for f in os.listdir(out_dir)
                       if f.startswith("snps_") and f.endswith(".npy"))
        if len(cands) == 1:
            sp = os.path.join(out_dir, cands[0])
        else:
            raise FileNotFoundError("%s not found in %s (candidates: %s)"
                                    % (snps_file, out_dir, ", ".join(cands) or "none"))
    snps = np.load(sp, allow_pickle=True).astype(str)
    overlap = np.load(p["overlap"]) if os.path.exists(p["overlap"]) else None
    return names, M, L, snps, overlap


# ---------------------------------------------------------------- ldsc output

def read_results(path: str) -> pd.DataFrame:
    """Read an ldsc ``.results`` table."""
    return pd.read_csv(path, sep="\t")


_LOG_FLOAT = r"([-+]?[\d.]+(?:[eE][-+]?\d+)?)"


def parse_ldsc_log(path: str) -> dict:
    """Extract the call arguments and headline numbers from an ldsc h2 log."""
    out: dict = {"ref_ld_chr": None, "w_ld_chr": None, "h2": None,
                 "frqfile_chr": None, "n_snps": None,
                 "tot_h2": None, "tot_h2_se": None,
                 "intercept": None, "intercept_se": None,
                 "mean_chisq": None, "lambda_gc": None, "ratio": None}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for ln in fh:
            s = ln.strip().rstrip("\\").strip()
            for key, flag in (("ref_ld_chr", "--ref-ld-chr "),
                              ("w_ld_chr", "--w-ld-chr "),
                              ("h2", "--h2 "), ("frqfile_chr", "--frqfile-chr ")):
                if s.startswith(flag):
                    out[key] = s[len(flag):].strip()
            m = re.match(r"Removed \d+ SNPs with chi\^2 > [\d.]+ \((\d+) SNPs remain\)", s)
            if m:
                out["n_snps"] = int(m.group(1))
            m = re.match(r"Total Observed scale h2: " + _LOG_FLOAT + r" \(" + _LOG_FLOAT + r"\)", s)
            if m:
                out["tot_h2"], out["tot_h2_se"] = float(m.group(1)), float(m.group(2))
            m = re.match(r"Intercept: " + _LOG_FLOAT + r" \(" + _LOG_FLOAT + r"\)", s)
            if m:
                out["intercept"], out["intercept_se"] = float(m.group(1)), float(m.group(2))
            m = re.match(r"Mean Chi\^2: " + _LOG_FLOAT, s)
            if m:
                out["mean_chisq"] = float(m.group(1))
            m = re.match(r"Lambda GC: " + _LOG_FLOAT, s)
            if m:
                out["lambda_gc"] = float(m.group(1))
    if out["ref_ld_chr"]:
        out["ref_ld_chr"] = [x for x in out["ref_ld_chr"].split(",") if x]
    return out


def gz_open(path: str, mode: str = "rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)
