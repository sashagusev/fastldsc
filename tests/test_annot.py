"""Gene-set -> annotation builders."""

import gzip

import numpy as np
import pandas as pd
import pytest

from fastldsc import annot as fa
from fastldsc import io as fio


def test_gene_windows_and_mask():
    coords = pd.DataFrame({"GENE": ["g1", "g2", "g3"], "CHR": ["1", "1", "2"],
                           "START": [1000, 5000, 100], "END": [1500, 5200, 200]})
    bp = np.array([900, 1000, 1400, 1600, 4950, 5300, 9000])
    w = fa.gene_windows(coords, ["g1", "g2", "g3"], "1", window=100)
    assert w.tolist() == [[900, 1600], [4900, 5300]]
    mask = fa.windows_to_mask(bp, w)
    assert mask.tolist() == [True, True, True, True, True, True, False]
    A = fa.gene_set_annotation(coords, {"a": ["g1"], "b": ["g3"], "c": ["g2", "g1"]},
                               bp, "1", window=0)
    assert A[:, 0].tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert A[:, 1].sum() == 0                      # g3 is on chr 2
    assert A[:, 2].tolist() == [0, 1, 1, 0, 0, 0, 0]


def test_unsorted_positions():
    coords = pd.DataFrame({"GENE": ["g"], "CHR": ["3"], "START": [50], "END": [60]})
    bp = np.array([70, 40, 55, 10])
    A = fa.gene_set_annotation(coords, {"s": ["g"]}, bp, 3, window=5)
    assert A[:, 0].tolist() == [0, 0, 1, 0]


def test_gtf_parsing(tmp_path):
    gtf = tmp_path / "t.gtf.gz"
    lines = [
        "#comment",
        'chr1\tHAVANA\tgene\t100\t900\t.\t+\t.\tgene_id "ENSG1.4"; gene_name "A";',
        'chr1\tHAVANA\ttranscript\t100\t900\t.\t+\t.\tgene_id "ENSG1.4"; gene_name "A";',
        'chr6\tHAVANA\tgene\t30000000\t30001000\t.\t-\t.\tgene_id "ENSG2.1"; gene_name "HLA";',
        'chrX\tHAVANA\tgene\t5\t10\t.\t-\t.\tgene_id "ENSG3.1"; gene_name "X";',
    ]
    with gzip.open(gtf, "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    df = fa.gene_coords_from_gtf(str(gtf))
    assert df["GENE"].tolist() == ["ENSG1", "ENSG2"]
    assert df["CHR"].tolist() == ["1", "6"]
    assert df["SYMBOL"].tolist() == ["A", "HLA"]
    assert fa.mhc_genes(df) == {"ENSG2"}
    assert fa.mappable_genes(df, ["ENSG1", "ENSG2", "nope"]).tolist() == [True, False, False]


def test_bh_mask():
    p = np.array([0.001, 0.02, 0.5, np.nan, 0.03, 0.9])
    keep = fa.bh_mask(p, 0.10)
    # BH at 10% over 5 finite p: sorted 0.001,0.02,0.03,0.5,0.9 vs 0.02,0.04,0.06,0.08,0.10
    assert keep.tolist() == [True, True, False, False, True, False]
    assert not fa.bh_mask(np.array([0.5, 0.6])).any()
    assert not fa.bh_mask(np.array([np.nan])).any()


def test_indegree_and_top_sets():
    score = np.array([[5, 1, np.nan], [4, np.nan, -np.inf], [3, 2, 0]], float)
    deg = fa.indegree_from_pairs(score, 3, topn=4)     # top 4: 5,4,3,2 -> genes 0,0,0,1
    assert deg.tolist() == [3, 1, 0]
    sets = fa.top_genes_by_indegree(score, ["a", "b", "c"], [1, 2], topn=4, prefix="t")
    assert sets == {"t_top1": ["a"], "t_top2": ["a", "b"]}


def test_matched_control():
    score = np.array([10, 9, 8, 7, 1, 2, 3, 0.5], float)
    chosen = fa.matched_control([0, 1], score, np.ones(8, bool))
    assert sorted(chosen) == [2, 3]


def test_annot_write_read(tmp_path):
    A = np.array([[1, 0], [1, 1], [0, 0]])
    bim = pd.DataFrame({"CHR": [1, 1, 1], "SNP": ["a", "b", "c"], "CM": [0.1, 0.2, 0.3],
                        "BP": [10, 20, 30], "A1": "A", "A2": "G"})
    p = str(tmp_path / "x.annot.gz")
    fio.write_annot(p, A, ["s1", "s2"], bim)
    meta, ann = fio.read_annot(p)
    assert meta["SNP"].tolist() == ["a", "b", "c"]
    assert ann.to_numpy().tolist() == A.tolist()
    q = str(tmp_path / "thin.annot.gz")
    fio.write_annot(q, A, ["s1", "s2"])
    meta, ann = fio.read_annot(q)
    assert meta is None and list(ann.columns) == ["s1", "s2"]


def test_sets_npz_roundtrip(tmp_path):
    p = str(tmp_path / "s.npz")
    fio.write_sets_npz(p, {"a": ["g1", "g2"], "b": ["g3"]})
    assert fio.read_sets_npz(p) == {"a": ["g1", "g2"], "b": ["g3"]}
    assert fa.load_sets([p]) == {"a": ["g1", "g2"], "b": ["g3"]}
