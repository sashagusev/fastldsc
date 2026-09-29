import os

import numpy as np
import pytest

from fastldsc import io as fio
from fastldsc.simulate import simulate_all
from fastldsc.sldsc import RefData, TraitData


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    """A complete synthetic dataset (two chromosomes, six gene sets, GWAS)."""
    out = str(tmp_path_factory.mktemp("synth"))
    paths = simulate_all(out, seed=1, n_indiv=80, n_snp=600, chroms=(1, 2), n_traits=2)
    paths["chroms"] = [1, 2]
    return paths


@pytest.fixture(scope="session")
def set_ldscores(synth):
    """LD scores of the gene sets in ldsc's per-chromosome format, plus a bundle."""
    from fastldsc.cli import main
    out = os.path.join(synth["out_dir"], "sets_l2")
    os.makedirs(out, exist_ok=True)
    main(["l2", "--bfile", synth["bfile"], "--sets", synth["sets_dir"],
          "--coords", synth["coords"], "--window", synth["window"],
          "--print-snps", synth["print_snps"], "--chr", "1-2",
          "--out", os.path.join(out, "sets."), "--bundle-dir", out, "--tag", "sets",
          "--overlap-with", synth["base"]])
    return os.path.join(out, "sets.")


@pytest.fixture(scope="session")
def ref(synth, set_ldscores):
    r = RefData(synth["base"], synth["w"], {"sets": set_ldscores}, chroms=[1, 2],
                verbose=False)
    return r


@pytest.fixture(scope="session")
def trait(synth, ref):
    return TraitData(ref, os.path.join(synth["sumstats_dir"], "trait0.sumstats.gz"))


def design(ref, keys):
    cols = ref.base_columns() + ref.columns(keys)
    M = np.concatenate([ref.Mbase, ref.M_for(keys)])
    return cols, M
