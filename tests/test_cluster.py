import shutil
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CLUSTER_BIN = REPO_ROOT / "bin" / "cluster"
CLUSTER_INPUT = REPO_ROOT / "tests" / "refs" / "cluster_input.txt"
NCL = 3


def input_size():
    """np as cluster.f computes it: the largest point index in the
    dissimilarity matrix file."""
    npoints = 0
    with open(CLUSTER_INPUT) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            i, j = int(parts[0]), int(parts[1])
            npoints = max(npoints, i, j)
    return npoints


def run_cluster(tmp_path, xfix=None):
    # cluster.f declares file/fout as character*72: run with short, relative
    # filenames in a scratch cwd so long tmp_path/repo paths never get
    # silently truncated by the Fortran runtime.
    shutil.copy(CLUSTER_INPUT, tmp_path / "input.txt")

    cmd = [str(CLUSTER_BIN), "-#%d" % NCL]
    if xfix is not None:
        (tmp_path / "fixed.txt").write_text(xfix)
        cmd += ["-X", "fixed.txt"]
    cmd += ["-o", "out.txt", "input.txt"]

    subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=tmp_path)
    return (tmp_path / "out.txt").read_text()


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == NCL + 1:
            try:
                data.append([float(p) for p in parts])
            except ValueError:
                continue
    return np.array(data)


def test_cluster_output_is_a_valid_partition(tmp_path):
    data = parse_output(run_cluster(tmp_path))
    npoints = input_size()

    # one row per input point, and every label is a cluster in [1, NCL]
    assert data.shape[0] == npoints
    labels = data[:, 0].astype(int)
    assert set(labels).issubset(set(range(1, NCL + 1)))
    assert len(set(labels)) == NCL


def test_cluster_respects_fixed_assignment(tmp_path):
    data = parse_output(run_cluster(tmp_path, xfix="3 2\n"))

    assert int(data[2, 0]) == 2


def test_cluster_is_deterministic_given_same_flags(tmp_path):
    out1 = run_cluster(tmp_path)
    out2 = run_cluster(tmp_path)

    assert out1 == out2
