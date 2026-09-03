import subprocess
import numpy as np

INPUT = "./tests/refs/cluster_input.txt"
NCL = 3


def input_size():
    """np as cluster.f computes it: the largest point index in the
    dissimilarity matrix file."""
    npoints = 0
    with open(INPUT) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            i, j = int(parts[0]), int(parts[1])
            npoints = max(npoints, i, j)
    return npoints


def run_cluster(tmp_path, xfile=None):
    outfile = tmp_path / "cluster_out.txt"
    cmd = ["./bin/cluster", "-#%d" % NCL]
    if xfile is not None:
        cmd += ["-X", str(xfile)]
    cmd += ["-o", str(outfile), INPUT]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return outfile.read_text()


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
    xfile = tmp_path / "fixed.txt"
    xfile.write_text("3 2\n")

    data = parse_output(run_cluster(tmp_path, xfile=xfile))

    assert int(data[2, 0]) == 2


def test_cluster_is_deterministic_given_same_flags(tmp_path):
    out1 = run_cluster(tmp_path)
    out2 = run_cluster(tmp_path)

    assert out1 == out2
