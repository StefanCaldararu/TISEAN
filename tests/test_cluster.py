import subprocess
import numpy as np


def run_cluster():
    subprocess.run(
        ["./bin/cluster", "-#3", "-o", "./tests/refs/cluster_out.txt", "./tests/refs/cluster_input.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    with open("./tests/refs/cluster_out.txt") as f:
        return f.read()


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers)
        if len(parts) == 4:
            try:
                data.append([float(p) for p in parts])
            except ValueError:
                continue

    return np.array(data)


def test_cluster_regression():
    out = run_cluster()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/cluster_ref.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
