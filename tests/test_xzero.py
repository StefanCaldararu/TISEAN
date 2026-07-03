import subprocess
import numpy as np


def run_xzero():
    result = subprocess.run(
        ["./bin/xzero", "-c1,2", "-m3", "-d1", "-n50", "-k10", "-s5",
         "./tests/refs/henon_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers)
        if len(parts) == 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue

    return np.array(data)


def test_xzero_regression():
    out = run_xzero()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/xzero_c12m3d1n50k10s5.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
