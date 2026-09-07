import subprocess
import numpy as np


def run_nstat_z():
    result = subprocess.run(
        ["./bin/nstat_z", "-#4", "-m2", "-n20", "-k10", "./tests/refs/ar-run_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers/blank separator lines)
        if len(parts) == 3:
            try:
                data.append([float(p) for p in parts])
            except ValueError:
                continue

    return np.array(data)


def test_nstat_z_regression():
    out = run_nstat_z()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/nstat_z_h4m2n20k10.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
