import subprocess
import numpy as np


def run_corr():
    result = subprocess.run(
        ["./bin/corr", "./tests/refs/ar-run_l1000.txt"],
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


def test_corr_regression():
    out = run_corr()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/corr.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)