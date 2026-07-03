import subprocess
import numpy as np


def run_poincare():
    result = subprocess.run(
        ["./bin/poincare", "-m2", "-d1", "-c3", "-l1000", "./tests/refs/lorenz_l1000.txt"],
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


def test_poincare_regression():
    out = run_poincare()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/poincare_m2d1c3l1000.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
