import subprocess
import numpy as np


def run_fsle():
    result = subprocess.run(
        ["./bin/fsle", "-m2", "-d1", "-l500", "./tests/refs/lorenz_l1000.txt"],
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
        if len(parts) == 3:
            try:
                data.append([float(p) for p in parts])
            except ValueError:
                continue

    return np.array(data)


def test_fsle_regression():
    out = run_fsle()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/fsle_m2l500.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
