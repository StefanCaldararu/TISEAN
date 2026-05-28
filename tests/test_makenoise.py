import subprocess
import numpy as np


def run_makenoise():
    result = subprocess.run(
        ["./bin/makenoise", "-I", "1", "-l", "1000", "-r", "1", "-0"],
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
        if len(parts) == 1:
            try:
                data.append([float(parts[0])])
            except ValueError:
                continue

    return np.array(data)


def test_makenoise_regression():
    out = run_makenoise()
    data = parse_output(out).flatten()

    ref = np.loadtxt("tests/refs/makenoise_I1l1000r1.txt").flatten()

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)