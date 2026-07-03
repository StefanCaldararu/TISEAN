import subprocess
import numpy as np


def run_autocor():
    result = subprocess.run(
        ["./bin/autocor", "-l", "100", "./tests/refs/ar-run_l1000.txt"],
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


def test_autocor_regression():
    out = run_autocor()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/autocor_l100.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
