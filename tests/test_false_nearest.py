import subprocess
import numpy as np


def run_false_nearest():
    result = subprocess.run(
        ["./bin/false_nearest", "-m1", "-M1,5", "-d1", "-l500", "./tests/refs/ar-run_l1000.txt"],
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
        if len(parts) == 4:
            try:
                data.append([float(p) for p in parts])
            except ValueError:
                continue

    return np.array(data)


def test_false_nearest_regression():
    out = run_false_nearest()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/false_nearest_m1M15l500.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
