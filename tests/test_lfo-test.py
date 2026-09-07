import subprocess
import numpy as np


def run_lfo_test():
    result = subprocess.run(
        ["./bin/lfo-test", "-m1,2", "-d1", "-n50", "-k10", "./tests/refs/ar-run_l1000.txt"],
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
                data.append(float(parts[0]))
            except ValueError:
                continue

    return np.array(data)


def test_lfo_test_regression():
    out = run_lfo_test()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/lfo-test_m1_2d1n50k10.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
