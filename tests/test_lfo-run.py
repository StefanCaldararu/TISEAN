import subprocess
import numpy as np


def run_lfo_run():
    result = subprocess.run(
        ["./bin/lfo-run", "-m1,2", "-d1", "-L50", "-k10", "./tests/refs/ar-run_l1000.txt"],
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


def test_lfo_run_regression():
    out = run_lfo_run()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/lfo-run_m1_2d1L50k10.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
