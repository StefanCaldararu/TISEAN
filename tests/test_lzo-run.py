import subprocess
import numpy as np


def run_lzo_run():
    result = subprocess.run(
        ["./bin/lzo-run", "-m1,2", "-d1", "-L50", "-k10", "-I1", "./tests/refs/ar-run_l1000.txt"],
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


def test_lzo_run_regression():
    out = run_lzo_run()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/lzo-run_m1_2d1L50k10I1.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
