import subprocess
import numpy as np


def run_lfo_ar():
    result = subprocess.run(
        ["./bin/lfo-ar", "-m1,2", "-d1", "-l300", "-i50", "./tests/refs/ar-run_l1000.txt"],
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
        if len(parts) == 5:
            try:
                data.append([float(p) for p in parts])
            except ValueError:
                continue

    return np.array(data)


def test_lfo_ar_regression():
    out = run_lfo_ar()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/lfo-ar_m12d1l300i50.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
