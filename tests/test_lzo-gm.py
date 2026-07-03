import subprocess
import numpy as np


def run_lzo_gm():
    result = subprocess.run(
        ["./bin/lzo-gm", "-m1,2", "-d1", "-i50", "./tests/refs/ar-run_l1000.txt"],
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


def test_lzo_gm_regression():
    out = run_lzo_gm()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/lzo-gm_m1_2d1i50.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
