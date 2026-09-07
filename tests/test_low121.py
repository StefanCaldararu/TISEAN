import subprocess
import numpy as np


def run_low121():
    result = subprocess.run(
        ["./bin/low121", "-c1", "-i2", "./tests/refs/ar-run_l1000.txt"],
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


def test_low121_regression():
    out = run_low121()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/low121_c1i2.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
