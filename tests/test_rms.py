import subprocess
import numpy as np


def run_rms():
    result = subprocess.run(
        ["./bin/rms", "-v", "-l300", "./tests/refs/ar-run_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers/blank lines)
        if len(parts) == 1:
            try:
                data.append(float(parts[0]))
            except ValueError:
                continue

    return np.array(data)


def test_rms_regression():
    out = run_rms()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/rms_v_l300.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
