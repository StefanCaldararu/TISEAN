import subprocess
import numpy as np


def run_ar():
    result = subprocess.run(
        ["./bin/arima-model", "-p10", "-P2,0,1", "-V0", "./tests/refs/henon_l1000.txt"],
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


def test_ar_regression():
    out = run_ar()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/arima-model_p10.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)