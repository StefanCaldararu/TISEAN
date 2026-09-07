import subprocess
import numpy as np


def run_ar_run():
    result = subprocess.run(
        ["./bin/ar-run", "-l", "1000", "./tests/refs/ar-run_model.txt"],
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


def test_ar_run_regression():
    out = run_ar_run()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/ar-run_p2l1000.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
