import subprocess
import numpy as np


def run_histogram():
    result = subprocess.run(
        ["./bin/histogram", "-c1", "-b20", "-l300", "./tests/refs/ar-run_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers / comments)
        if len(parts) == 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue

    return np.array(data)


def test_histogram_regression():
    out = run_histogram()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/histogram_c1b20l300.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
