import subprocess
import numpy as np


def run_ikeda():
    result = subprocess.run(
        ["./bin/ikeda", "-l", "500"],
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
        if len(parts) == 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue

    return np.array(data)


def test_ikeda_regression():
    out = run_ikeda()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/ikeda_l500.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
