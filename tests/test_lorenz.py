import subprocess
import numpy as np


def run_lorenz():
    result = subprocess.run(
        ["./bin/lorenz", "-l", "1000"],
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
        if len(parts) == 3:
            try:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    return np.array(data)


def test_lorenz_regression():
    out = run_lorenz()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/lorenz_l1000.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)