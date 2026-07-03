import subprocess
import numpy as np


def run_mutual():
    result = subprocess.run(
        ["./bin/mutual", "-c1", "-b8", "-D10", "./tests/refs/ar-run_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers like "#shannon= ...")
        if len(parts) == 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue

    return np.array(data)


def test_mutual_regression():
    out = run_mutual()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/mutual_c1b8D10.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
