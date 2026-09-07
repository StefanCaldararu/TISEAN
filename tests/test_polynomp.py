import subprocess
import numpy as np


def run_polynomp():
    result = subprocess.run(
        ["./bin/polynomp", "-m2", "-d1", "-n300", "-L100",
         "-p", "tests/refs/parameter_m2p3.pol", "./tests/refs/ar-run_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip "#..." headers)
        if len(parts) == 1:
            try:
                data.append(float(parts[0]))
            except ValueError:
                continue

    return np.array(data)


def test_polynomp_regression():
    out = run_polynomp()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/polynomp_m2d1n300L100.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
