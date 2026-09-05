import subprocess
import numpy as np


def run_lazy():
    result = subprocess.run(
        ["./bin/lazy", "-m2", "-r0.1", "-i1", "-l300", "./tests/refs/ar-run_l1000.txt"],
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


def test_lazy_regression():
    out = run_lazy()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/lazy_m2r01i1l300.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
