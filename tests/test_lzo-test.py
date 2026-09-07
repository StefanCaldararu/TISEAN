import subprocess
import numpy as np


def run_lzo_test():
    result = subprocess.run(
        ["./bin/lzo-test", "-m1,2", "-d1", "-n50", "-k10", "./tests/refs/ar-run_l1000.txt"],
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


def test_lzo_test_regression():
    out = run_lzo_test()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/lzo-test_m1_2d1n50k10.txt")
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
