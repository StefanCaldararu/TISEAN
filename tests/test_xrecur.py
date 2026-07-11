import subprocess
import numpy as np


def run_xrecur():
    result = subprocess.run(
        ["./bin/xrecur", "-m1,2", "-d1", "-r0.5", "-l150", "-L150",
         "-o", "./tests/refs/xrecur_run_out.txt",
         "./tests/refs/ar-run_l1000.txt", "./tests/refs/henon_l1000.txt"],
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


def test_xrecur_regression():
    run_xrecur()
    with open("./tests/refs/xrecur_run_out.txt") as f:
        out = f.read()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/xrecur_m12d1r05l150L150.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
