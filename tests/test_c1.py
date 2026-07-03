import subprocess
import numpy as np


def run_c1():
    subprocess.run(
        ["./bin/c1", "-d1", "-m1", "-M3", "-t0", "-n10", "-l300",
         "-o", "./tests/refs/c1_out.txt", "./tests/refs/ar-run_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    with open("./tests/refs/c1_out.txt") as f:
        return f.read()


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers/blank lines)
        if len(parts) == 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue

    return np.array(data)


def test_c1_regression():
    out = run_c1()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/c1_ref.txt", comments="#")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
