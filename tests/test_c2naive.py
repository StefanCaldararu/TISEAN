import subprocess
import numpy as np


def run_c2naive():
    subprocess.run(
        ["./bin/c2naive", "-d1", "-M3", "-t0", "-l300",
         "-o", "./tests/refs/c2naive_out.txt", "./tests/refs/ar-run_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    with open("./tests/refs/c2naive_out.txt") as f:
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


def test_c2naive_regression():
    out = run_c2naive()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/c2naive_ref.txt", comments="#")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
