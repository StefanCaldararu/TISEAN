import subprocess
import numpy as np


def run_c2g():
    subprocess.run(
        ["./bin/c2g", "-o", "./tests/refs/c2g_out.txt", "./tests/refs/c2naive_ref.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    with open("./tests/refs/c2g_out.txt") as f:
        return f.read()


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers/blank lines)
        if len(parts) == 3:
            try:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    return np.array(data)


def test_c2g_regression():
    out = run_c2g()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/c2g_ref.txt", comments="#")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
