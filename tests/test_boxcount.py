import subprocess
import numpy as np


def run_boxcount():
    subprocess.run(
        ["./bin/boxcount", "-l", "200", "-M", "1,3",
         "-o", "./tests/refs/boxcount_out.txt", "./tests/refs/ar-run_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    with open("./tests/refs/boxcount_out.txt") as f:
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


def test_boxcount_regression():
    out = run_boxcount()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/boxcount_ref.txt", comments="#")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
