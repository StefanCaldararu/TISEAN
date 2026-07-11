import subprocess
import numpy as np


def run_d2():
    subprocess.run(
        ["./bin/d2", "-d1", "-M1,3", "-l300", "-N200",
         "-o", "./tests/refs/d2_out", "./tests/refs/ar-run_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    with open("./tests/refs/d2_out.d2") as f:
        return f.read()


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


def test_d2_regression():
    out = run_d2()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/d2_ref.d2", comments="#")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
