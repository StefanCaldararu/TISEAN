import subprocess
import numpy as np


def run_avd2():
    subprocess.run(
        ["./bin/av-d2", "-m1", "-M3", "-o", "./tests/refs/avd2_out.txt", "./tests/refs/d2_ref.d2"],
        capture_output=True,
        text=True,
        check=True
    )
    with open("./tests/refs/avd2_out.txt") as f:
        return f.read()


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip blank separators between dim blocks)
        if len(parts) == 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue

    return np.array(data)


def test_avd2_regression():
    out = run_avd2()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/avd2_ref.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
