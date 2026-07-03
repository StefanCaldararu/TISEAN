import subprocess
import numpy as np


def run_upoembed():
    result = subprocess.run(
        ["./bin/upoembed", "-d1", "-m2", "-p2", "tests/refs/upoembed_input.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers/blank separator lines)
        if len(parts) == 2:
            try:
                data.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue

    return np.array(data)


def test_upoembed_regression():
    out = run_upoembed()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/upoembed_d1m2p2.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
