import subprocess
import numpy as np


def run_timerev():
    result = subprocess.run(
        ["./bin/timerev", "-d1", "-l300", "./tests/refs/ar-run_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # timerev prints a single numeric value followed by the input
        # filename on the same line, so only the first token is numeric
        if len(parts) >= 1:
            try:
                data.append(float(parts[0]))
            except ValueError:
                continue

    return np.array(data)


def test_timerev_regression():
    out = run_timerev()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/timerev_d1l300.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
