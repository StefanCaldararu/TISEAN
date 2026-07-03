import subprocess
import numpy as np


def run_compare():
    result = subprocess.run(
        ["./bin/compare", "-c1,2", "./tests/refs/henon_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    # compare writes its report to stderr rather than stdout
    return result.stderr


def parse_output(text):
    data = []
    for line in text.splitlines():
        s = line.strip()
        if (s.startswith("mean difference")
                or s.startswith("root mean squared difference")
                or s.startswith("standard deviation")):
            try:
                data.append(float(s.split()[-1]))
            except ValueError:
                continue

    return np.array(data)


def test_compare_regression():
    out = run_compare()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/compare_c12_henon.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
