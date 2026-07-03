import subprocess
import numpy as np


def run_upo():
    result = subprocess.run(
        ["./bin/upo", "-m2", "-r0.1", "-p1", "-l300", "./tests/refs/henon_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only the "period / accuracy / stability" numeric rows
        if len(parts) == 3:
            try:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    return np.array(data)


def test_upo_regression():
    out = run_upo()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/upo_m2r01p1l300.txt").reshape(data.shape)

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
