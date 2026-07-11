import subprocess
import numpy as np


def run_sav_gol():
    result = subprocess.run(
        ["./bin/sav_gol", "-c1", "-m1", "-n2,2", "-p2", "-D0", "-l300",
         "./tests/refs/ar-run_l1000.txt"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers)
        if len(parts) == 1:
            try:
                data.append(float(parts[0]))
            except ValueError:
                continue

    return np.array(data)


def test_sav_gol_regression():
    out = run_sav_gol()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/sav_gol_c1m1n22p2D0l300.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
