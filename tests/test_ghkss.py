import os
import subprocess
import numpy as np


def run_ghkss():
    # ghkss always writes 'datafile'.opt.N per iteration in addition to
    # echoing the last iteration to stdout when -o is omitted. Clean up
    # that side-effect artifact so the test doesn't litter tests/refs/.
    artifact = "./tests/refs/ar-run_l1000.txt.opt.1"
    try:
        result = subprocess.run(
            ["./bin/ghkss", "-m1,3", "-d1", "-q2", "-k20", "-i1", "-l300", "./tests/refs/ar-run_l1000.txt"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    finally:
        if os.path.exists(artifact):
            os.remove(artifact)


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


def test_ghkss_regression():
    out = run_ghkss()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/ghkss_m13d1q2k20i1l300.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
