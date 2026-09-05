import subprocess
import tempfile
import os
import numpy as np


def run_lyap_r():
    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = os.path.join(tmpdir, "lyap_r_out.txt")
        subprocess.run(
            ["./bin/lyap_r", "-m2", "-d1", "-s20", "-l500", "-o", outfile,
             "./tests/refs/lorenz_l1000.txt"],
            capture_output=True,
            text=True,
            check=True
        )
        with open(outfile) as f:
            return f.read()


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers)
        if len(parts) == 2:
            try:
                data.append([float(p) for p in parts])
            except ValueError:
                continue

    return np.array(data)


def test_lyap_r_regression():
    out = run_lyap_r()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/lyap_r_m2d1s20l500.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
