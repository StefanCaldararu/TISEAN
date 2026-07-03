import subprocess
import tempfile
import os
import numpy as np


def run_lyap_k():
    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = os.path.join(tmpdir, "lyap_k_out.txt")
        subprocess.run(
            ["./bin/lyap_k", "-m2", "-M2", "-d1", "-s20", "-l500", "-o", outfile,
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

        # keep only numeric rows (skip "#epsilon=..." headers)
        if len(parts) == 3:
            try:
                data.append([float(p) for p in parts])
            except ValueError:
                continue

    return np.array(data)


def test_lyap_k_regression():
    out = run_lyap_k()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/lyap_k_m2M2d1s20l500.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
