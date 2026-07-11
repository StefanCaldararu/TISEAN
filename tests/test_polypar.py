import subprocess
import tempfile
import os
import numpy as np


def run_polypar():
    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = os.path.join(tmpdir, "polypar_out.pol")
        subprocess.run(
            ["./bin/polypar", "-m2", "-p3", "-o", outfile],
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


def test_polypar_regression():
    out = run_polypar()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/parameter_m2p3.pol")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
