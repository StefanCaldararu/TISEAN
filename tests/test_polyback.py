import subprocess
import shutil
import tempfile
import os
import numpy as np


def run_polyback():
    # polyback writes intermediate "<parameter file>.N" files next to the
    # parameter file as it eliminates terms, so use a scratch copy of the
    # parameter file to avoid littering tests/refs/ with those artifacts.
    with tempfile.TemporaryDirectory() as tmpdir:
        parfile = os.path.join(tmpdir, "parameter_m2p3.pol")
        shutil.copyfile("tests/refs/parameter_m2p3.pol", parfile)

        result = subprocess.run(
            ["./bin/polyback", "-m2", "-d1", "-n300", "-s1", "-#1",
             "-p", parfile, "./tests/refs/ar-run_l1000.txt"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout


def parse_output(text):
    data = []
    for line in text.splitlines():
        parts = line.strip().split()

        # keep only numeric rows (skip headers); only the first 3 columns
        # (term count, insample error, out-of-sample error) are present on
        # every row -- later rows also list the removed term indices.
        if len(parts) >= 3:
            try:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    return np.array(data)


def test_polyback_regression():
    out = run_polyback()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/polyback_m2d1n300s1_1.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
