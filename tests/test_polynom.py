import subprocess
import tempfile
import os
import numpy as np


def run_polynom():
    # -L triggers make_cast(), which used to fclose() the output file that
    # main() then fclose()s again -- a double free that glibc catches and
    # aborts on (SIGABRT). subprocess's check=True turns that abort into a
    # CalledProcessError, failing this test.
    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = os.path.join(tmpdir, "out.pol")
        subprocess.run(
            ["./bin/polynom", "-m2", "-d1", "-n300", "-L100",
             "-o", outfile, "./tests/refs/ar-run_l1000.txt"],
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

        # keep only numeric rows (skip "#..." headers)
        if len(parts) == 1:
            try:
                data.append(float(parts[0]))
            except ValueError:
                continue

    return np.array(data)


def test_polynom_regression():
    out = run_polynom()
    data = parse_output(out)

    ref = np.loadtxt("tests/refs/polynom_m2d1n300L100.txt")

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
