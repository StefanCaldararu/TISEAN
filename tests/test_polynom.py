import re
import subprocess

import numpy as np

POLYNOM_BIN = "./bin/polynom"
DATAFILE = "tests/refs/ar-run_l1000.txt"
REF = "tests/refs/polynom_l1000.txt"

# polynom always writes to a file (never stdout - there's no -o-less-means-
# stdout branch in its main()), defaulting to '<datafile>.pol'/'stdin.pol' or
# whatever -o names, so the CLI helper has to point -o at an explicit path
# and read that file back rather than capturing stdout.
NUMBER_RE = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")


def run_polynom(args, outfile):
    subprocess.run(
        [POLYNOM_BIN] + args + ["-o", str(outfile), DATAFILE],
        capture_output=True,
        text=True,
        check=True,
    )
    return outfile.read_text()


def parse_output(text):
    return np.array([float(x) for x in NUMBER_RE.findall(text)])


def test_polynom_regression(tmp_path):
    out = run_polynom(["-m3", "-d2", "-p3", "-n800", "-L50"], tmp_path / "out.pol")
    data = parse_output(out)

    with open(REF) as f:
        ref = parse_output(f.read())

    np.testing.assert_allclose(data, ref, rtol=1e-7, atol=1e-7)
