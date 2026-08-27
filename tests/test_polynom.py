import subprocess

POLYNOM_BIN = "./bin/polynom"
DATAFILE = "tests/refs/ar-run_l1000.txt"
REF = "tests/refs/polynom_l1000.txt"


def test_generate_polynom_reference():
    subprocess.run(
        [POLYNOM_BIN, "-m3", "-d2", "-p3", "-n800", "-L50", "-o", REF, DATAFILE],
        capture_output=True,
        text=True,
        check=True,
    )
    with open(REF) as f:
        content = f.read()
    assert "#number of free parameters" in content
    assert "#average insample error" in content
    assert "#average out of sample error" in content
