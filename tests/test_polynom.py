import subprocess

POLYNOM_BIN = "./bin/polynom"
DATAFILE = "tests/refs/ar-run_l1000.txt"
REF = "tests/refs/polynom_l1000.txt"


def test_generate_polynom_reference():
    cases = [
        ["-m2", "-d1", "-p2"],
        ["-m2", "-d1", "-p2", "-n800"],
        ["-m2", "-d1", "-p2", "-n800", "-L50"],
        ["-m3", "-d2", "-p3"],
        ["-m3", "-d2", "-p3", "-n800"],
        ["-m3", "-d2", "-p3", "-n800", "-L50"],
    ]
    report = []
    for flags in cases:
        outfile = "/tmp/" + "_".join(f.lstrip("-") for f in flags) + ".pol"
        result = subprocess.run(
            [POLYNOM_BIN] + flags + ["-o", outfile, DATAFILE],
            capture_output=True,
            text=True,
        )
        report.append(
            f"flags={flags} returncode={result.returncode} "
            f"stderr={result.stderr!r} stdout={result.stdout!r}"
        )
    assert False, "\n".join(report)
