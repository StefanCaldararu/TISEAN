import subprocess

import pytest


def test_capture_resample_reference():
    result = subprocess.run(
        ["./bin/resample", "-c1", "-l20", "-p4", "-s0.5", "./tests/refs/ar-run_l1000.txt"],
        capture_output=True,
        text=True,
        check=True,
    )
    pytest.fail("CAPTURE:\n" + result.stdout + "\nEND CAPTURE")


def test_capture_resample_singular_order():
    report = []
    for order in [80, 100, 150, 200, 250, 300, 400, 500]:
        result = subprocess.run(
            ["./bin/resample", "-c1", "-l1000", f"-p{order}", "-s0.5",
             "./tests/refs/ar-run_l1000.txt"],
            capture_output=True,
            text=True,
        )
        report.append(f"order={order} rc={result.returncode} stderr={result.stderr!r}")
    pytest.fail("CAPTURE:\n" + "\n".join(report) + "\nEND CAPTURE")


def test_capture_resample_short_series():
    report = []
    for length in [0, 1, 2, 3, 4, 5]:
        result = subprocess.run(
            ["./bin/resample", "-c1", f"-l{length}", "-p4", "-s0.5",
             "./tests/refs/ar-run_l1000.txt"],
            capture_output=True,
            text=True,
        )
        report.append(
            f"length={length} rc={result.returncode} stdout={result.stdout!r} "
            f"stderr_tail={result.stderr[-200:]!r}"
        )
    pytest.fail("CAPTURE:\n" + "\n".join(report) + "\nEND CAPTURE")
