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
    for order in [10, 15, 20, 25, 30, 40, 50]:
        result = subprocess.run(
            ["./bin/resample", "-c1", "-l1000", f"-p{order}", "-s0.5",
             "./tests/refs/ar-run_l1000.txt"],
            capture_output=True,
            text=True,
        )
        report.append(f"order={order} rc={result.returncode} stderr={result.stderr!r}")
    pytest.fail("CAPTURE:\n" + "\n".join(report) + "\nEND CAPTURE")
