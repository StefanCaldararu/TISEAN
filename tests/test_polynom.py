import subprocess
import tempfile
import os


def run_polynom():
    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = os.path.join(tmpdir, "out.pol")
        result = subprocess.run(
            ["./bin/polynom", "-m2", "-d1", "-n300", "-L100",
             "-o", outfile, "./tests/refs/ar-run_l1000.txt"],
            capture_output=True,
            text=True,
            check=True
        )
        return result


def test_polynom_cast_does_not_crash():
    # -L triggers make_cast(), which used to fclose() the output file that
    # main() then fclose()s again -- a double free that glibc catches and
    # aborts on (SIGABRT). subprocess's check=True turns that abort into a
    # CalledProcessError, failing this test.
    run_polynom()
