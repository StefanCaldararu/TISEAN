import shutil
import uuid
from pathlib import Path

import pytest

# The Fortran binaries under test store their -o (and -i) path in a
# character*72 (see source_f/lorenz.f:30, source_f/henon.f:27,
# source_f/ikeda.f:27, and 32 others). A path longer than that is
# silently truncated -- no error, the binary just writes somewhere else
# -- so tests must never hand these binaries an absolute tmp_path: it's
# short enough on Linux CI to pass by accident, but on macOS
# (/private/var/folders/.../pytest-of-user/...) it routinely runs past
# 72 and breaks silently.
_MAX_FORTRAN_PATH = 72
# Generous upper bound on the filenames tests actually join onto the
# fixture's directory (e.g. "lorenz.out", "henon.out").
_MAX_FILENAME_LEN = 32


@pytest.fixture
def short_outdir():
    d = Path("tests") / ("t" + uuid.uuid4().hex[:8])
    d.mkdir()
    try:
        resolved = str(d.resolve())
        assert len(resolved) + 1 + _MAX_FILENAME_LEN < _MAX_FORTRAN_PATH, (
            f"short_outdir yielded a path too long for a Fortran "
            f"character*72 ({len(resolved)} chars): {resolved}"
        )
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)
