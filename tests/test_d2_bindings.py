import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout/output
# files can't be tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

VECTOR_TOO_LARGE_FOR_LENGTH = 100
RESCALE_DATA_ZERO_INTERVAL = 11

AR_RUN = "tests/refs/ar-run_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
D2_BIN = os.path.abspath("./bin/d2")


def run_d2(args, datafile, out_prefix):
    subprocess.run(
        [D2_BIN] + args + ["-o", str(out_prefix), datafile],
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        ext: open(f"{out_prefix}.{ext}").read() for ext in ("c2", "h2", "d2", "stat")
    }


def parse_blocks(text):
    """Returns {block_index (1-based): (M, 2) array of [eps, value]} for a
    d2-style output file: repeated '#dim= N' headers followed by
    blank-line-separated blocks of two-column numeric rows, exactly what
    the CLI's .c2/.h2/.d2 files contain."""
    blocks = {}
    cur_block = None
    cur_rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#dim="):
            if cur_block is not None:
                blocks[cur_block] = np.array(cur_rows).reshape(-1, 2)
            cur_block = int(line.split("=")[1])
            cur_rows = []
        elif line.startswith("#"):
            continue
        else:
            parts = line.split()
            cur_rows.append([float(parts[0]), float(parts[1])])
    if cur_block is not None:
        blocks[cur_block] = np.array(cur_rows).reshape(-1, 2)
    return blocks


def py_block(result, table, block_index):
    """block_index is 1-based, matching the CLI's '#dim=' label - table is
    one of "c2"/"h2"/"d2". Filters out the NaN-gated entries the same way
    the CLI's own fprintf() guards would have skipped them (see d2.h)."""
    values = getattr(result, table)[block_index - 1]
    mask = ~np.isnan(values)
    return np.stack([result.eps[mask], values[mask]], axis=1)


def load_series(path, columns, length=None, exclude=0):
    """Replicates get_multi_series()'s -x/-l/-c handling: skip `exclude`
    lines, keep up to `length` of the rest, then pick `columns` (1-indexed,
    in the given order). Returns a (dim, length) array."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    return raw[:, [c - 1 for c in columns]].T.copy()


def assert_matches_cli(result, cli_files):
    for table, ext in (("c2", "c2"), ("h2", "h2"), ("d2", "d2")):
        cli_blocks = parse_blocks(cli_files[ext])
        assert len(cli_blocks) == result.n_blocks
        for block_index in cli_blocks:
            cli_rows = cli_blocks[block_index]
            py_rows = py_block(result, table, block_index)
            np.testing.assert_allclose(
                py_rows, cli_rows, **CLI_TEXT_TOL,
                err_msg=f"table={table} block={block_index}",
            )


# label, cli_args, kwargs, datafile, columns, length, exclude
CASES = [
    ("defaults", ["-l300", "-M1,3"], dict(embed=3), AR_RUN, [1], 300, 0),
    ("embed_5", ["-l300", "-M1,5"], dict(embed=5), AR_RUN, [1], 300, 0),
    ("multi_dim", ["-l300", "-M2,3", "-c1,2"], dict(embed=3), HENON, [1, 2], 300, 0),
    ("delay_2", ["-l300", "-d2", "-M1,4"], dict(embed=4, delay=2), AR_RUN, [1], 300, 0),
    ("theiler_5", ["-l300", "-t5", "-M1,3"], dict(embed=3, theiler=5), AR_RUN, [1], 300, 0),
    (
        "abs_eps",
        ["-l300", "-R0.5", "-r0.001", "-M1,3"],
        dict(embed=3, eps_max=0.5, eps_max_absolute=True, eps_min=0.001, eps_min_absolute=True),
        AR_RUN, [1], 300, 0,
    ),
    ("howoften_20", ["-l300", "-#20", "-M1,3"], dict(embed=3, howoften=20), AR_RUN, [1], 300, 0),
    ("maxfound_unlimited", ["-l300", "-N0", "-M1,3"], dict(embed=3, maxfound=0), AR_RUN, [1], 300, 0),
    ("maxfound_200", ["-l300", "-N200", "-M1,3"], dict(embed=3, maxfound=200), AR_RUN, [1], 300, 0),
    ("rescale", ["-l300", "-E", "-M1,3"], dict(embed=3, rescale=True), AR_RUN, [1], 300, 0),
    ("exclude", ["-x50", "-l300", "-M1,3"], dict(embed=3), AR_RUN, [1], 300, 50),
    ("verbosity_0", ["-l300", "-V0", "-M1,3"], dict(embed=3), AR_RUN, [1], 300, 0),
]


@pytest.mark.parametrize(
    "label,cli_args,kwargs,datafile,columns,length,exclude",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli(
    label, cli_args, kwargs, datafile, columns, length, exclude, tmp_path
):
    cli_files = run_d2(cli_args, datafile, tmp_path / "out")

    series = load_series(datafile, columns, length=length, exclude=exclude)
    result = tisean.d2.compute(series, **kwargs)

    assert result.dim == len(columns)
    assert result.embed == kwargs["embed"]
    assert result.n_blocks == len(columns) * kwargs["embed"]
    assert_matches_cli(result, cli_files)


def test_compute_default_kwargs_match_cli_defaults(tmp_path):
    # No -M/-d/-t/-R/-r/-#/-N/-E at all: defaults are dim,embed=1,10,
    # delay=1, theiler=0, eps_max=1.0 (factor), eps_min=1e-3 (factor),
    # howoften=100, maxfound=1000, not rescaled.
    cli_files = run_d2(["-l300"], AR_RUN, tmp_path / "out")

    series = load_series(AR_RUN, [1], length=300)
    result = tisean.d2.compute(series)

    assert result.dim == 1
    assert result.embed == 10
    assert result.howoften == 100
    assert_matches_cli(result, cli_files)


def test_compute_rejects_delay_vector_too_large_like_cli(tmp_path):
    result = subprocess.run(
        [D2_BIN, "-M1,500", "-l300", "-o", str(tmp_path / "out"), AR_RUN],
        capture_output=True,
        text=True,
    )
    assert result.returncode == VECTOR_TOO_LARGE_FOR_LENGTH

    series = load_series(AR_RUN, [1], length=300)
    with pytest.raises(ValueError):
        tisean.d2.compute(series, embed=500)


def test_compute_rejects_constant_data_with_rescale_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 20) + "\n")

    result = subprocess.run(
        [D2_BIN, "-E", "-o", str(tmp_path / "out"), str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL

    series = np.full((1, 20), 1.5)
    with pytest.raises(ValueError):
        tisean.d2.compute(series, rescale=True)


def test_compute_does_not_mutate_input_series():
    series = load_series(AR_RUN, [1], length=300)
    original = series.copy()

    tisean.d2.compute(series, embed=3, rescale=True)

    np.testing.assert_array_equal(series, original)


def test_compute_rejects_bad_series_shape():
    series_1d = np.arange(10.0)
    with pytest.raises(ValueError, match="2D array"):
        tisean.d2.compute(series_1d)


@pytest.mark.parametrize("bad_kwarg", ["embed", "delay", "howoften"])
def test_compute_rejects_zero_parameters(bad_kwarg):
    series = load_series(AR_RUN, [1], length=300)
    with pytest.raises(ValueError):
        tisean.d2.compute(series, **{bad_kwarg: 0})
