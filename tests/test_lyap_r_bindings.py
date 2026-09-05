import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

RESCALE_DATA_ZERO_INTERVAL = 11

LORENZ = "tests/refs/lorenz_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
LYAP_R_BIN = os.path.abspath("./bin/lyap_r")


def run_cli(args, datafile, tmp_path, name="out.ros"):
    outfile = tmp_path / name
    subprocess.run(
        [LYAP_R_BIN] + args + ["-o", str(outfile), datafile],
        capture_output=True,
        text=True,
        check=True,
    )
    return outfile.read_text()


def parse_output(text):
    """lyap_r's output file is just '<step> <value>' rows, one per step
    that had at least one contributing neighbor pair - no header/comment
    lines at all."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            rows.append([int(parts[0]), float(parts[1])])
    return np.array(rows).reshape(-1, 2)


def load_column(path, column, length=None, exclude=0):
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    return raw[:, column - 1].copy()


def py_rows(result):
    rows = [
        [i, result.lyap[i] / result.found[i] / 2.0]
        for i in range(result.steps + 1)
        if result.found[i]
    ]
    return np.array(rows).reshape(-1, 2)


CASES = [
    # label, args, datafile, column, length, exclude, dim, delay, mindist, steps, eps0, epsset
    ("defaults", ["-l500"], LORENZ, 1, 500, 0, 2, 1, 0, 10, 1.e-3, False),
    ("dim_3", ["-l500", "-m3"], LORENZ, 1, 500, 0, 3, 1, 0, 10, 1.e-3, False),
    ("delay_2", ["-l500", "-d2"], LORENZ, 1, 500, 0, 2, 2, 0, 10, 1.e-3, False),
    ("mindist_5", ["-l500", "-t5"], LORENZ, 1, 500, 0, 2, 1, 5, 10, 1.e-3, False),
    ("steps_20", ["-l500", "-s20"], LORENZ, 1, 500, 0, 2, 1, 0, 20, 1.e-3, False),
    ("raw_eps", ["-l500", "-r0.5"], LORENZ, 1, 500, 0, 2, 1, 0, 10, 0.5, True),
    ("column_2", ["-l500", "-c2"], LORENZ, 2, 500, 0, 2, 1, 0, 10, 1.e-3, False),
    ("length_300", ["-l300"], LORENZ, 1, 300, 0, 2, 1, 0, 10, 1.e-3, False),
    ("exclude_50", ["-x50", "-l300"], LORENZ, 1, 300, 50, 2, 1, 0, 10, 1.e-3, False),
    ("verbosity_0", ["-l500", "-V0"], LORENZ, 1, 500, 0, 2, 1, 0, 10, 1.e-3, False),
    ("henon", ["-l500"], HENON, 1, 500, 0, 2, 1, 0, 10, 1.e-3, False),
    (
        "combo_dim3_delay2_mindist2_steps15",
        ["-l500", "-m3", "-d2", "-t2", "-s15"],
        LORENZ,
        1,
        500,
        0,
        3,
        2,
        2,
        15,
        1.e-3,
        False,
    ),
]


@pytest.mark.parametrize(
    "label,args,datafile,column,length,exclude,dim,delay,mindist,steps,eps0,epsset",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli(
    label, args, datafile, column, length, exclude, dim, delay, mindist, steps,
    eps0, epsset, tmp_path,
):
    cli_text = run_cli(args, datafile, tmp_path)
    cli_result = parse_output(cli_text)

    series = load_column(datafile, column, length=length, exclude=exclude)
    result = tisean.lyap_r.compute(
        series, dim=dim, delay=delay, mindist=mindist, steps=steps,
        eps0=eps0, epsset=epsset,
    )

    assert result.steps == steps
    got = py_rows(result)

    assert got.shape == cli_result.shape
    np.testing.assert_array_equal(got[:, 0], cli_result[:, 0])
    np.testing.assert_allclose(got[:, 1], cli_result[:, 1], **CLI_TEXT_TOL)


def test_compute_matches_cli_dash_o_output_file(tmp_path):
    cli_text = run_cli(["-l500", "-s15"], LORENZ, tmp_path, name="custom.ros")
    cli_result = parse_output(cli_text)

    series = load_column(LORENZ, 1, length=500)
    result = tisean.lyap_r.compute(series, steps=15)

    got = py_rows(result)
    assert got.shape == cli_result.shape
    np.testing.assert_allclose(got[:, 1], cli_result[:, 1], **CLI_TEXT_TOL)


def test_compute_default_kwargs_match_cli_documented_defaults():
    # show_options(): -m default 2, -d default 1, -t default 0, -s default
    # 10, -r default "(data interval)/1000" i.e. eps0=1.e-3 used directly
    # in the already-rescaled [0,1) space (epsset stays unset/False).
    series = load_column(LORENZ, 1, length=500)

    default = tisean.lyap_r.compute(series)
    explicit = tisean.lyap_r.compute(
        series, dim=2, delay=1, mindist=0, steps=10, eps0=1.e-3, epsset=False,
    )

    assert default.steps == explicit.steps == 10
    np.testing.assert_array_equal(default.found, explicit.found)
    np.testing.assert_array_equal(default.lyap, explicit.lyap)


def test_compute_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 50) + "\n")
    outfile = tmp_path / "out.ros"

    result = subprocess.run(
        [LYAP_R_BIN, "-l50", "-o", str(outfile), str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL
    assert "data ranges from 1.500000e+00 to 1.500000e+00" in result.stderr
    # Matches the CLI: test_outfile() (called while parsing options, before
    # the constant-data check) touches the output file to verify write
    # access, but the check itself runs before the file is ever opened for
    # writing, so it stays empty on this error path.
    assert outfile.stat().st_size == 0

    series = np.full(50, 1.5)
    with pytest.raises(ValueError):
        tisean.lyap_r.compute(series)


def test_compute_rejects_dim_zero():
    series = load_column(LORENZ, 1, length=500)
    with pytest.raises(ValueError):
        tisean.lyap_r.compute(series, dim=0)


def test_compute_rejects_series_too_short_for_dim_delay_steps():
    # The box-building step reads series[i] for i up to
    # length-delay*(dim-1)-steps, so length must be > delay*(dim-1)+steps;
    # with the defaults (dim=2, delay=1, steps=10) that means length must
    # be > 11. This is a memory-safety contract (see lyap_r.h): below this
    # bound the CLI itself reads out of bounds too. Above it, whether the
    # search loop actually terminates is a separate, data-dependent
    # question (the CLI's own geometric eps growth can run forever if no
    # two points ever end up close enough - not something a shape check
    # can rule out), so this only checks the boundary itself; the
    # parametrized cases above already cover realistic data that
    # terminates normally.
    rng = np.random.default_rng(0)
    too_short = rng.normal(size=11)
    with pytest.raises(ValueError):
        tisean.lyap_r.compute(too_short, dim=2, delay=1, steps=10)
