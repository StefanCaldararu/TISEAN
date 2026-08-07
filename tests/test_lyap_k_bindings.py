import os
import subprocess

import numpy as np
import pytest

tisean = pytest.importorskip("tisean")

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout/output
# file can't be tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

RESCALE_DATA_ZERO_INTERVAL = 11
LYAP_K__MAXITER_TOO_LARGE = 72

LORENZ = "tests/refs/lorenz_l1000.txt"
HENON = "tests/refs/henon_l1000.txt"
LYAP_K_BIN = os.path.abspath("./bin/lyap_k")


def run_cli(args, datafile, tmp_path, name="out.lyap"):
    outfile = tmp_path / name
    subprocess.run(
        [LYAP_K_BIN] + args + ["-o", str(outfile), datafile],
        capture_output=True,
        text=True,
        check=True,
    )
    return outfile.read_text()


def parse_output(text):
    """lyap_k's output file is grouped into blocks: a '#epsilon= <eps>
    dim= <d>' header line followed by '<iter> <value> <count>' rows (only
    for iteration steps that had at least one contributing reference
    point), each block terminated by a blank line."""
    rows = []
    eps = None
    dim = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#epsilon="):
            parts = line.split()
            eps = float(parts[1])
            dim = int(parts[3])
            continue
        parts = line.split()
        if len(parts) == 3:
            rows.append([eps, dim, int(parts[0]), float(parts[1]), int(parts[2])])
    return np.array(rows).reshape(-1, 5)


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
    rows = []
    ndim = result.maxdim - result.mindim + 1
    for e in range(result.epscount):
        for d in range(ndim):
            dim = result.mindim + d
            for j in range(result.maxiter + 1):
                if result.count[e, d, j]:
                    rows.append([
                        result.epsilon[e], dim, j,
                        result.lyap[e, d, j] / result.count[e, d, j],
                        result.count[e, d, j],
                    ])
    return np.array(rows).reshape(-1, 5)


CASES = [
    # label, args, datafile, column, length, exclude,
    # mindim, maxdim, delay, epsmin, epsmax, eps0set, eps1set,
    # epscount, reference, maxiter, window
    ("defaults", ["-l400", "-s15"], LORENZ, 1, 400, 0,
     2, 2, 1, 1.e-3, 1.e-2, False, False, 5, None, 15, 0),
    ("maxdim_4", ["-l400", "-s15", "-M4"], LORENZ, 1, 400, 0,
     2, 4, 1, 1.e-3, 1.e-2, False, False, 5, None, 15, 0),
    ("mindim_3_maxdim_4", ["-l400", "-s15", "-m3", "-M4"], LORENZ, 1, 400, 0,
     3, 4, 1, 1.e-3, 1.e-2, False, False, 5, None, 15, 0),
    ("delay_2", ["-l400", "-s15", "-d2"], LORENZ, 1, 400, 0,
     2, 2, 2, 1.e-3, 1.e-2, False, False, 5, None, 15, 0),
    ("raw_eps", ["-l400", "-s15", "-r0.05", "-R0.5", "-#3"], LORENZ, 1, 400, 0,
     2, 2, 1, 0.05, 0.5, True, True, 3, None, 15, 0),
    ("epscount_3", ["-l400", "-s15", "-#3"], LORENZ, 1, 400, 0,
     2, 2, 1, 1.e-3, 1.e-2, False, False, 3, None, 15, 0),
    ("reference_50", ["-l400", "-s15", "-n50"], LORENZ, 1, 400, 0,
     2, 2, 1, 1.e-3, 1.e-2, False, False, 5, 50, 15, 0),
    ("window_5", ["-l400", "-s15", "-t5"], LORENZ, 1, 400, 0,
     2, 2, 1, 1.e-3, 1.e-2, False, False, 5, None, 15, 5),
    ("column_2", ["-l400", "-s15", "-c2"], LORENZ, 2, 400, 0,
     2, 2, 1, 1.e-3, 1.e-2, False, False, 5, None, 15, 0),
    ("length_300", ["-l300", "-s15"], LORENZ, 1, 300, 0,
     2, 2, 1, 1.e-3, 1.e-2, False, False, 5, None, 15, 0),
    ("exclude_50", ["-x50", "-l300", "-s15"], LORENZ, 1, 300, 50,
     2, 2, 1, 1.e-3, 1.e-2, False, False, 5, None, 15, 0),
    ("verbosity_0", ["-l400", "-s15", "-V0"], LORENZ, 1, 400, 0,
     2, 2, 1, 1.e-3, 1.e-2, False, False, 5, None, 15, 0),
    ("henon", ["-l400", "-s15"], HENON, 1, 400, 0,
     2, 2, 1, 1.e-3, 1.e-2, False, False, 5, None, 15, 0),
    (
        "combo_mindim3_maxdim4_delay2_window3_ref100",
        ["-l400", "-s15", "-m3", "-M4", "-d2", "-t3", "-n100"],
        LORENZ, 1, 400, 0,
        3, 4, 2, 1.e-3, 1.e-2, False, False, 5, 100, 15, 3,
    ),
]


@pytest.mark.parametrize(
    "label,args,datafile,column,length,exclude,mindim,maxdim,delay,epsmin,"
    "epsmax,eps0set,eps1set,epscount,reference,maxiter,window",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_compute_matches_cli(
    label, args, datafile, column, length, exclude, mindim, maxdim, delay,
    epsmin, epsmax, eps0set, eps1set, epscount, reference, maxiter, window,
    tmp_path,
):
    cli_text = run_cli(args, datafile, tmp_path)
    cli_result = parse_output(cli_text)

    series = load_column(datafile, column, length=length, exclude=exclude)
    kwargs = dict(
        mindim=mindim, maxdim=maxdim, delay=delay, epsmin=epsmin,
        epsmax=epsmax, eps0set=eps0set, eps1set=eps1set, epscount=epscount,
        maxiter=maxiter, window=window,
    )
    if reference is not None:
        kwargs["reference"] = reference
    result = tisean.lyap_k.compute(series, **kwargs)

    assert result.mindim == mindim
    assert result.maxdim == maxdim
    assert result.maxiter == maxiter

    got = py_rows(result)
    assert got.shape == cli_result.shape
    np.testing.assert_allclose(got[:, 0], cli_result[:, 0], **CLI_TEXT_TOL)
    np.testing.assert_array_equal(got[:, 1], cli_result[:, 1])
    np.testing.assert_array_equal(got[:, 2], cli_result[:, 2])
    np.testing.assert_allclose(got[:, 3], cli_result[:, 3], **CLI_TEXT_TOL)
    np.testing.assert_array_equal(got[:, 4], cli_result[:, 4])


def test_compute_matches_cli_dash_o_output_file(tmp_path):
    cli_text = run_cli(["-l400", "-s15", "-t5"], LORENZ, tmp_path, name="custom.lyap")
    cli_result = parse_output(cli_text)

    series = load_column(LORENZ, 1, length=400)
    result = tisean.lyap_k.compute(series, maxiter=15, window=5)

    got = py_rows(result)
    assert got.shape == cli_result.shape
    np.testing.assert_allclose(got[:, 3], cli_result[:, 3], **CLI_TEXT_TOL)


def test_compute_default_kwargs_match_cli_documented_defaults():
    # show_options(): -M default 2, -m default 2, -d default 1, -r default
    # "(data interval)/1000" i.e. epsmin=1.e-3 used directly in the
    # already-rescaled [0,1) space (eps0set stays unset/False), -R default
    # "(data interval)/100" i.e. epsmax=1.e-2 (eps1set stays unset/False),
    # -# default 5, -n default "# of data" (reference left at its sentinel
    # default so every point is used), -s default 50, -t default 0.
    series = load_column(LORENZ, 1, length=400)

    default = tisean.lyap_k.compute(series, maxiter=15)
    explicit = tisean.lyap_k.compute(
        series, mindim=2, maxdim=2, delay=1, epsmin=1.e-3, epsmax=1.e-2,
        eps0set=False, eps1set=False, epscount=5, maxiter=15, window=0,
    )

    assert default.epscount == explicit.epscount == 5
    assert default.mindim == explicit.mindim == 2
    assert default.maxdim == explicit.maxdim == 2
    np.testing.assert_array_equal(default.epsilon, explicit.epsilon)
    np.testing.assert_array_equal(default.count, explicit.count)
    np.testing.assert_array_equal(default.lyap, explicit.lyap)


def test_compute_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 200) + "\n")
    outfile = tmp_path / "out.lyap"

    result = subprocess.run(
        [LYAP_K_BIN, "-l200", "-s15", "-o", str(outfile), str(datafile)],
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

    series = np.full(200, 1.5)
    with pytest.raises(ValueError):
        tisean.lyap_k.compute(series, maxiter=15)


def test_compute_rejects_too_few_points_like_cli(tmp_path):
    datafile = tmp_path / "short.txt"
    datafile.write_text("\n".join(str(float(i)) for i in range(10)) + "\n")
    outfile = tmp_path / "out.lyap"

    result = subprocess.run(
        [LYAP_K_BIN, "-l10", "-s50", "-o", str(outfile), str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == LYAP_K__MAXITER_TOO_LARGE
    assert "Too few points to handle these parameters!" in result.stderr
    assert outfile.stat().st_size == 0

    series = np.arange(10.0)
    with pytest.raises(ValueError):
        tisean.lyap_k.compute(series, maxiter=50)


def test_compute_rejects_series_too_short_for_mindim_maxdim_delay_maxiter():
    # The box-building step reads series[i] for i up to
    # length-(maxdim-1)*delay-maxiter, so length must be >
    # (maxdim-1)*delay+maxiter; with the defaults (maxdim=2, delay=1,
    # maxiter=50) that means length must be > 51. This is a memory-safety
    # contract (see lyap_k.h): below this bound the CLI itself reads out
    # of bounds too.
    rng = np.random.default_rng(0)
    too_short = rng.normal(size=51)
    with pytest.raises(ValueError):
        tisean.lyap_k.compute(too_short, mindim=2, maxdim=2, delay=1, maxiter=50)


def test_compute_clamps_maxdim_below_2_like_cli(tmp_path):
    # show_options() documents -M's default as 2 and the CLI silently
    # floors any smaller value to 2 (main() clamps *after* its own
    # too-few-points check, which is why lyap_k_compute()'s Python binding
    # re-validates using the clamped value - see lyap_k.h). This just
    # confirms the clamped behavior itself is preserved.
    cli_text = run_cli(["-l400", "-s15", "-M1"], LORENZ, tmp_path)
    cli_result = parse_output(cli_text)

    series = load_column(LORENZ, 1, length=400)
    result = tisean.lyap_k.compute(series, maxdim=1, maxiter=15)

    assert result.maxdim == 2
    got = py_rows(result)
    assert got.shape == cli_result.shape
    np.testing.assert_allclose(got[:, 3], cli_result[:, 3], **CLI_TEXT_TOL)
