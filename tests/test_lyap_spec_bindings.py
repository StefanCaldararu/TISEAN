import json
import os
import subprocess
import sys

import numpy as np
import pytest

import tisean

# The CLI prints everything through "%e" (6 digits after the point, so ~7
# significant digits), so comparisons against numbers parsed from its
# stdout can't be tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

RESCALE_DATA_ZERO_INTERVAL = 11
LYAP_SPEC_NOT_ENOUGH_NEIGHBORS = 50
LYAP_SPEC_DATA_TOO_SHORT = 51

LORENZ = "tests/refs/lorenz_l1000.txt"
LYAP_SPEC_BIN = os.path.abspath("./bin/lyap_spec")


def run_cli(args, **kwargs):
    result = subprocess.run(
        [LYAP_SPEC_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_output(text):
    """Returns a dict with the CLI's final printed data row (count,
    exponents) plus the summary lines that follow it. lyap_spec prints a
    data row every OUT=10 wall-clock seconds *and* unconditionally on the
    final iteration - for the short/fast runs used here only the final row
    ever appears, but we take the last data row regardless to be safe."""
    data_rows = []
    rel_err = abs_err = None
    avg_neighborhood_size = avg_num_neighbors = ky_dimension = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#Average relative forecast errors:="):
            rel_err = [float(x) for x in line.split(":=", 1)[1].split()]
        elif line.startswith("#Average absolute forecast errors:="):
            abs_err = [float(x) for x in line.split(":=", 1)[1].split()]
        elif line.startswith("#Average Neighborhood Size="):
            avg_neighborhood_size = float(line.split("=", 1)[1])
        elif line.startswith("#Average num. of neighbors="):
            avg_num_neighbors = float(line.split("=", 1)[1])
        elif line.startswith("#estimated KY-Dimension="):
            ky_dimension = float(line.split("=", 1)[1])
        elif not line.startswith("#"):
            data_rows.append([float(x) for x in line.split()])

    count, *exponents = data_rows[-1]
    return dict(
        count=count,
        exponents=exponents,
        rel_forecast_error=rel_err,
        abs_forecast_error=abs_err,
        avg_neighborhood_size=avg_neighborhood_size,
        avg_num_neighbors=avg_num_neighbors,
        ky_dimension=ky_dimension,
    )


def load_multi_series(path, columns, length=None, exclude=0):
    """Replicates get_multi_series()'s -x/-l/-c handling: skip `exclude`
    lines, keep up to `length` of the rest, then pick `columns` (1-indexed,
    in the given order). Returns a (dim, length) array. Unlike ar-model,
    lyap_spec does its own rescaling internally, so - unlike
    tests/test_ar_model_bindings.py's load_columns() - there is no
    centering step here."""
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if exclude:
        raw = raw[exclude:]
    if length is not None:
        raw = raw[:length]
    raw = raw[:, [c - 1 for c in columns]]
    return raw.T.copy()


def run_py_compute(datafile, columns, length=None, exclude=0, kwargs=None):
    """Runs tisean.lyap_spec.compute() in a fresh subprocess and returns a
    dict shaped like parse_output()'s. A fresh process is required (not
    just a fresh call): rand.c's rnd_init() only actually (re)seeds once
    per process (a static flag guards it - see source_c/routines/rand.c),
    and lyap_spec_compute() calls rnd_init() on every invocation to build
    its initial tangent-vector basis, so a second compute() call in the
    same process silently continues the RNG stream left behind by the
    first call instead of reseeding - see
    test_compute_is_not_reproducible_across_calls_in_one_process below."""
    kwargs = kwargs or {}
    script = (
        "import json, numpy as np, tisean\n"
        f"raw = np.loadtxt({datafile!r})\n"
        "if raw.ndim == 1:\n"
        "    raw = raw.reshape(-1, 1)\n"
        f"raw = raw[{exclude}:]\n"
        f"raw = raw[:{length!r}]\n"
        f"raw = raw[:, {[c - 1 for c in columns]!r}]\n"
        "series = raw.T.copy()\n"
        f"r = tisean.lyap_spec.compute(series, **{kwargs!r})\n"
        "print(json.dumps({\n"
        "    'count': r.count,\n"
        "    'exponents': r.exponents.tolist(),\n"
        "    'rel_forecast_error': r.rel_forecast_error.tolist(),\n"
        "    'abs_forecast_error': r.abs_forecast_error.tolist(),\n"
        "    'avg_neighborhood_size': r.avg_neighborhood_size,\n"
        "    'avg_num_neighbors': r.avg_num_neighbors,\n"
        "    'ky_dimension': r.ky_dimension,\n"
        "}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def assert_matches_cli(py_result, cli_result):
    assert py_result["count"] == cli_result["count"]
    np.testing.assert_allclose(py_result["exponents"], cli_result["exponents"], **CLI_TEXT_TOL)
    np.testing.assert_allclose(
        py_result["rel_forecast_error"], cli_result["rel_forecast_error"], **CLI_TEXT_TOL
    )
    np.testing.assert_allclose(
        py_result["abs_forecast_error"], cli_result["abs_forecast_error"], **CLI_TEXT_TOL
    )
    np.testing.assert_allclose(
        py_result["avg_neighborhood_size"], cli_result["avg_neighborhood_size"], **CLI_TEXT_TOL
    )
    np.testing.assert_allclose(
        py_result["avg_num_neighbors"], cli_result["avg_num_neighbors"], **CLI_TEXT_TOL
    )
    np.testing.assert_allclose(
        py_result["ky_dimension"], cli_result["ky_dimension"], **CLI_TEXT_TOL
    )


# label, cli_args, columns, kwargs, length, exclude
CASES = [
    ("default_dim1_embed2", ["-k10", "-n30"], [1], dict(minneighbors=10, iterations=30), None, 0),
    ("m1_3_k20_n50", ["-m1,3", "-k20", "-n50"], [1],
     dict(embed=3, minneighbors=20, iterations=50), None, 0),
    ("m2_2_k15_n20", ["-m2,2", "-k15", "-n20"], [1, 2],
     dict(embed=2, minneighbors=15, iterations=20), None, 0),
    ("m1_2_r_and_f", ["-m1,2", "-r0.01", "-f1.5", "-k12", "-n25"], [1],
     dict(embed=2, epsmin=0.01, epsset=True, epsstep=1.5, minneighbors=12, iterations=25),
     None, 0),
    ("inverse", ["-m1,3", "-k20", "-n30", "-I"], [1],
     dict(embed=3, minneighbors=20, iterations=30, inverse=True), None, 0),
    ("column_2", ["-c2", "-m1,2", "-k15", "-n20"], [2],
     dict(embed=2, minneighbors=15, iterations=20), None, 0),
    ("verbosity_0", ["-V0", "-m1,3", "-k20", "-n50"], [1],
     dict(embed=3, minneighbors=20, iterations=50), None, 0),
    ("length_and_exclude", ["-m1,2", "-l500", "-x100", "-k15", "-n20"], [1],
     dict(embed=2, minneighbors=15, iterations=20), 500, 100),
]


@pytest.mark.parametrize(
    "label,cli_args,columns,kwargs,length,exclude", CASES, ids=[c[0] for c in CASES]
)
def test_compute_matches_cli(label, cli_args, columns, kwargs, length, exclude):
    out = run_cli(cli_args + [LORENZ])
    cli_result = parse_output(out)

    py_result = run_py_compute(LORENZ, columns, length=length, exclude=exclude, kwargs=kwargs)

    assert len(py_result["exponents"]) == kwargs.get("embed", 2) * len(columns)
    assert_matches_cli(py_result, cli_result)


def test_compute_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.lyaps"
    run_cli(["-m1,3", "-k20", "-n50", "-o" + str(outfile), LORENZ])
    cli_result = parse_output(outfile.read_text())

    py_result = run_py_compute(
        LORENZ, [1], kwargs=dict(embed=3, minneighbors=20, iterations=50)
    )

    assert_matches_cli(py_result, cli_result)


def test_compute_default_kwargs_match_cli_documented_defaults():
    # show_options(): -m default "1,2" (dimension inferred from series
    # shape, embed default 2), -r default "(data interval)/1000" i.e.
    # epsset stays False and epsmin is unused, -f default 1.2, -k default
    # 30, -n default "length" (i.e. unbounded, clamped internally), -I
    # default off. Both calls are their own fresh subprocess (see
    # run_py_compute's docstring), so identical logical settings must
    # reproduce bit-identical output.
    default_result = run_py_compute(LORENZ, [1], length=200)
    explicit_result = run_py_compute(
        LORENZ, [1], length=200,
        kwargs=dict(
            embed=2, iterations=2**64 - 1, epsmin=1.e-3, epsset=False, epsstep=1.2,
            minneighbors=30, inverse=False,
        ),
    )

    assert default_result == explicit_result


def test_compute_is_not_reproducible_across_calls_in_one_process():
    # Documents the RNG-leakage caveat explained in run_py_compute's
    # docstring: rnd_init() is a process-wide, one-shot seed, so a second
    # compute() call in the same process (even with identical arguments)
    # continues whatever RNG state the first call left behind rather than
    # reseeding. This is inherited behavior from the underlying C library,
    # not something this binding changes.
    raw = np.loadtxt(LORENZ)
    series = raw[:, 0].reshape(1, -1)
    kwargs = dict(embed=3, minneighbors=20, iterations=30)

    first = tisean.lyap_spec.compute(series, **kwargs)
    second = tisean.lyap_spec.compute(series, **kwargs)

    assert not np.allclose(first.exponents, second.exponents)


def test_compute_rejects_data_too_short_for_minneighbors_like_cli(tmp_path):
    datafile = tmp_path / "short.txt"
    datafile.write_text("\n".join(str(float(i % 7)) for i in range(15)) + "\n")

    result = subprocess.run(
        [LYAP_SPEC_BIN, "-m1,2", "-k20", str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == LYAP_SPEC_DATA_TOO_SHORT

    series = np.array([[float(i % 7) for i in range(15)]])
    with pytest.raises(ValueError):
        tisean.lyap_spec.compute(series, embed=2, minneighbors=20)


def test_compute_rejects_constant_data_like_cli(tmp_path):
    datafile = tmp_path / "constant.txt"
    datafile.write_text("\n".join(["1.5"] * 50) + "\n")

    result = subprocess.run(
        [LYAP_SPEC_BIN, "-m1,2", "-k5", "-n5", str(datafile)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == RESCALE_DATA_ZERO_INTERVAL

    series = np.full((1, 50), 1.5)
    with pytest.raises(ValueError):
        tisean.lyap_spec.compute(series, embed=2, minneighbors=5, iterations=5)


def test_compute_rejects_non_2d_series():
    with pytest.raises(ValueError):
        tisean.lyap_spec.compute(np.zeros(20))


def test_compute_rejects_embed_below_one():
    series = np.zeros((1, 50))
    with pytest.raises(ValueError):
        tisean.lyap_spec.compute(series, embed=0)


def test_compute_rejects_minneighbors_below_one():
    raw = np.loadtxt(LORENZ)
    series = raw[:200, 0].reshape(1, -1)
    with pytest.raises(ValueError):
        tisean.lyap_spec.compute(series, embed=2, minneighbors=0)
