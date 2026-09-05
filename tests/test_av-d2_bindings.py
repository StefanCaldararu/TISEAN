import os
import subprocess

import numpy as np
import pytest

import tisean

# The CLI prints with "%e" (6 digits after the point, ~7 significant
# digits), so comparisons against numbers parsed from its stdout can't be
# tighter than that text roundtrip allows.
CLI_TEXT_TOL = dict(rtol=1e-6, atol=1e-6)

D2_REF = "tests/refs/d2_ref.d2"
AV_D2_BIN = os.path.abspath("./bin/av-d2")

# Number of data points in each dimension-block of D2_REF, in file order.
D2_REF_BLOCK_SIZES = {1: 99, 2: 95, 3: 68}


def run_cli(args, **kwargs):
    result = subprocess.run(
        [AV_D2_BIN] + args,
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )
    return result.stdout


def parse_blocks(text):
    """Splits av-d2's stdout into per-dim blocks of (eps, y) rows, matching
    av-d2.c's blank-line block separators. Blocks with zero data rows (the
    averaging window didn't fit in the block) are dropped, same as they'd
    be invisible in a naive line-by-line reading of the output."""
    blocks = []
    current = []
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                blocks.append(np.array(current))
                current = []
            continue
        parts = line.split()
        if len(parts) == 2:
            current.append([float(parts[0]), float(parts[1])])
    if current:
        blocks.append(np.array(current))
    return blocks


def load_blocks(path, rescaled=False):
    """Replicates av-d2.c's own file-format parsing: scans for '#...m= N'
    header lines (matches '#dim= N', since 'dim= ' contains the substring
    'm= ', exactly like the C code's strstr(instr,"m= ") check), then reads
    data lines up to the next blank line. Returns {dim: (eps, y)}.

    If rescaled, replicates form2="%*lf%lf%lf": the first column is
    ignored, the second column is y, the third is eps (mirrors the -E
    flag)."""
    blocks = {}
    with open(path) as f:
        lines = f.readlines()

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("#") and "m= " in line:
            dim = int(line.split()[-1])
            i += 1
            eps_list, y_list = [], []
            while i < n and lines[i].strip() != "":
                if not lines[i].startswith("#"):
                    parts = lines[i].split()
                    if not rescaled:
                        e, yv = float(parts[0]), float(parts[1])
                    else:
                        yv, e = float(parts[1]), float(parts[2])
                    eps_list.append(e)
                    y_list.append(yv)
                i += 1
            blocks[dim] = (np.array(eps_list), np.array(y_list))
        else:
            i += 1
    return blocks


REF_BLOCKS = load_blocks(D2_REF)


CASES = [
    # label, args, dims included (in file order), aver
    ("defaults", [], [1, 2, 3], 1),
    ("m2", ["-m2"], [2, 3], 1),
    ("M2", ["-M2"], [1, 2], 1),
    ("m2_M2", ["-m2", "-M2"], [2], 1),
    ("aver0", ["-a0"], [1, 2, 3], 0),
    ("aver10", ["-a10"], [1, 2, 3], 10),
    ("verbosity0", ["-V0"], [1, 2, 3], 1),
]


@pytest.mark.parametrize("label,args,dims,aver", CASES, ids=[c[0] for c in CASES])
def test_average_matches_cli(label, args, dims, aver):
    out = run_cli(args + [D2_REF])
    cli_blocks = parse_blocks(out)
    assert len(cli_blocks) == len(dims)

    for dim, cli_block in zip(dims, cli_blocks):
        eps, y = REF_BLOCKS[dim]
        result = tisean.av_d2.average(eps, y, aver=aver)

        assert result.n_points == len(eps) - 2 * aver
        np.testing.assert_allclose(result.avg_eps, cli_block[:, 0], **CLI_TEXT_TOL)
        np.testing.assert_allclose(result.avg_y, cli_block[:, 1], **CLI_TEXT_TOL)


def test_average_matches_cli_dash_o_output_file(tmp_path):
    outfile = tmp_path / "out.av"
    run_cli(["-m1", "-M1", "-a2", "-o" + str(outfile), D2_REF])

    cli_blocks = parse_blocks(outfile.read_text())
    assert len(cli_blocks) == 1

    eps, y = REF_BLOCKS[1]
    result = tisean.av_d2.average(eps, y, aver=2)

    np.testing.assert_allclose(result.avg_eps, cli_blocks[0][:, 0], **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.avg_y, cli_blocks[0][:, 1], **CLI_TEXT_TOL)


def test_average_matches_cli_dash_e_rescaled_flag(tmp_path):
    datafile = tmp_path / "rescaled.d2"
    rows = [(i * 0.37, i * 0.11 + 1.0, i * 0.53 + 2.0) for i in range(1, 21)]
    lines = ["#dim= 1"]
    lines += [f"{c1:.17g} {c2:.17g} {c3:.17g}" for c1, c2, c3 in rows]
    lines.append("")
    datafile.write_text("\n".join(lines) + "\n")

    out = run_cli(["-E", "-a1", str(datafile)])
    cli_blocks = parse_blocks(out)
    assert len(cli_blocks) == 1

    eps = np.array([c3 for _, _, c3 in rows])
    y = np.array([c2 for _, c2, _ in rows])
    result = tisean.av_d2.average(eps, y, aver=1)

    np.testing.assert_allclose(result.avg_eps, cli_blocks[0][:, 0], **CLI_TEXT_TOL)
    np.testing.assert_allclose(result.avg_y, cli_blocks[0][:, 1], **CLI_TEXT_TOL)


def test_average_default_aver_is_one():
    eps, y = REF_BLOCKS[1]

    default = tisean.av_d2.average(eps, y)
    explicit = tisean.av_d2.average(eps, y, aver=1)

    assert default.n_points == explicit.n_points
    np.testing.assert_array_equal(default.avg_eps, explicit.avg_eps)
    np.testing.assert_array_equal(default.avg_y, explicit.avg_y)


def test_average_window_too_large_yields_no_points_without_crash():
    # dim 3 has 68 points; a window of 2*40+1=81 doesn't fit anywhere in
    # it. The CLI's own loop bound (k < howmany-aver, computed via
    # unsigned arithmetic) would silently underflow and read far out of
    # bounds for input this small; the reentrant core guards against that
    # explicitly and just reports zero points instead (see include/av-d2.h).
    out = run_cli(["-m3", "-M3", "-a40", D2_REF])
    assert parse_blocks(out) == []

    eps, y = REF_BLOCKS[3]
    result = tisean.av_d2.average(eps, y, aver=40)
    assert result.n_points == 0
    assert len(result.avg_eps) == 0
    assert len(result.avg_y) == 0


def test_cli_rejects_missing_datafile():
    result = subprocess.run(
        [AV_D2_BIN, "-m1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 127
    assert "datafile" in result.stderr.lower()


def test_average_rejects_negative_aver():
    eps, y = REF_BLOCKS[1]
    with pytest.raises(ValueError):
        tisean.av_d2.average(eps, y, aver=-1)


def test_average_rejects_mismatched_lengths():
    eps, y = REF_BLOCKS[1]
    with pytest.raises(ValueError):
        tisean.av_d2.average(eps, y[:-1])


def test_average_rejects_non_1d_arrays():
    eps, y = REF_BLOCKS[1]
    with pytest.raises(ValueError):
        tisean.av_d2.average(eps.reshape(-1, 1), y.reshape(-1, 1))
