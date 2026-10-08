#!/usr/bin/env python3
"""Roofline plot for the GEMM ladder (matmul/).

A roofline shows each kernel's achieved throughput (GFLOP/s) against its
arithmetic intensity (FLOPs per byte of DRAM traffic). Two ceilings bound what
is possible: the memory roof (peak bandwidth x intensity) and the compute roof
(peak FLOP/s). A kernel near the slanted memory roof is bandwidth-bound; one
near the flat compute roof is compute-bound.

What it does:
- Reads the GPU's SM count, SM clock, compute capability, and peak DRAM
  bandwidth from build/bin/basics/03_runtime_api_device_query.
- Runs every binary in build/bin/matmul/ to get its GFLOP/s.
- If Nsight Compute (`ncu`) is on PATH, profiles one launch of each kernel to
  measure its real DRAM traffic (dram__bytes_read.sum + dram__bytes_write.sum).
  Without ncu it falls back to the minimum traffic (read A and B once, read
  and write C once), which puts every kernel at the same intensity.
- Draws the FP32 compute roof (SMs x FP32 lanes per SM x 2 x clock) and the
  memory roof. Tensor Core stages use FP16 inputs and can exceed the FP32 roof;
  pass --tensor-tflops to draw their roof too (it varies by GPU, so it is not
  derived automatically).

Usage:
    python3 tools/roofline.py [--build build] [--out docs/roofline.png]
                              [--size 4096] [--tensor-tflops 989]

Use --size large enough that A, B, and C do not fit in L2 (4096 or more on
GPUs with 40-50 MB of L2, such as A100 and H100); otherwise every kernel
reads DRAM only once and they all land at the same intensity.
Requires matplotlib (pip install matplotlib).
"""
import argparse
import csv
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

# FP32 lanes (CUDA cores) per SM by compute capability.
FP32_LANES_PER_SM = {
    70: 64, 72: 64, 75: 64, 80: 64, 86: 128, 87: 128, 89: 128,
    90: 128, 100: 128, 101: 128, 120: 128,
}

# Reference categorical palette (validated: CVD and normal-vision separation,
# >= 3:1 contrast on the light surface), plus text inks.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_FP32 = "#2a78d6"    # slot 1, blue
SERIES_TENSOR = "#eb6834"  # slot 2, orange


def device_info(build):
    binary = Path(build) / "bin" / "basics" / "03_runtime_api_device_query"
    out = subprocess.run([str(binary)], capture_output=True, text=True, check=True).stdout

    def field(label, cast=float):
        m = re.search(rf"^\s*{re.escape(label)}:\s*([\d.]+)", out, re.M)
        if not m:
            sys.exit(f"could not find '{label}' in device query output")
        return cast(m[1])

    name = re.search(r"^Device 0: (.+)$", out, re.M)[1].strip()
    cc = field("Compute capability")
    return {
        "name": name,
        "cc": int(round(cc * 10)),
        "sms": field("Multiprocessors (SMs)", int),
        "clock_ghz": field("SM clock") / 1000.0,
        "peak_gbs": field("Peak DRAM bandwidth"),
    }


def problem_size(output):
    m = re.search(r"M=(\d+), N=(\d+), K=(\d+)", output)
    return tuple(int(v) for v in m.groups()) if m else (1024, 1024, 1024)


def ncu_dram_bytes(binary, extra_args):
    """DRAM bytes moved by one launch of the stage's own kernel, or None."""
    if not shutil.which("ncu"):
        return None
    cmd = ["ncu", "--csv", "--print-units", "base", "--launch-count", "1",
           "--kernel-name", "regex:^[sh]gemm_",
           "--metrics", "dram__bytes_read.sum,dram__bytes_write.sum",
           str(binary)] + extra_args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    lines = proc.stdout.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        return None
    total = 0.0
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        if row.get("Metric Name", "").startswith("dram__bytes_"):
            total += float(row["Metric Value"].replace(",", ""))
    return total or None


def measure(build, extra_args):
    points = []
    for binary in sorted((Path(build) / "bin" / "matmul").glob("*")):
        print(f"running {binary.name} ...", file=sys.stderr)
        proc = subprocess.run([str(binary)] + extra_args, capture_output=True, text=True)
        _, rows, verdict, _ = bench.parse(proc.stdout)
        if verdict != "PASS" or not rows or "gflops" not in rows[0]:
            print(f"  skipped ({verdict})", file=sys.stderr)
            continue
        m, n, k = problem_size(proc.stdout)
        flops = 2.0 * m * n * k
        tensor = any(t in binary.name for t in ("wmma", "mma_sync", "wgmma"))
        in_bytes = 2 if tensor else 4
        min_bytes = in_bytes * (m * k + k * n) + 4 * 2 * m * n
        dram = ncu_dram_bytes(binary, extra_args)
        points.append({
            "step": binary.name.split("_")[0],
            "name": binary.name,
            "gflops": float(rows[0]["gflops"]),
            "intensity": flops / (dram or min_bytes),
            "measured": dram is not None,
            "tensor": tensor,
        })
    return points


def label_points(ax, points, dpi):
    """Labels each point with its step number, choosing among four positions
    around the marker the one farthest from every other marker and every label
    already placed, so clustered points stay readable."""
    import numpy as np
    to_px = ax.transData.transform
    markers = [to_px((p["intensity"], p["gflops"])) for p in points]
    scale = dpi / 72.0  # Offsets are in points; transforms are in pixels.
    candidates = [(7, 5), (7, -13), (-7, 5), (-7, -13)]
    placed = []
    for i, p in enumerate(sorted(points, key=lambda q: q["intensity"])):
        here = to_px((p["intensity"], p["gflops"]))
        others = [m for m in markers if not np.allclose(m, here)] + placed
        def clearance(offset):
            center = here + np.array([offset[0] + (6 if offset[0] > 0 else -6), offset[1] + 4]) * scale
            return min((np.hypot(*(center - o)) for o in others), default=1e9)
        best = max(candidates, key=clearance)
        placed.append(here + np.array([best[0] + (6 if best[0] > 0 else -6), best[1] + 4]) * scale)
        ax.annotate(p["step"], (p["intensity"], p["gflops"]), xytext=best,
                    textcoords="offset points", ha="left" if best[0] > 0 else "right",
                    color=TEXT_PRIMARY, fontsize=9, zorder=4)


def plot(points, dev, out, tensor_tflops):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lanes = FP32_LANES_PER_SM.get(dev["cc"], 128)
    fp32_peak = dev["sms"] * lanes * 2 * dev["clock_ghz"]  # GFLOP/s
    bw = dev["peak_gbs"]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")

    xs = [p["intensity"] for p in points] or [1.0]
    x_lo, x_hi = min(xs) / 4, max(xs) * 4
    roofs = [("FP32 compute roof", fp32_peak, "-")]
    if tensor_tflops:
        roofs.append(("FP16 Tensor Core roof", tensor_tflops * 1000.0, "--"))
    x_hi = max(x_hi, max(peak / bw for _, peak, _ in roofs) * 2)
    for label, peak, style in roofs:
        ridge = peak / bw
        ax.plot([x_lo, ridge], [bw * x_lo, peak], color=TEXT_SECONDARY, lw=2, ls=style)
        ax.plot([ridge, x_hi], [peak, peak], color=TEXT_SECONDARY, lw=2, ls=style)
        ax.annotate(f"{label}: {peak / 1000:.1f} TFLOP/s", (x_hi, peak), xytext=(-4, 5),
                    textcoords="offset points", ha="right", va="bottom",
                    color=TEXT_SECONDARY, fontsize=9)
    ax.annotate(f"Memory roof: {bw:.0f} GB/s", (x_lo * 1.2, bw * x_lo * 1.2),
                xytext=(4, 2), textcoords="offset points", rotation=0,
                color=TEXT_SECONDARY, fontsize=9)

    for tensor, color, marker, label in ((False, SERIES_FP32, "o", "FP32 (CUDA cores)"),
                                         (True, SERIES_TENSOR, "s", "FP16 inputs (Tensor Cores)")):
        group = [p for p in points if p["tensor"] == tensor]
        if not group:
            continue
        ax.scatter([p["intensity"] for p in group], [p["gflops"] for p in group],
                   s=70, color=color, marker=marker, edgecolors=SURFACE, linewidths=2,
                   label=label, zorder=3)

    ax.set_xlim(x_lo, x_hi)
    label_points(ax, points, fig.dpi)

    measured = all(p["measured"] for p in points) and points
    ax.set_xlabel("Arithmetic intensity (FLOP per byte of DRAM traffic"
                  + (", measured with ncu)" if measured else ", minimum traffic)"),
                  color=TEXT_PRIMARY)
    ax.set_ylabel("Achieved GFLOP/s", color=TEXT_PRIMARY)
    ax.set_title(f"GEMM ladder roofline: {dev['name']}", color=TEXT_PRIMARY, loc="left")
    ax.grid(True, which="major", color=GRID, lw=1)
    ax.tick_params(colors=TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", default="build")
    parser.add_argument("--out", default="docs/roofline.png")
    parser.add_argument("--size", type=int, default=None,
                        help="run every GEMM as size x size x size (default: each binary's default)")
    parser.add_argument("--tensor-tflops", type=float, default=None,
                        help="dense FP16 Tensor Core peak of this GPU, to draw its roof")
    args = parser.parse_args()
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        sys.exit("matplotlib is required: pip install matplotlib")
    dev = device_info(args.build)
    extra_args = ["--m", str(args.size), "--n", str(args.size), "--k", str(args.size)] if args.size else []
    points = measure(args.build, extra_args)
    # Also print the data as a table, so the plot is never the only view.
    print("| Step | GFLOP/s | FLOP/byte | Traffic |")
    print("|---|---|---|---|")
    for p in points:
        print(f"| `{p['name']}` | {p['gflops']:.1f} | {p['intensity']:.1f} | "
              f"{'measured' if p['measured'] else 'minimum'} |")
    plot(points, dev, args.out, args.tensor_tflops)


if __name__ == "__main__":
    main()
