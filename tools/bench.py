#!/usr/bin/env python3
"""Run the built examples of one or more topics and print Markdown results tables.

Every example prints the same lines (see common/lab.cuh), for example:

    GPU: NVIDIA H100 80GB HBM3 (sm_90, 132 SMs, 3352 GB/s peak DRAM bandwidth)
    6. Warptiling                      1.234 ms |   1740.2 GFLOP/s
    cuBLAS (cublasSgemm)               0.456 ms |   4711.0 GFLOP/s
    Speed relative to cuBLAS            37.0 %
    PASS

This script runs each binary under build/bin/<topic>/, parses those lines, and
writes one table per topic, ready to paste into the topic's README.

Usage:
    python3 tools/bench.py                      # matmul, reduction, matrix_transpose
    python3 tools/bench.py --topics scan sort   # any topics
    python3 tools/bench.py --out results.md     # also write the tables to a file
    python3 tools/bench.py --quick              # small sizes, for a smoke test
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

GPU_RE = re.compile(r"^GPU: (?P<name>.+?) \(sm_(?P<sm>\d+), (?P<sms>\d+) SMs")
PERF_RE = re.compile(
    r"^(?P<label>.+?)\s+(?P<ms>[\d.]+) ms"
    r"(?: \|\s+(?P<gflops>[\d.]+) GFLOP/s)?"
    r"(?: \|\s+(?P<gbs>[\d.]+) GB/s(?: \((?P<peak>\d+)% of peak\))?)?\s*$"
)
CUBLAS_RE = re.compile(r"^Speed relative to cuBLAS\s+(?P<pct>[\d.]+) %")
VERDICT_RE = re.compile(r"^(PASS|FAIL|SKIP)\b")


def parse(output):
    """Returns (gpu, rows): rows are dicts for each performance line."""
    gpu, rows, verdict, vs_cublas = None, [], "?", None
    for line in output.splitlines():
        if m := GPU_RE.match(line):
            gpu = f"{m['name']} (sm_{m['sm']}, {m['sms']} SMs)"
        elif m := CUBLAS_RE.match(line):
            vs_cublas = float(m["pct"])
        elif m := VERDICT_RE.match(line):
            verdict = m[1]
        elif m := PERF_RE.match(line):
            rows.append({k: v for k, v in m.groupdict().items() if v is not None})
    return gpu, rows, verdict, vs_cublas


def run_topic(binaries, quick):
    """Runs each binary; returns (gpu, table rows)."""
    gpu, table = None, []
    for binary in binaries:
        args = [str(binary)] + (["--quick"] if quick else [])
        print(f"running {binary.name} ...", file=sys.stderr)
        proc = subprocess.run(args, capture_output=True, text=True)
        g, rows, verdict, vs_cublas = parse(proc.stdout)
        gpu = gpu or g
        if verdict == "?":
            verdict = "SKIP" if proc.returncode == 77 else ("PASS" if proc.returncode == 0 else "FAIL")
        # The first performance line is the example itself; others are baselines.
        main = rows[0] if rows else {"label": binary.name}
        table.append({"name": binary.name, "verdict": verdict, "vs_cublas": vs_cublas, **main})
    return gpu, table


def markdown(topic, gpu, table):
    has_flops = any("gflops" in r for r in table)
    has_cublas = any(r.get("vs_cublas") is not None for r in table)
    has_bw = any("gbs" in r for r in table)
    header = ["Step", "Time (ms)"]
    if has_flops:
        header.append("GFLOP/s")
    if has_cublas:
        header.append("% of cuBLAS")
    if has_bw:
        header += ["GB/s", "% of peak"]
    header.append("Check")
    lines = [f"### {topic}", "", f"GPU: {gpu or 'unknown'}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for r in table:
        cells = [f"`{r['name']}`", r.get("ms", "-")]
        if has_flops:
            cells.append(r.get("gflops", "-"))
        if has_cublas:
            cells.append(f"{r['vs_cublas']:.0f}%" if r.get("vs_cublas") is not None else "-")
        if has_bw:
            cells += [r.get("gbs", "-"), f"{r['peak']}%" if "peak" in r else "-"]
        cells.append(r["verdict"])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", default="build", help="CMake build directory (default: build)")
    parser.add_argument("--topics", nargs="+", default=["matmul", "reduction", "matrix_transpose"])
    parser.add_argument("--out", help="also write the Markdown to this file")
    parser.add_argument("--quick", action="store_true", help="pass --quick to every example")
    args = parser.parse_args()

    sections = []
    for topic in args.topics:
        bin_dir = Path(args.build) / "bin" / topic
        binaries = sorted(p for p in bin_dir.glob("*") if p.is_file())
        if not binaries:
            sys.exit(f"no binaries in {bin_dir}; build first with `make`")
        gpu, table = run_topic(binaries, args.quick)
        sections.append(markdown(topic, gpu, table))

    text = "\n".join(sections)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
