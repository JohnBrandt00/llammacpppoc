#!/usr/bin/env python3
"""Summarize GGML_MOE_PREFETCH_METRICS log lines.

The C++ side (ggml-backend.cpp) emits cumulative counters as a JSON object on
every Nth expert-slice copy:

    ggml_moe_copy_metrics: {"copy_ops":...,"payload_bytes":...,...}

Because the counters are cumulative from process start, the *last* sample in a
log already holds the full totals for that process. Diffing first-vs-last (the
previous behaviour) silently dropped the first INTERVAL copy ops plus anything
after the final dump. This script reports the cumulative totals directly and
derives per-copy-op averages, which are independent of the dump interval.
"""
import argparse
import json
import re
from pathlib import Path


METRIC_RE = re.compile(r"ggml_moe_copy_metrics:\s*(\{.*\})")

GB = 1024 ** 3
MB = 1024 ** 2


def read_text(path: Path) -> str:
    """Read a log file, tolerating UTF-8/UTF-16 with or without BOM.

    PowerShell's Tee-Object on Windows PowerShell 5.1 writes UTF-16LE, while the
    bash bench script writes UTF-8. Sniff the BOM, then fall back to detecting
    BOM-less UTF-16 by the tell-tale interleaved null bytes.
    """
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le", errors="replace")
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be", errors="replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    # BOM-less UTF-16 shows up as many NUL bytes among ASCII text.
    head = raw[:4096]
    if head and head.count(0) > len(head) // 4:
        order = "utf-16-le" if raw[1:2] == b"\x00" else "utf-16-be"
        return raw.decode(order, errors="replace")
    return raw.decode("utf-8", errors="replace")


def iter_metrics(path: Path):
    for line in read_text(path).splitlines():
        match = METRIC_RE.search(line)
        if match:
            yield json.loads(match.group(1))


def safe_div(num, den):
    return num / den if den else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize GGML_MOE_PREFETCH_METRICS log lines.")
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.logs:
        samples = list(iter_metrics(path))
        if not samples:
            print(f"{path}: no moe metrics")
            continue

        # Counters are cumulative; the last sample is the running total.
        last = samples[-1]
        copy_ops = last["copy_ops"]
        copy_groups = last["copy_groups"]
        expert_slots = last["expert_slots"]
        payload = last["payload_bytes"]
        scheduled = last["scheduled_bytes"]
        wait_us = last["wait_us"]
        schedule_us = last["schedule_us"]
        ids_us = last["ids_us"]

        print(path)
        print(f"  samples:            {len(samples)}")
        print(f"  copy ops:           {copy_ops}")
        print(f"  copy groups:        {copy_groups}")
        print(f"  expert slots:       {expert_slots}")
        print(f"  payload GB:         {payload / GB:.3f}")
        print(f"  scheduled GB:       {scheduled / GB:.3f}  (incl. padding)")
        print(f"  padding overhead:   {safe_div(scheduled - payload, payload) * 100:.2f}%")
        print(f"  schedule ms:        {schedule_us / 1000.0:.3f}")
        print(f"  ids read ms:        {ids_us / 1000.0:.3f}")
        print(f"  copy wait ms:       {wait_us / 1000.0:.3f}")
        print("  per copy-op:")
        print(f"    payload MB:       {safe_div(payload, copy_ops) / MB:.3f}")
        print(f"    expert slots:     {safe_div(expert_slots, copy_ops):.2f}")
        print(f"    copy groups:      {safe_div(copy_groups, copy_ops):.2f}")
        print(f"    schedule us:      {safe_div(schedule_us, copy_ops):.2f}")
        print(f"    wait us:          {safe_div(wait_us, copy_ops):.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
