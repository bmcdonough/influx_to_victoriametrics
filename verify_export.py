#!/usr/bin/env python3
"""
Verify that data was successfully exported from InfluxDB to VictoriaMetrics.

Checks:
  1. Every InfluxDB series exists in VictoriaMetrics
  2. Point counts match (within tolerance)
  3. Last value matches for a random sample of series

InfluxDB connection uses the same env vars as influx_export.py.
VictoriaMetrics address is read from VM_ADDR env var or --vm-addr arg.
"""

import os
import re
import sys
import random
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from influxdb_client import InfluxDBClient
from influxdb_client.client.warnings import MissingPivotFunction

warnings.simplefilter("ignore", MissingPivotFunction)

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env")
except ImportError:
    pass

# Counts within this fraction are considered matching (e.g. 0.05 = 5% tolerance).
COUNT_TOLERANCE = 0.05
# Number of series to spot-check for last-value equality.
SPOT_CHECK_SAMPLE = 5


def sanitize_name(name: str) -> str:
    """Replace non-alphanumeric/underscore characters with underscores (Prometheus naming)."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def vm_metric_name(measurement: str, field: str) -> str:
    # VictoriaMetrics stores the metric name as "{measurement}_{field}" verbatim
    # (it does not sanitize spaces or special chars from the line protocol).
    return f"{measurement}_{field}"


def get_tag_cols(df: pd.DataFrame) -> List[str]:
    return [
        k
        for k in df.columns
        if not k.startswith("_") and k not in ("result", "table")
    ]


def discover_series(query_api, bucket: str) -> List[Tuple[str, str, Dict[str, str]]]:
    """Return list of (measurement, field, tags_dict) for all series in the bucket."""
    query = f"""
    from(bucket: "{bucket}")
    |> range(start: 0, stop: now())
    |> first()
    """
    raw = query_api.query_data_frame(query)
    frames: List[pd.DataFrame] = raw if isinstance(raw, list) else [raw]

    series = []
    for df in frames:
        if df.empty:
            continue
        tag_cols = get_tag_cols(df)
        for (meas, field), group in df.groupby(["_measurement", "_field"]):
            # Collect tag values from the first row (tags are constant per group).
            row = group.iloc[0]
            tags = {col: str(row[col]) for col in tag_cols if pd.notna(row[col])}
            series.append((meas, field, tags))
    return series


def influx_count(query_api, bucket: str, measurement: str, field: str) -> int:
    query = f"""
    from(bucket: "{bucket}")
    |> range(start: 0, stop: now())
    |> filter(fn: (r) => r["_measurement"] == "{measurement}")
    |> filter(fn: (r) => r["_field"] == "{field}")
    |> count()
    """
    raw = query_api.query_data_frame(query)
    frames = raw if isinstance(raw, list) else [raw]
    total = 0
    for df in frames:
        if not df.empty and "_value" in df.columns:
            total += int(df["_value"].sum())
    return total


def influx_last(
    query_api, bucket: str, measurement: str, field: str
) -> Tuple[Optional[float], Optional[int]]:
    """Return (value, timestamp_ns) of the last data point."""
    query = f"""
    from(bucket: "{bucket}")
    |> range(start: 0, stop: now())
    |> filter(fn: (r) => r["_measurement"] == "{measurement}")
    |> filter(fn: (r) => r["_field"] == "{field}")
    |> last()
    """
    raw = query_api.query_data_frame(query)
    frames = raw if isinstance(raw, list) else [raw]
    for df in frames:
        if not df.empty:
            row = df.iloc[-1]
            ts_ns = int(pd.Timestamp(row["_time"]).value)
            return row["_value"], ts_ns
    return None, None


def vm_selector(metric: str, tags: Dict[str, str], bucket: str) -> str:
    """Build a PromQL label selector safe for metric names with special characters."""
    all_tags = {"__name__": metric, "db": bucket, **tags}
    return "{" + ",".join(f'{k}="{v}"' for k, v in all_tags.items()) + "}"


def vm_series_exists(
    vm_addr: str, metric: str, tags: Dict[str, str], bucket: str
) -> bool:
    """Return True if VictoriaMetrics has at least one series matching metric+tags+db."""
    selector = vm_selector(metric, tags, bucket)
    resp = requests.get(
        f"{vm_addr}/api/v1/series",
        params={"match[]": selector, "start": "2000-01-01T00:00:00Z", "end": "2100-01-01T00:00:00Z"},
    )
    resp.raise_for_status()
    return len(resp.json().get("data", [])) > 0


def vm_count(vm_addr: str, metric: str, tags: Dict[str, str], bucket: str) -> int:
    """Approximate data point count in VictoriaMetrics using count_over_time."""
    selector = vm_selector(metric, tags, bucket)
    query = f"count_over_time({selector}[100y])"
    resp = requests.get(
        f"{vm_addr}/api/v1/query",
        params={"query": query, "time": "2100-01-01T00:00:00Z"},
    )
    resp.raise_for_status()
    results = resp.json().get("data", {}).get("result", [])
    return sum(int(r["value"][1]) for r in results)


def vm_last(
    vm_addr: str, metric: str, tags: Dict[str, str], bucket: str
) -> Tuple[Optional[str], Optional[int]]:
    """Return (value_str, actual_sample_timestamp_s) of the last data point in VictoriaMetrics."""
    selector = vm_selector(metric, tags, bucket)
    # Use timestamp() to get the actual sample time, not the query evaluation time.
    resp = requests.get(
        f"{vm_addr}/api/v1/query",
        params={
            "query": f"last_over_time({selector}[100y])",
            "time": "2100-01-01T00:00:00Z",
        },
    )
    resp.raise_for_status()
    results = resp.json().get("data", {}).get("result", [])
    if not results:
        return None, None
    _, val = results[0]["value"]

    # tlast_over_time is a VictoriaMetrics extension that returns the actual
    # timestamp of the last sample (not the query evaluation time).
    ts_resp = requests.get(
        f"{vm_addr}/api/v1/query",
        params={
            "query": f"tlast_over_time({selector}[100y])",
            "time": "2100-01-01T00:00:00Z",
        },
    )
    ts_resp.raise_for_status()
    ts_results = ts_resp.json().get("data", {}).get("result", [])
    if not ts_results:
        return val, None
    _, ts_str = ts_results[0]["value"]
    return val, int(float(ts_str))


def main(bucket: str, vm_addr: str, args: Dict):
    for k, v in args.items():
        if v is not None:
            os.environ[k] = v

    client = InfluxDBClient.from_env_properties()
    query_api = client.query_api()

    print(f"Discovering series in InfluxDB bucket '{bucket}'...")
    series = discover_series(query_api, bucket)
    print(f"Found {len(series)} unique time series\n")

    if not series:
        print("No series found — nothing to verify.")
        sys.exit(1)

    failures = []
    missing = []

    # --- Check 1: series presence + point counts ---
    print("=== Check 1: Series presence and point counts ===")
    for meas, field, tags in series:
        metric = vm_metric_name(meas, field)
        label = f"{meas}.{field}"

        exists = vm_series_exists(vm_addr, metric, tags, bucket)
        if not exists:
            print(f"  MISSING  {label}  (VM metric: {metric})")
            missing.append((meas, field))
            failures.append(label)
            continue

        influx_n = influx_count(query_api, bucket, meas, field)
        vm_n = vm_count(vm_addr, metric, tags, bucket)

        if influx_n == 0:
            print(f"  SKIP     {label}  (InfluxDB count=0)")
            continue

        diff_ratio = abs(influx_n - vm_n) / influx_n
        status = "OK" if diff_ratio <= COUNT_TOLERANCE else "COUNT_MISMATCH"
        print(
            f"  {status:<14} {label}  influx={influx_n}  vm={vm_n}"
            + (f"  diff={diff_ratio:.1%}" if status != "OK" else "")
        )
        if status != "OK":
            failures.append(label)

    # --- Check 2: spot-check last values ---
    print(f"\n=== Check 2: Last-value spot-check (sample of {SPOT_CHECK_SAMPLE}) ===")
    candidates = [(m, f, t) for m, f, t in series if (m, f) not in missing]
    sample = random.sample(candidates, min(SPOT_CHECK_SAMPLE, len(candidates)))

    for meas, field, tags in sample:
        metric = vm_metric_name(meas, field)
        label = f"{meas}.{field}"

        influx_val, influx_ts_ns = influx_last(query_api, bucket, meas, field)
        vm_val, vm_ts_s = vm_last(vm_addr, metric, tags, bucket)

        if influx_val is None or vm_val is None:
            print(f"  SKIP     {label}  (could not retrieve last value)")
            continue

        # Compare timestamps: InfluxDB ns vs VM seconds.
        influx_ts_s = influx_ts_ns // 1_000_000_000
        ts_match = abs(influx_ts_s - vm_ts_s) <= 1  # allow 1s rounding

        # Compare values: numeric only. VictoriaMetrics stores string fields as 0,
        # so skip value comparison when the InfluxDB value is non-numeric.
        try:
            val_match = abs(float(influx_val) - float(vm_val)) < 1e-6
        except (TypeError, ValueError):
            val_match = True  # non-numeric influx value can't round-trip through VM

        if ts_match and val_match:
            print(f"  OK       {label}  last_value={vm_val}  ts={vm_ts_s}")
        else:
            detail = []
            if not val_match:
                detail.append(f"value: influx={influx_val} vm={vm_val}")
            if not ts_match:
                detail.append(f"ts: influx={influx_ts_s} vm={vm_ts_s}")
            print(f"  MISMATCH {label}  " + "  ".join(detail))
            failures.append(label)

    # --- Summary ---
    print(f"\n=== Summary ===")
    total = len(series)
    print(f"Series checked : {total}")
    print(f"Failures       : {len(failures)}")
    if failures:
        print("Failed series  :")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify InfluxDB → VictoriaMetrics export. "
        "InfluxDB settings can be set via env vars or args (same as influx_export.py)."
    )
    parser.add_argument("bucket", type=str, help="InfluxDB source bucket name")
    parser.add_argument(
        "--vm-addr",
        "-a",
        type=str,
        default=os.getenv("VM_ADDR", "http://localhost:8428"),
        help="VictoriaMetrics base URL (default: $VM_ADDR or http://localhost:8428)",
    )
    parser.add_argument("--INFLUXDB_V2_URL", "-u", type=str)
    parser.add_argument("--INFLUXDB_V2_ORG", "-o", type=str)
    parser.add_argument("--INFLUXDB_V2_TOKEN", "-t", type=str)
    parser.add_argument("--INFLUXDB_V2_SSL_CA_CERT", "-S", type=str)
    parser.add_argument("--INFLUXDB_V2_TIMEOUT", "-T", type=str)
    parser.add_argument("--INFLUXDB_V2_VERIFY_SSL", "-V", type=str)

    args = vars(parser.parse_args())
    bucket = args.pop("bucket")
    vm_addr = args.pop("vm_addr")
    main(bucket, vm_addr, args)
