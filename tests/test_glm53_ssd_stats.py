#!/usr/bin/env python3
"""Check that SSD serving metadata describes the admitted cache, not preflight."""

import argparse
import json
from pathlib import Path
import re
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stats-file", type=Path)
    source.add_argument("--url")
    parser.add_argument("--native-log", type=Path, required=True)
    args = parser.parse_args()

    if args.stats_file:
        stats = json.loads(args.stats_file.read_text())
    else:
        with urllib.request.urlopen(args.url.rstrip("/") + "/v1/stats", timeout=10) as response:
            stats = json.load(response)

    matches = re.findall(r"SSD admission: .*effective=(\d+) experts (\d+) bytes",
                         args.native_log.read_text())
    assert len(matches) == 1, matches
    count, size = map(int, matches[0])
    effective = stats["serving"]["effective"]
    actual = (effective["ssd_streaming_cache_experts"], effective["ssd_streaming_cache_bytes"])
    assert actual == (count, size), f"reported {actual}, admitted {(count, size)}"
    assert stats["serving"]["quote"]["expert_cache"] == size
    print(json.dumps(dict(status="PASS", experts=count, cache_bytes=size)))


if __name__ == "__main__":
    main()
