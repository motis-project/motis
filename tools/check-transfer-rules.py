#!/usr/bin/env python3

"""Check that the journeys motis returned obey their feeds' transfer rules.

Reads the rules straight out of the feeds and judges every transfer in a
response dump (`motis batch -r`) against them (see transfer_rules.py). Unlike
a comparison of two motis builds, this also finds mistakes both builds share.
validate.py runs the same check in CI.

A build without transfer rule support is a positive control: it must report
violations. If it comes out clean, the checker is broken, not the timetable.

Usage:
  # one GTFS feed (zip or unpacked directory)
  check-transfer-rules.py --gtfs ch.gtfs.zip --tag ch_ responses.json

  # many GTFS feeds, resolved through a motis config's dataset tags
  check-transfer-rules.py --config config_europe.yml --feed-dir feeds/ resp.json

Exits non-zero if any violation was found.
"""

import argparse
import os
import sys

from transfer_rules import Checker, format_violation


def main():
    ap = argparse.ArgumentParser(
        description="check motis journeys against their feeds' transfer rules")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gtfs", metavar="PATH",
                     help="one GTFS feed, zip or unpacked directory")
    src.add_argument("--config", metavar="PATH",
                     help="motis config, to resolve many feeds by dataset tag")
    ap.add_argument("--feed-dir", metavar="DIR",
                    help="where --config's dataset paths are rooted "
                         "(default: the config's directory)")
    ap.add_argument("--tag", default="",
                    help="with --gtfs: the id prefix motis gave the dataset, "
                         "e.g. 'ch_'")
    ap.add_argument("--max-shown", type=int, default=10, metavar="N",
                    help="violations to list per file (default 10)")
    ap.add_argument("responses", nargs="+",
                    help="response dumps written by `motis batch -r`")
    args = ap.parse_args()

    if args.gtfs:
        checker = Checker({args.tag: args.gtfs})
    else:
        checker = Checker.from_config(
            args.config,
            args.feed_dir or os.path.dirname(os.path.abspath(args.config)))

    total = 0
    for path in args.responses:
        with open(path) as f:
            summary, violations = checker.check(f.read().splitlines())
        verdict = f"{len(violations)} VIOLATIONS" if violations else "OK"
        print(f"{os.path.basename(path)}: {summary}, {verdict}")
        for v in violations[:args.max_shown]:
            print(f"    {format_violation(v)}")
        if len(violations) > args.max_shown:
            print(f"    ... and {len(violations) - args.max_shown} more")
        total += len(violations)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
