#!/usr/bin/env python3
"""Reduce a full MOTIS config.yml (e.g. Transitous') to its timetable part.

Drops street routing, OSM, tiles, geocoding and elevators, turns off
osr_footpath (so the import writes tt.bin, not tt_ext.bin), shapes and
railviz, and removes the datasets' realtime feeds so nothing tries to reach
the network at load time. Dataset paths are left untouched: run the import
from the directory the paths are relative to.
"""
import argparse

import yaml

# everything config validation ties to OSM or street routing
DROP_TOP_LEVEL = (
    "street_routing",
    "osm",
    "tiles",
    "geocoding",
    "reverse_geocoding",
    "elevators",
    "gbfs",
    "prima",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", help="full config.yml")
    p.add_argument("dst", help="where to write the stripped config")
    p.add_argument(
        "--gpu-states",
        type=int,
        help="server.gpu_states: concurrent GPU searches motis batch can run",
    )
    p.add_argument(
        "--num-days",
        type=int,
        help="override timetable.num_days (shrinks the import if RAM is short)",
    )
    p.add_argument(
        "--first-day",
        help="override timetable.first_day (a date, for importing an older "
        "GTFS mirror whose feeds no longer cover today)",
    )
    a = p.parse_args()

    with open(a.src) as f:
        c = yaml.safe_load(f)

    for k in DROP_TOP_LEVEL:
        c.pop(k, None)
    c["osr_footpath"] = False
    c.get("server", {}).pop("web_folder", None)
    if a.gpu_states:
        c.setdefault("server", {})["gpu_states"] = a.gpu_states

    tt = c["timetable"]
    tt["with_shapes"] = False
    tt["railviz"] = False
    if a.num_days:
        tt["num_days"] = a.num_days
    if a.first_day:
        tt["first_day"] = a.first_day
    for d in tt["datasets"].values():
        d.pop("rt", None)

    with open(a.dst, "w") as f:
        yaml.safe_dump(c, f, sort_keys=False, allow_unicode=True, width=1 << 16)
    print(f"{a.dst}: {len(tt['datasets'])} datasets")


if __name__ == "__main__":
    main()
