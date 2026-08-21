#!/usr/bin/env python3

"""Check that the journeys motis returned obey the feed's own transfer rules.

Comparing two motis builds against each other only proves they agree - it
cannot find a bug both of them have. This reads the rules straight out of the
feed and judges every transfer in a response dump against them, so it sees
mistakes no differential test can.

Always run a build without transfer rule support as a positive control: it
must report violations. If it comes out clean the checker is broken, not the
timetable.

Usage:
  # one GTFS feed (zip or unpacked directory)
  check-transfer-rules.py --gtfs ch.gtfs.zip --tag ch_ responses.json

  # many GTFS feeds, resolved through a motis config's dataset tags
  check-transfer-rules.py --config config_europe.yml --feed-dir feeds/ resp.json

  # HRDF (the stamm/ directory holding umsteigb.txt and friends)
  check-transfer-rules.py --hrdf feed/stamm --tag de_ responses.json

Exits non-zero if any violation was found.
"""

import argparse
import csv
import io
import json
import os
import sys
import zipfile
from collections import defaultdict
from datetime import datetime


# --- reading a feed, zipped or not ------------------------------------------

class Feed:
    """A GTFS feed's tables, read on demand from a zip or a directory."""

    def __init__(self, path):
        self.path = path
        self.zip = zipfile.ZipFile(path) if zipfile.is_zipfile(path) else None

    def rows(self, name):
        """Yield the rows of one table; a missing table is empty, not an error -
        a feed without transfers.txt simply states no rules."""
        try:
            if self.zip is not None:
                with self.zip.open(name) as f:
                    yield from csv.DictReader(
                        io.TextIOWrapper(f, "utf-8-sig", newline=""))
            else:
                p = os.path.join(self.path, name)
                if not os.path.exists(p):
                    return
                with open(p, encoding="utf-8-sig", newline="") as f:
                    yield from csv.DictReader(f)
        except KeyError:
            return


# --- GTFS transfers.txt ------------------------------------------------------

# Specificity ladder from the GTFS reference, least specific first. Mirrors
# `specificity` in nigiri's loader/gtfs/transfer_rules.cc - the checker has to
# pick the same winning rule the loader did, or it judges against a rule that
# never applied.
def gtfs_specificity(from_route, to_route, from_trip, to_trip):
    if from_trip and to_trip:
        return 5
    if (from_trip and to_route) or (from_route and to_trip):
        return 4
    if from_trip or to_trip:
        return 3
    if from_route and to_route:
        return 2
    if from_route or to_route:
        return 1
    return 0


class GtfsRules:
    def __init__(self, feed, wanted_stops=None, wanted_trips=None):
        self.by_pair = defaultdict(list)
        self.parent = {}
        self.trip_route = {}
        self.any_rule = False

        for r in feed.rows("transfers.txt"):
            try:
                ty = int((r.get("transfer_type") or "").strip() or "0")
            except ValueError:
                continue
            # 4 (stay seated) and 5 (no stay seated) are not transfer-time
            # rules; anything outside 0..3 is unknown to the loader as well.
            if ty < 0 or ty > 3:
                continue
            mtt = (r.get("min_transfer_time") or "").strip()
            # A recommended transfer with no time is a preference, not a
            # constraint - the loader drops it too.
            if ty == 0 and not mtt:
                continue
            key = ((r.get("from_stop_id") or "").strip(),
                   (r.get("to_stop_id") or "").strip())
            from_route = (r.get("from_route_id") or "").strip()
            to_route = (r.get("to_route_id") or "").strip()
            from_trip = (r.get("from_trip_id") or "").strip()
            to_trip = (r.get("to_trip_id") or "").strip()
            self.by_pair[key].append((
                gtfs_specificity(from_route, to_route, from_trip, to_trip),
                ty, int(mtt) // 60 if mtt else 0,
                from_route, to_route, from_trip, to_trip))
            self.any_rule = True

        if not self.any_rule:
            return

        for r in feed.rows("stops.txt"):
            if wanted_stops is None or r["stop_id"] in wanted_stops:
                self.parent[r["stop_id"]] = \
                    (r.get("parent_station") or "").strip() or None
        for r in feed.rows("trips.txt"):
            if wanted_trips is None or r["trip_id"] in wanted_trips:
                self.trip_route[r["trip_id"]] = r["route_id"]

    def applicable(self, a, b, trip_a, trip_b):
        """The rule that governs arriving at `a` on `trip_a` and leaving `b` on
        `trip_b`, or None. A rule may name a stop or its station, so both are
        tried; ties are broken the way the loader ranks them - the GTFS ladder
        first, then whether the rule named the stops exactly."""
        route_a = self.trip_route.get(trip_a)
        route_b = self.trip_route.get(trip_b)
        best = None
        for from_stop in (a, self.parent.get(a)):
            for to_stop in (b, self.parent.get(b)):
                if from_stop is None or to_stop is None:
                    continue
                for (spec, ty, mins, fr, tr, ft, tt) in \
                        self.by_pair.get((from_stop, to_stop), ()):
                    if fr and fr != route_a:
                        continue
                    if tr and tr != route_b:
                        continue
                    if ft and ft != trip_a:
                        continue
                    if tt and tt != trip_b:
                        continue
                    rank = (spec << 2) | ((from_stop == a) + (to_stop == b))
                    if best is None or rank > best[0]:
                        best = (rank, ty, mins)
        return best


# --- HRDF umsteig* -----------------------------------------------------------

class HrdfRules:
    """Station transfer times from umsteigb.txt, plus the set of stations that
    carry a pair rule (umsteigv/umsteigl/umsteigz).

    HAFAS precedence (HRDF 5.40.41 ch. 8) is: trip pair at the station, line
    pair at the station, admin pair at the station, the station transfer time,
    then the global line and admin rules, then the default. So for a station
    that has an umsteigb entry and no pair rule of its own, the ladder stops at
    the station transfer time and no global rule can undercut it - those
    transfers are decided by one number and can be checked without recovering
    a trip's admin, category and line from the response. Everything else is
    reported as out of scope rather than counted as a pass."""

    def __init__(self, stamm):
        self.station_time = {}
        self.default_time = None
        self.pair_rule_stations = set()

        for line in self._lines(stamm, "umsteigb.txt"):
            if len(line) < 13:
                continue
            eva = self._eva(line[0:7])
            if eva is None:
                continue
            try:
                minutes = int(line[11:13].strip())
            except ValueError:
                continue
            if eva == 9999999:
                self.default_time = minutes
            else:
                self.station_time[eva] = minutes

        for name in ("umsteigv.txt", "umsteigl.txt", "umsteigz_vt.txt",
                     "umsteigz.txt"):
            for line in self._lines(stamm, name):
                eva = self._eva(line[0:7])
                if eva is not None:
                    self.pair_rule_stations.add(eva)

    @staticmethod
    def _lines(stamm, name):
        path = os.path.join(stamm, name)
        if not os.path.exists(path):
            return
        with open(path, encoding="latin-1") as f:
            for line in f:
                line = line.rstrip("\n")
                # '%' and '*' start a comment; the global rules use '@' as the
                # station column, which _eva rejects.
                if line and line[0] not in "%*":
                    yield line

    @staticmethod
    def _eva(s):
        s = s.strip()
        if not s or s.startswith("@"):
            return None
        try:
            return int(s)
        except ValueError:
            return None


# --- walking the response ----------------------------------------------------

def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def transfers(path):
    """Yield (query index, arrival leg, departure leg) for every transfer
    between two transit legs. Walk legs in between are part of the transfer,
    not a break in it, so the gap is measured from vehicle to vehicle."""
    with open(path) as f:
        for query, line in enumerate(f):
            if not line.strip():
                continue
            for itinerary in json.loads(line).get("itineraries", []):
                legs = [l for l in itinerary.get("legs", [])
                        if l.get("mode") != "WALK"]
                for arrive, depart in zip(legs, legs[1:]):
                    yield query, arrive, depart


def stop_id(leg, side, tag):
    sid = (leg.get(side) or {}).get("stopId") or ""
    return sid[len(tag):] if tag and sid.startswith(tag) else sid


def trip_id(leg, tag):
    """motis trip ids are '<date>_<time>_<src>_<feed trip id>'."""
    tid = (leg.get("tripId") or "").split("_", 2)[-1]
    return tid[len(tag):] if tag and tid.startswith(tag) else tid


def gap_minutes(arrive, depart):
    return (parse_time(depart["startTime"])
            - parse_time(arrive["endTime"])).total_seconds() / 60


# --- the three checks --------------------------------------------------------

def check_gtfs(path, feed_path, tag, limit):
    # Collect the transfers first so stops.txt and trips.txt can be filtered
    # down to what the journeys mention - a national feed has millions of trips
    # and a response dump touches a few thousand.
    work = []
    for query, arrive, depart in transfers(path):
        a = stop_id(arrive, "to", tag)
        b = stop_id(depart, "from", tag)
        if not a or not b:
            continue
        work.append((query, a, b, trip_id(arrive, tag), trip_id(depart, tag),
                     gap_minutes(arrive, depart)))

    rules = GtfsRules(Feed(feed_path),
                      {x[1] for x in work} | {x[2] for x in work},
                      {x[3] for x in work} | {x[4] for x in work})
    checked = governed = 0
    violations = []
    for (query, a, b, trip_a, trip_b, gap) in work:
        checked += 1
        hit = rules.applicable(a, b, trip_a, trip_b)
        if hit is None:
            continue
        governed += 1
        _, ty, minutes = hit
        if ty == 3:
            violations.append((query, a, b, "FORBIDDEN", gap, minutes))
        elif gap < minutes:
            violations.append((query, a, b, "TOO SHORT", gap, minutes))
    report(path, f"{checked} transfers, {governed} rule-governed",
           violations, limit)
    return len(violations)


def check_multi(path, config, feed_dir, limit):
    tags = read_dataset_tags(config)

    def split(sid):
        """Response stop ids are '<tag>_<feed stop id>'; a tag may itself
        contain an underscore, so every split point is tried."""
        i = sid.find("_")
        while i != -1:
            if sid[:i] in tags:
                return sid[:i], sid[i + 1:]
            i = sid.find("_", i + 1)
        return None, sid

    # Collect the work per feed first: only the feeds a journey actually
    # touches get opened, and only the stops and trips they mention get read.
    per_feed = defaultdict(list)
    cross_feed = 0
    for query, arrive, depart in transfers(path):
        tag_a, a = split((arrive.get("to") or {}).get("stopId") or "")
        tag_b, b = split((depart.get("from") or {}).get("stopId") or "")
        if tag_a is None or tag_a != tag_b:
            # No feed states a transfer between two feeds' stops, so this one
            # cannot be judged. Motis produces them when it merges duplicate
            # stops across sources.
            cross_feed += 1
            continue
        per_feed[tag_a].append((query, a, b,
                                trip_id(arrive, tag_a + "_"),
                                trip_id(depart, tag_b + "_"),
                                gap_minutes(arrive, depart)))

    checked = governed = feeds = 0
    violations = []
    for tag, work in per_feed.items():
        feed_path = os.path.join(feed_dir, tags[tag])
        if not os.path.exists(feed_path):
            continue
        feeds += 1
        checked += len(work)
        try:
            rules = GtfsRules(Feed(feed_path),
                              {x[1] for x in work} | {x[2] for x in work},
                              {x[3] for x in work} | {x[4] for x in work})
        except (zipfile.BadZipFile, OSError):
            continue
        if not rules.any_rule:
            continue
        for (query, a, b, trip_a, trip_b, gap) in work:
            hit = rules.applicable(a, b, trip_a, trip_b)
            if hit is None:
                continue
            governed += 1
            _, ty, minutes = hit
            if ty == 3:
                violations.append((query, f"{tag}:{a}", b, "FORBIDDEN",
                                   gap, minutes))
            elif gap < minutes:
                violations.append((query, f"{tag}:{a}", b, "TOO SHORT",
                                   gap, minutes))
    report(path,
           f"{feeds} feeds touched, {checked} transfers, {governed} "
           f"rule-governed, {cross_feed} cross-feed not checkable",
           violations, limit)
    return len(violations)


def check_hrdf(path, stamm, tag, limit):
    rules = HrdfRules(stamm)
    checked = 0
    skipped = defaultdict(int)
    violations = []
    for query, arrive, depart in transfers(path):
        a = stop_id(arrive, "to", tag)
        b = stop_id(depart, "from", tag)
        try:
            eva_a, eva_b = int(a), int(b)
        except ValueError:
            continue
        if eva_a != eva_b:
            skipped["inter-station (metabhf footpath)"] += 1
            continue
        if eva_a in rules.pair_rule_stations:
            skipped["station has pair rules"] += 1
            continue
        if eva_a not in rules.station_time:
            skipped["no umsteigb entry"] += 1
            continue
        checked += 1
        gap = gap_minutes(arrive, depart)
        required = rules.station_time[eva_a]
        if gap < required:
            violations.append((query, eva_a, "", "TOO SHORT", gap, required))
    out = ", ".join(f"{n} {why}" for why, n in sorted(skipped.items()))
    report(path, f"{checked} checkable same-station transfers "
                 f"(out of scope: {out})", violations, limit)
    return len(violations)


def read_dataset_tags(config):
    """tag -> feed path, from the `datasets:` block of a motis config. Kept to
    a hand-rolled scan so the tool needs no yaml dependency."""
    tags = {}
    in_datasets = False
    tag = None
    for line in open(config):
        if line.strip() == "datasets:":
            in_datasets = True
            continue
        if not in_datasets:
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 4 and line.strip().endswith(":"):
            tag = line.strip()[:-1]
        elif tag and "path:" in line:
            tags[tag] = line.split("path:", 1)[1].strip()
    return tags


def report(path, summary, violations, limit):
    name = os.path.basename(path)
    verdict = "OK" if not violations else f"{len(violations)} VIOLATIONS"
    print(f"{name}: {summary}, {verdict}")
    for (query, a, b, why, gap, required) in violations[:limit]:
        where = f"{a} -> {b}" if b else f"{a}"
        print(f"    q{query} {where}  {why}  "
              f"gap={gap:.0f}min required={required}min")
    if len(violations) > limit:
        print(f"    ... and {len(violations) - limit} more")


def main():
    ap = argparse.ArgumentParser(
        description="check motis journeys against a feed's transfer rules")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gtfs", metavar="PATH",
                     help="one GTFS feed, zip or unpacked directory")
    src.add_argument("--config", metavar="PATH",
                     help="motis config, to resolve many feeds by dataset tag")
    src.add_argument("--hrdf", metavar="STAMM",
                     help="HRDF stamm directory (umsteigb.txt et al.)")
    ap.add_argument("--feed-dir", metavar="DIR",
                    help="where --config's dataset paths are rooted")
    ap.add_argument("--tag", default="",
                    help="stop id prefix motis gave the dataset, e.g. 'ch_'")
    ap.add_argument("--max-shown", type=int, default=10, metavar="N",
                    help="violations to list per file (default 10)")
    ap.add_argument("responses", nargs="+",
                    help="response dumps written by `motis batch -r`")
    args = ap.parse_args()

    if args.config and not args.feed_dir:
        ap.error("--config needs --feed-dir")

    total = 0
    for path in args.responses:
        if args.gtfs:
            total += check_gtfs(path, args.gtfs, args.tag, args.max_shown)
        elif args.config:
            total += check_multi(path, args.config, args.feed_dir,
                                 args.max_shown)
        else:
            total += check_hrdf(path, args.hrdf, args.tag, args.max_shown)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
