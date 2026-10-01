"""Judging motis journeys by the transfer rules (transfers.txt) of their feeds.

Shared by check-transfer-rules.py (standalone) and validate.py (CI). The rules
are read straight from the feeds, so this also finds mistakes every motis
build shares. It follows nigiri's GTFS loader:

- the most specific rule wins: the GTFS specificity ladder first, then how
  many of the two stops the rule names exactly (not their station),
- a rule may name a stop or its station,
- type 1 (timed) holds the departing vehicle: 0 min,
- type 3 forbids the transfer,
- types 4 and 5 (in-seat) and rows without a time are no constraint.

Only transfers a rule governs are judged: without one, the stop's default
applies, which the feed does not state.
"""

import csv
import io
import json
import os
import zipfile
from collections import defaultdict
from datetime import datetime

# motis API modes that are no GTFS trip (openapi.yaml, Mode: "Street"; ODM
# and RIDE_SHARING are both): transfers.txt says nothing about them
STREET_MODES = {"WALK", "BIKE", "RENTAL", "CAR", "HGV", "CAR_PARKING",
                "CAR_DROPOFF", "ODM", "RIDE_SHARING", "FLEX",
                "DEBUG_BUS_ROUTE", "DEBUG_RAILWAY_ROUTE", "DEBUG_FERRY_ROUTE"}

FORBIDDEN, TOO_SHORT = "FORBIDDEN", "TOO SHORT"


class Feed:
    """A GTFS feed's tables, read on demand from a zip or a directory."""

    def __init__(self, path):
        self.path = path
        self.zip = zipfile.ZipFile(path) if zipfile.is_zipfile(path) else None

    def rows(self, name):
        """Yield the rows of one table; a missing table is empty, not an error:
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


def specificity_level(from_route, to_route, from_trip, to_trip):
    """The GTFS specificity ladder, least specific first. Mirrors the
    `specificity` levels of nigiri's loader/gtfs/transfer_rules.cc."""
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
    """The rules of one feed. Only the stops and trips in `wanted_stops` and
    `wanted_trips` are read: a national feed has millions of trips, a response
    dump touches a few thousand."""

    def __init__(self, feed, wanted_stops=None, wanted_trips=None):
        self.by_pair = defaultdict(list)
        self.parent = {}
        self.trip_route = {}

        for r in feed.rows("transfers.txt"):
            try:
                ty = int((r.get("transfer_type") or "").strip() or "0")
            except ValueError:
                continue
            if ty < 0 or ty > 3:
                continue  # 4, 5: in-seat, no transfer time
            mtt = (r.get("min_transfer_time") or "").strip()
            if ty in (0, 2) and not mtt:
                continue  # no time stated: no constraint
            from_trip = (r.get("from_trip_id") or "").strip()
            to_trip = (r.get("to_trip_id") or "").strip()
            if from_trip and from_trip == to_trip:
                continue  # a trip does not transfer to itself
            from_route = (r.get("from_route_id") or "").strip()
            to_route = (r.get("to_route_id") or "").strip()
            try:
                minutes = 0 if ty in (1, 3) else int(mtt) // 60
            except ValueError:
                continue
            key = ((r.get("from_stop_id") or "").strip(),
                   (r.get("to_stop_id") or "").strip())
            self.by_pair[key].append((
                specificity_level(from_route, to_route, from_trip, to_trip),
                ty, minutes, from_route, to_route, from_trip, to_trip))

        if not self.by_pair:
            return

        for r in feed.rows("stops.txt"):
            if wanted_stops is None or r["stop_id"] in wanted_stops:
                self.parent[r["stop_id"]] = \
                    (r.get("parent_station") or "").strip() or None
        for r in feed.rows("trips.txt"):
            if wanted_trips is None or r["trip_id"] in wanted_trips:
                self.trip_route[r["trip_id"]] = r["route_id"]

    def any_rule(self):
        return bool(self.by_pair)

    def required(self, a, b, trip_a, trip_b):
        """What the rules demand for arriving at `a` on `trip_a` and leaving
        `b` on `trip_b`: None (no rule), FORBIDDEN or the minutes. Rules of the
        same specificity that disagree are resolved in favor of the journey
        (the loader breaks such ties by row order, which a checker cannot
        see). The specificity is nigiri's `get_specificity`: the level, then
        how many of the two stops the rule names exactly."""
        route_a = self.trip_route.get(trip_a)
        route_b = self.trip_route.get(trip_b)
        best_specificity, best = -1, []
        for from_stop in (a, self.parent.get(a)):
            for to_stop in (b, self.parent.get(b)):
                if from_stop is None or to_stop is None:
                    continue
                for (level, ty, minutes, fr, tr, ft, tt) in \
                        self.by_pair.get((from_stop, to_stop), ()):
                    if (fr and fr != route_a) or (tr and tr != route_b) or \
                            (ft and ft != trip_a) or (tt and tt != trip_b):
                        continue
                    specificity = (level << 2) | \
                        ((from_stop == a) + (to_stop == b))
                    if specificity > best_specificity:
                        best_specificity, best = specificity, []
                    if specificity == best_specificity:
                        best.append((ty, minutes))
        if not best:
            return None
        allowed = [minutes for ty, minutes in best if ty != 3]
        return min(allowed) if allowed else FORBIDDEN


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def transfers(lines):
    """Yield (query index, arrival leg, departure leg) for every transfer
    between two transit legs of the responses (one JSON object per line).
    Street legs in between (walking, cycling, a car, flex, ...) are part of
    the transfer, so the gap is measured from vehicle to vehicle."""
    for query, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            response = json.loads(line)
        except ValueError:
            continue
        for itinerary in response.get("itineraries") or []:
            legs = [l for l in itinerary.get("legs") or []
                    if l.get("mode") not in STREET_MODES]
            for arrive, depart in zip(legs, legs[1:]):
                yield query, arrive, depart


def read_dataset_tags(config):
    """tag -> feed path, from the `datasets:` block of a motis config. Kept to
    a hand-rolled scan so the tools need no yaml dependency."""
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
        if indent == 0 and line.strip():
            break  # the next top level key
        if indent == 4 and line.strip().endswith(":"):
            tag = line.strip()[:-1]
        elif tag and "path:" in line:
            tags[tag] = line.split("path:", 1)[1].strip()
    return tags


class Checker:
    """Judges response dumps. `feeds` maps the id prefix motis gives a
    dataset's stops and trips ("<tag>_", or "" for a single untagged feed) to
    the feed's path."""

    def __init__(self, feeds):
        self.feeds = feeds

    @staticmethod
    def from_config(config, feed_dir):
        """The feeds of a motis config, their paths rooted at `feed_dir` (where
        the import ran)."""
        return Checker({tag + "_": os.path.join(feed_dir, path)
                        for tag, path in read_dataset_tags(config).items()})

    def split(self, sid):
        """(prefix, feed stop id), or (None, sid) for an unknown dataset. A tag
        may itself contain an underscore, so every split point is tried."""
        i = sid.find("_")
        while i != -1:
            if sid[:i + 1] in self.feeds:
                return sid[:i + 1], sid[i + 1:]
            i = sid.find("_", i + 1)
        return ("", sid) if "" in self.feeds else (None, sid)

    def check(self, lines):
        """Returns (summary, violations) for the responses in `lines`. A
        violation is (query index, feed stop a, feed stop b, FORBIDDEN or
        TOO_SHORT, gap in minutes, required minutes)."""
        per_feed = defaultdict(list)
        cross_feed = 0
        for query, arrive, depart in transfers(lines):
            pa, a = self.split((arrive.get("to") or {}).get("stopId") or "")
            pb, b = self.split((depart.get("from") or {}).get("stopId") or "")
            if pa is None or pa != pb:
                # No feed states a transfer between two feeds' stops. motis
                # produces them when it merges duplicate stops across feeds.
                cross_feed += 1
                continue
            per_feed[pa].append((
                query, a, b, trip_id(arrive, pa), trip_id(depart, pb),
                (parse_time(depart["startTime"]) -
                 parse_time(arrive["endTime"])).total_seconds() / 60))

        checked = governed = 0
        violations = []
        for prefix, work in per_feed.items():
            path = self.feeds[prefix]
            if not os.path.exists(path):
                continue
            try:
                rules = GtfsRules(Feed(path),
                                  {x[1] for x in work} | {x[2] for x in work},
                                  {x[3] for x in work} | {x[4] for x in work})
            except (zipfile.BadZipFile, OSError):
                continue
            checked += len(work)
            if not rules.any_rule():
                continue
            for (query, a, b, trip_a, trip_b, gap) in work:
                req = rules.required(a, b, trip_a, trip_b)
                if req is None:
                    continue
                governed += 1
                if req == FORBIDDEN:
                    violations.append((query, prefix + a, prefix + b,
                                       FORBIDDEN, gap, 0))
                elif gap < req:
                    violations.append((query, prefix + a, prefix + b,
                                       TOO_SHORT, gap, req))
        summary = (f"{checked} transfers, {governed} rule-governed, "
                   f"{cross_feed} between feeds (not judged)")
        return summary, violations


def trip_id(leg, prefix):
    """motis trip ids are '<date>_<time>_<prefix><feed trip id>'."""
    tid = (leg.get("tripId") or "").split("_", 2)[-1]
    return tid[len(prefix):] if prefix and tid.startswith(prefix) else tid


def format_violation(v):
    query, a, b, why, gap, required = v
    if why == FORBIDDEN:
        return f"q{query + 1} {a} -> {b}: forbidden (gap {gap:.0f} min)"
    return (f"q{query + 1} {a} -> {b}: {gap:.0f} min, "
            f"the rule requires {required} min")
