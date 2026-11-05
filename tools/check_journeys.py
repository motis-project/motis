#!/usr/bin/env python3
"""Check the journeys of a `motis batch` response file against the GTFS feeds.

Every transit leg is checked against the feed named by the tag in its tripId
(`<service date>_<start HH:MM>_<tag>_<GTFS trip_id>`):
  * the trip exists, runs on that service date (calendar.txt,
    calendar_dates.txt) and starts at that time (first stop_times.txt row, or
    a frequencies.txt departure),
  * from, intermediate and to stops are the trip's stop_times rows in travel
    order; trips joined by nigiri are followed the same way: stay-seated
    transfers (transfers.txt, transfer_type 4, dwell time below two hours)
    and, for trips without them, block_id (the next trip of the block whose
    first stop is the previous trip's last stop),
  * the scheduled departure and arrival times equal the stop_times entries,
    converted with the agency's time zone (GTFS "noon minus 12 h" rule, times
    cut to the minute like nigiri reads them),
  * boarding at `from` and alighting at `to` is allowed, the routeId is the
    trip's route, and without real-time data the actual times equal the
    scheduled ones.
Every itinerary (direct connections too) is checked for consistency: legs are
contiguous in time, the itinerary's times and duration follow from its legs,
and it respects the requested departure or arrival time.

Names (route short name, agency name, headsign) are compared as warnings.

usage: tools/check_journeys.py [-r responses.txt] [-c config.yml]
                               [--gtfs TAG=PATH ...]
The feeds are taken from the config's timetable.datasets (tag -> path,
relative to the config file) and/or --gtfs; PATH is a GTFS zip or directory.
Query numbers in the report are 1-based like those of `motis compare`. Exit
status 1 if a check fails, with --strict also on warnings.
"""
import argparse
import collections
import csv
import datetime as dt
import io
import json
import os
import re
import sys
import zipfile
from zoneinfo import ZoneInfo

UTC = dt.timezone.utc
# API times are cut to whole minutes (openapi::date_time_t), durations keep
# seconds: a street leg's duration may differ from its time span by up to 59 s.
TIME_PRECISION = 60
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday")
MAX_CHAIN = 64  # trips followed through blocks / stay-seated transfers
MAX_CHAINS = 32  # alternative continuations considered per leg
SEATED_MAX_DWELL = 120 * 60  # nigiri: stay seated only below two hours

# One stop_times.txt row. `date` (service date) and `tz` are set when a chain
# of trips is built, since a stay-seated successor may run on the next day.
Row = collections.namedtuple(
    "Row", "seq stop arr dep pickup dropoff headsign trip date tz")


def parse_iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def gtfs_seconds(s):
    """'25:03:30' -> 90180: seconds since the start of the service day, cut to
    the minute like nigiri reads them. None for an empty field."""
    s = (s or "").strip()
    if not s:
        return None
    h, m, _ = s.split(":")
    return int(h) * 3600 + int(m) * 60


def hhmm(seconds):
    return "%02d:%02d" % (seconds // 3600, seconds // 60 % 60)


def split_trip_id(trip_id):
    """'20250110_08:15_tag_trip' -> (date, minutes, tag, 'trip')."""
    m = re.match(r"^(\d{8})_(\d{2,}):(\d{2})_([^_]+)_(.+)$", trip_id, re.S)
    if not m:
        raise ValueError("tripId not <date>_<HH:MM>_<tag>_<trip_id>: %r"
                         % trip_id)
    date = dt.date(int(m[1][:4]), int(m[1][4:6]), int(m[1][6:8]))
    return date, int(m[2]) * 60 + int(m[3]), m[4], m[5]


def split_tag(x):
    """'tag_id' -> ('tag', 'id')."""
    tag, _, rest = (x or "").partition("_")
    return tag, rest


class Feed:
    """One GTFS feed, loaded only as far as the responses need it."""

    def __init__(self, tag, path):
        self.tag = tag
        self.path = path
        self.zip = None
        if not os.path.isdir(path):
            self.zip = zipfile.ZipFile(path)
            self.names = {}
            for n in self.zip.namelist():
                base = os.path.basename(n)
                if base.endswith(".txt") and base not in self.names:
                    self.names[base] = n
        self.noon_cache = {}

    def rows(self, name):
        if self.zip is not None:
            if name not in self.names:
                return
            f = io.TextIOWrapper(self.zip.open(self.names[name]),
                                 encoding="utf-8-sig", newline="")
        else:
            p = os.path.join(self.path, name)
            if not os.path.exists(p):
                return
            f = open(p, encoding="utf-8-sig", newline="")
        with f:
            r = csv.DictReader(f)
            if r.fieldnames is None:
                return
            r.fieldnames = [c.strip() for c in r.fieldnames]
            for row in r:
                yield row

    def open_text(self, name):
        if self.zip is not None:
            if name not in self.names:
                return None
            return io.TextIOWrapper(self.zip.open(self.names[name]),
                                    encoding="utf-8-sig", newline="")
        p = os.path.join(self.path, name)
        if not os.path.exists(p):
            return None
        return open(p, encoding="utf-8-sig", newline="")

    def stop_time_rows(self, wanted):
        """stop_times.txt rows of the wanted trips. The file is the bulk of a
        feed, so when trip_id is the first column (the usual layout) each line
        is tested on its first field before it is parsed as CSV."""
        f = self.open_text("stop_times.txt")
        if f is None:
            return
        with f:
            header = f.readline()
            if not header:
                return
            fields = [c.strip() for c in next(csv.reader([header]))]
            fast = (fields and fields[0] == "trip_id"
                    and not any("," in t or '"' in t for t in wanted))
            if fast:
                def selected():
                    for line in f:
                        key = line.partition(",")[0]
                        if key[:1] == '"':
                            key = key[1:-1]
                        if key in wanted:
                            yield line
                reader = csv.DictReader(selected(), fieldnames=fields)
            else:
                reader = csv.DictReader(f, fieldnames=fields)
            for r in reader:
                if r["trip_id"] in wanted:
                    yield r

    def load(self, needed_trips):
        self.agencies = {}  # agency_id -> (name, tz)
        for a in self.rows("agency.txt"):
            tz = a.get("agency_timezone", "").strip()
            self.agencies[a.get("agency_id", "").strip()] = (
                a.get("agency_name", "").strip(), ZoneInfo(tz) if tz else None)
        self.default_agency = (next(iter(self.agencies.values()))
                               if len(self.agencies) == 1 else None)
        self.routes = {r["route_id"]: r for r in self.rows("routes.txt")}
        self.trips = {}  # trip_id -> row
        blocks = collections.defaultdict(list)
        for t in self.rows("trips.txt"):
            self.trips[t["trip_id"]] = t
            if t.get("block_id", "").strip():
                blocks[t["block_id"].strip()].append(t["trip_id"])
        # Stay-seated transfers between trips (nigiri prefers them over
        # block_id). The CH feed repeats rows per service_id: deduplicated.
        self.seated_out = collections.defaultdict(list)
        self.seated_in = collections.defaultdict(list)
        for t in self.rows("transfers.txt"):
            if (t.get("transfer_type", "").strip() == "4"
                    and t.get("from_trip_id", "").strip()
                    and t.get("to_trip_id", "").strip()):
                a, b = t["from_trip_id"].strip(), t["to_trip_id"].strip()
                if b not in self.seated_out[a]:
                    self.seated_out[a].append(b)
                    self.seated_in[b].append(a)
        # stop_times are read for the referenced trips and everything they can
        # be joined with.
        wanted = set(needed_trips)
        queue = list(needed_trips)
        while queue:
            tid = queue.pop()
            t = self.trips.get(tid)
            if t is None:
                continue
            more = list(self.seated_out.get(tid, []))
            if t.get("block_id", "").strip():
                more += blocks[t["block_id"].strip()]
            for m in more:
                if m not in wanted:
                    wanted.add(m)
                    queue.append(m)
        self.stop_times = collections.defaultdict(list)
        for r in self.stop_time_rows(wanted):
            self.stop_times[r["trip_id"]].append(Row(
                int(r["stop_sequence"]), r["stop_id"],
                gtfs_seconds(r.get("arrival_time")),
                gtfs_seconds(r.get("departure_time")),
                (r.get("pickup_type") or "0").strip() or "0",
                (r.get("drop_off_type") or "0").strip() or "0",
                (r.get("stop_headsign") or "").strip(), r["trip_id"],
                None, None))
        for tid, v in self.stop_times.items():
            v.sort(key=lambda x: x.seq)
            for i, st in enumerate(v):
                if st.arr is None or st.dep is None:
                    v[i] = st._replace(arr=st.arr if st.arr is not None
                                       else st.dep,
                                       dep=st.dep if st.dep is not None
                                       else st.arr)
        # Block mates in nigiri's order: by first departure.
        self.blocks = {}
        for b, tids in blocks.items():
            tids = [t for t in tids if self.stop_times.get(t)]
            if tids:
                tids.sort(key=lambda t: self.stop_times[t][0].dep or 0)
                self.blocks[b] = tids
        self.calendar = {c["service_id"]: c for c in self.rows("calendar.txt")}
        self.calendar_dates = collections.defaultdict(dict)
        for c in self.rows("calendar_dates.txt"):
            self.calendar_dates[c["service_id"]][c["date"].strip()] = \
                c["exception_type"].strip()
        self.frequencies = collections.defaultdict(list)
        for f in self.rows("frequencies.txt"):
            if f["trip_id"] in wanted:
                self.frequencies[f["trip_id"]].append(
                    (gtfs_seconds(f["start_time"]), gtfs_seconds(f["end_time"]),
                     int(f["headway_secs"])))
        self.stops = {s["stop_id"]: s.get("stop_name", "")
                      for s in self.rows("stops.txt")}

    def active(self, service_id, date):
        ds = date.strftime("%Y%m%d")
        exc = self.calendar_dates.get(service_id, {}).get(ds)
        if exc == "1":
            return True
        if exc == "2":
            return False
        c = self.calendar.get(service_id)
        return (c is not None and c["start_date"].strip() <= ds
                <= c["end_date"].strip()
                and c[WEEKDAYS[date.weekday()]].strip() == "1")

    def agency_of(self, trip):
        r = self.routes.get(trip["route_id"])
        a = self.agencies.get((r or {}).get("agency_id", "").strip())
        return a or self.default_agency

    def tz_of(self, trip_id):
        a = self.agency_of(self.trips[trip_id])
        return a[1] if a else None

    def utc(self, tz, date, seconds):
        """GTFS time on a service date: noon minus 12 h, plus the time."""
        key = (tz, date)
        noon = self.noon_cache.get(key)
        if noon is None:
            noon = dt.datetime.combine(date, dt.time(12), tz).astimezone(UTC)
            self.noon_cache[key] = noon
        return noon - dt.timedelta(hours=12) + dt.timedelta(seconds=seconds)

    def start_offset(self, trip_id, date, start_min):
        """Seconds to add to the stop_times of the trip for the departure in
        the tripId (frequencies.txt), or an error message."""
        rows = self.stop_times.get(trip_id)
        first = rows[0].dep
        if first is None:
            return 0, None
        if trip_id in self.frequencies:
            for start, end, headway in self.frequencies[trip_id]:
                t = start
                while t < end:
                    if t // 60 == start_min:
                        return t - first, None
                    t += headway
            return None, ("tripId start %s is no frequencies.txt departure"
                          % hhmm(start_min * 60))
        if first // 60 != start_min:
            return None, ("trip starts %s, tripId says %s"
                          % (hhmm(first), hhmm(start_min * 60)))
        return 0, None

    def dated(self, trip_id, date):
        tz = self.tz_of(trip_id)
        return [r._replace(date=date, tz=tz) for r in self.stop_times[trip_id]]

    def joined(self, rows, nxt, seated):
        """rows + nxt the way nigiri concatenates trips: the shared stop once,
        with the arrival of the previous and the departure of the next trip.
        Boarding there follows the next trip for block_id joins
        (local_to_utc.h get_stop_seq) and the previous trip's last stop for
        stay-seated transfers (seated.cc keeps that stop entry)."""
        first = nxt[0]
        return rows[:-1] + [rows[-1]._replace(
            dep=first.dep, pickup=rows[-1].pickup if seated else first.pickup,
            headsign=first.headsign, trip=first.trip)] + nxt[1:]

    def chains(self, trip_id, date):
        """Possible stop sequences a leg starting on `trip_id` can run along:
        the trip's rows, continued through its stay-seated transfers (every
        successor runs on the day its first departure follows the previous
        trip's last arrival by less than two hours; several successors mean
        several transports) or, without any, through its block."""
        start = self.dated(trip_id, date)
        if trip_id in self.seated_out or trip_id in self.seated_in:
            out, todo = [], [(start, trip_id, 0)]
            while todo:
                rows, current, depth = todo.pop()
                last_arr = rows[-1].tz and rows[-1].arr is not None and \
                    self.utc(rows[-1].tz, rows[-1].date, rows[-1].arr)
                nexts = []
                for succ in self.seated_out.get(current, []):
                    if (depth >= MAX_CHAIN or last_arr is None
                            or not self.stop_times.get(succ)
                            or self.tz_of(succ) is None):
                        continue
                    for day in range(3):
                        d = rows[-1].date + dt.timedelta(days=day)
                        if not self.active(self.trips[succ]["service_id"], d):
                            continue
                        nxt = self.dated(succ, d)
                        if nxt[0].dep is None:
                            continue
                        dwell = (self.utc(nxt[0].tz, d, nxt[0].dep)
                                 - last_arr).total_seconds()
                        if 0 <= dwell < SEATED_MAX_DWELL:
                            nexts.append((self.joined(rows, nxt, True), succ,
                                          depth + 1))
                            break
                if nexts and len(out) + len(todo) < MAX_CHAINS:
                    todo.extend(nexts)
                else:
                    out.append(rows)
            return out
        rows = start
        block = self.trips[trip_id].get("block_id", "").strip()
        mates = self.blocks.get(block, []) if block else []
        current = trip_id
        for _ in range(MAX_CHAIN):
            if current not in mates:
                break
            succ = None
            for cand in mates[mates.index(current) + 1:]:
                if (self.stop_times[cand][0].stop == rows[-1].stop
                        and self.active(self.trips[cand]["service_id"], date)):
                    succ = cand
                    break
            if succ is None:
                break
            rows = self.joined(rows, self.dated(succ, date), False)
            current = succ
        return [rows]


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.counts = collections.Counter()

    def error(self, where, category, msg):
        self.errors.append((where, category, msg))
        self.counts["error: " + category] += 1

    def warn(self, where, category, msg):
        self.warnings.append((where, category, msg))
        self.counts["warning: " + category] += 1


def check_transit_leg(leg, where, params, feeds, rep, rt=False):
    """`rt`: real-time data was applied to this response. A leg with real-time
    data may then belong to an added trip or run along changed stops, so for
    such legs a trip missing from the feed is counted, not reported, and stop
    and time mismatches are warnings."""
    tolerant = rt and leg.get("realTime")
    fail = rep.warn if tolerant else rep.error
    try:
        date, start_min, tag, trip_id = split_trip_id(leg["tripId"])
    except ValueError as e:
        rep.error(where, "trip id", str(e))
        return
    feed = feeds.get(tag)
    if feed is None:
        rep.error(where, "feed", "tripId tag %r is no loaded feed" % tag)
        return
    trip = feed.trips.get(trip_id)
    if trip is None:
        if tolerant:
            rep.counts["real-time legs of trips not in the feed"] += 1
        else:
            rep.error(where, "trip", "trip %r not in %s" % (trip_id, tag))
        return
    if not feed.stop_times.get(trip_id):
        rep.error(where, "trip", "trip %r has no stop_times" % trip_id)
        return
    if not feed.active(trip["service_id"], date):
        fail(where, "service day", "trip %r does not run on %s"
             % (trip_id, date))
    agency = feed.agency_of(trip)
    if agency is None or agency[1] is None:
        rep.error(where, "feed", "no agency time zone for trip %r" % trip_id)
        return
    offset, err = feed.start_offset(trip_id, date, start_min)
    if err:
        fail(where, "trip start", err)
        offset = 0

    def at(row, event):
        """UTC time of a row's event, None if the feed leaves it empty."""
        t = row.dep if event == "dep" else row.arr
        if t is None:
            return None
        return feed.utc(row.tz, row.date,
                        t + (offset if row.trip == trip_id else 0))

    def fmt(row, event):
        t = at(row, event)
        return t.isoformat() if t else "(interpolated)"

    def stop_id(place):
        t, s = split_tag(place.get("stopId"))
        if t != tag:
            fail(where, "stop", "stop %r is not from feed %s"
                 % (place.get("stopId"), tag))
        elif s not in feed.stops:
            fail(where, "stop", "stop %r not in stops.txt" % s)
        return s

    def time_of(place, key, leg_key):
        """The scheduled time of a leg end: the place's field, else the leg's."""
        return parse_iso(place.get(key) or leg[leg_key])

    def find(rows, stop, time, lo, hi, event):
        """Index of the row with `stop` in rows[lo:hi] whose event time is
        `time`; the first row with the stop if none matches in time."""
        cands = [i for i in range(lo, hi) if rows[i].stop == stop]
        for i in cands:
            if at(rows[i], event) == time:
                return i, True
        return (cands[0], False) if cands else (None, False)

    from_stop = stop_id(leg["from"])
    to_stop = stop_id(leg["to"])
    dep = time_of(leg["from"], "scheduledDeparture", "scheduledStartTime")
    arr = time_of(leg["to"], "scheduledArrival", "scheduledEndTime")
    if parse_iso(leg["scheduledStartTime"]) != dep:
        rep.error(where, "place time", "from.scheduledDeparture %s != leg "
                  "scheduledStartTime %s" % (leg["from"].get(
                      "scheduledDeparture"), leg["scheduledStartTime"]))
    if parse_iso(leg["scheduledEndTime"]) != arr:
        rep.error(where, "place time", "to.scheduledArrival %s != leg "
                  "scheduledEndTime %s" % (leg["to"].get("scheduledArrival"),
                                           leg["scheduledEndTime"]))

    chains = feed.chains(trip_id, date)
    own = len(feed.stop_times[trip_id])  # rows of the tripId's own trip
    # The departure at the trip's last stop belongs to the next trip.
    i, ok = find(chains[0], from_stop, dep, 0, max(own - 1, 1), "dep")
    if i is None:
        fail(where, "stop order", "from stop %r is not on trip %r"
                  % (from_stop, trip_id))
        return
    if not ok:
        fail(where, "time", "departure %s at %r, stop_times says %s"
                  % (dep.isoformat(), from_stop, fmt(chains[0][i], "dep")))
    if chains[0][i].pickup == "1":
        fail(where, "boarding", "no pickup at %r (pickup_type 1)"
                  % from_stop)

    rows, j, ok = None, None, False
    for cand in chains:
        j2, ok2 = find(cand, to_stop, arr, i + 1, len(cand), "arr")
        if j2 is not None and (rows is None or ok2 and not ok):
            rows, j, ok = cand, j2, ok2
        if ok:
            break
    if rows is None:
        fail(where, "stop order", "to stop %r is not on trip %r after "
                  "%r%s" % (to_stop, trip_id, from_stop,
                            " (%d continuation%s followed)"
                            % (len(chains), "s" if len(chains) > 1 else "")
                            if any(len(c) > own for c in chains) else ""))
        return
    if not ok:
        fail(where, "time", "arrival %s at %r, stop_times says %s"
                  % (arr.isoformat(), to_stop, fmt(rows[j], "arr")))
    if rows[j].dropoff == "1":
        fail(where, "alighting", "no drop-off at %r (drop_off_type 1)"
                  % to_stop)

    inter = leg.get("intermediateStops")
    if inter is not None:
        # motis leaves out stops where neither boarding nor alighting is
        # allowed unless withScheduledSkippedStops is set.
        expected = [r for r in rows[i + 1:j]
                    if params.get("withScheduledSkippedStops") == "true"
                    or not (r.pickup == "1" and r.dropoff == "1")]
        if len(inter) != len(expected):
            fail(where, "intermediate stops", "%d intermediate stops, "
                      "stop_times has %d between %r and %r"
                      % (len(inter), len(expected), from_stop, to_stop))
        else:
            for k, (p, row) in enumerate(zip(inter, expected)):
                s = stop_id(p)
                if s != row.stop:
                    fail(where, "intermediate stops", "intermediate stop "
                              "%d is %r, stop_times says %r" % (k, s, row.stop))
                    break
                for key, ev in (("scheduledArrival", "arr"),
                                ("scheduledDeparture", "dep")):
                    t = at(row, ev)
                    if p.get(key) and t is not None and parse_iso(p[key]) != t:
                        fail(where, "time", "intermediate stop %d %r %s "
                                  "%s, stop_times says %s"
                                  % (k, s, key, p[key], t.isoformat()))

    route_id = "%s_%s" % (tag, trip["route_id"])
    if leg.get("routeId") != route_id:
        fail(where, "route", "routeId %r, trip %r has %r"
                  % (leg.get("routeId"), trip_id, route_id))

    if not leg.get("realTime"):
        for a, b in (("startTime", "scheduledStartTime"),
                     ("endTime", "scheduledEndTime")):
            if parse_iso(leg[a]) != parse_iso(leg[b]):
                rep.error(where, "realtime", "%s %s != %s %s without "
                          "real-time data" % (a, leg[a], b, leg[b]))

    route = feed.routes.get(trip["route_id"], {})
    short = (route.get("route_short_name") or "").strip()
    if short and leg.get("routeShortName") and leg["routeShortName"] != short:
        rep.warn(where, "route name", "routeShortName %r, routes.txt says %r"
                 % (leg["routeShortName"], short))
    if agency[0] and leg.get("agencyName") and leg["agencyName"] != agency[0]:
        rep.warn(where, "agency name", "agencyName %r, agency.txt says %r"
                 % (leg["agencyName"], agency[0]))
    headsigns = {rows[i].headsign, (trip.get("trip_headsign") or "").strip()}
    headsigns.discard("")
    if headsigns and leg.get("headsign") and leg["headsign"] not in headsigns:
        rep.warn(where, "headsign", "headsign %r, GTFS has %s"
                 % (leg["headsign"], sorted(headsigns)))
    rep.counts["transit legs checked"] += 1


def check_itinerary(it, where, params, feeds, rep, rt=False):
    legs = it.get("legs") or []
    if not legs:
        rep.error(where, "itinerary", "no legs")
        return
    start, end = parse_iso(it["startTime"]), parse_iso(it["endTime"])
    if parse_iso(legs[0]["startTime"]) != start:
        rep.error(where, "itinerary", "startTime %s != first leg start %s"
                  % (it["startTime"], legs[0]["startTime"]))
    if parse_iso(legs[-1]["endTime"]) != end:
        rep.error(where, "itinerary", "endTime %s != last leg end %s"
                  % (it["endTime"], legs[-1]["endTime"]))
    if abs(it["duration"] - (end - start).total_seconds()) >= TIME_PRECISION:
        rep.error(where, "itinerary", "duration %d != endTime - startTime %d"
                  % (it["duration"], (end - start).total_seconds()))
    time = params.get("time")
    if time:
        t = parse_iso(time)
        if params.get("arriveBy", "false") == "true":
            if end > t:
                rep.error(where, "query time", "arrives %s after requested %s"
                          % (it["endTime"], time))
        elif start < t:
            rep.error(where, "query time", "departs %s before requested %s"
                      % (it["startTime"], time))

    n_transit = n_interlined = 0
    prev = None
    for li, leg in enumerate(legs):
        lw = "%s leg %d (%s)" % (where, li, leg["mode"])
        ls, le = parse_iso(leg["startTime"]), parse_iso(leg["endTime"])
        if abs(leg["duration"] - (le - ls).total_seconds()) >= TIME_PRECISION:
            rep.error(lw, "leg", "duration %d != endTime - startTime %d"
                      % (leg["duration"], (le - ls).total_seconds()))
        if prev is not None:
            if ls < parse_iso(prev["endTime"]):
                rep.error(lw, "leg order", "starts %s before the previous leg "
                          "ends %s" % (leg["startTime"], prev["endTime"]))
            if leg.get("interlineWithPreviousLeg"):
                n_interlined += 1
                if not (leg.get("tripId") and prev.get("tripId")):
                    rep.error(lw, "interlining", "interlined with a non-transit "
                              "leg")
                elif prev["to"].get("stopId") != leg["from"].get("stopId"):
                    rep.error(lw, "interlining", "interlined leg starts at %r, "
                              "previous ends at %r"
                              % (leg["from"].get("stopId"),
                                 prev["to"].get("stopId")))
        if leg.get("tripId"):
            n_transit += 1
            check_transit_leg(leg, lw, params, feeds, rep, rt)
        prev = leg
    expected = max(0, n_transit - 1 - n_interlined)
    if n_transit and it.get("transfers") != expected:
        rep.warn(where, "transfers", "transfers %s, %d transit legs (%d "
                 "interlined) suggest %d" % (it.get("transfers"), n_transit,
                                             n_interlined, expected))
    rep.counts["itineraries with transit" if n_transit else
               "itineraries without transit"] += 1


def parse_responses(lines):
    """[(line number, plan response dict or None)] for the lines of a
    `motis batch` responses file."""
    out = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            out.append((i, None))
            continue
        try:
            r = json.loads(line)
        except ValueError:
            r = None
        out.append((i, r if isinstance(r, dict) and "itineraries" in r
                    else None))
    return out


def datasets_from_config(path):
    """(tag -> path as written, merge_dupes) from a motis config."""
    text = open(path).read()
    merge_dupes = re.search(r"^\s*merge_dupes_(intra|inter)_src:\s*true\s*$",
                            text, re.M) is not None
    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load(text)
        return {tag: d["path"] for tag, d in
                (cfg.get("timetable") or {}).get("datasets", {}).items()}, \
            merge_dupes
    except ImportError:
        pass
    # Minimal fallback: the `datasets:` mapping of a motis config.
    out, tag, indent = {}, None, None
    in_datasets = False
    for line in text.splitlines():
        if re.match(r"^\s*#", line) or not line.strip():
            continue
        ind = len(line) - len(line.lstrip())
        if re.match(r"^\s*datasets:\s*$", line):
            in_datasets, indent, tag = True, ind, None
            continue
        if in_datasets:
            if ind <= indent:
                in_datasets = False
                continue
            m = re.match(r"^\s*([^\s:]+):\s*$", line)
            if m and (tag is None or ind <= tag[1]):
                tag = (m[1], ind)
                continue
            m = re.match(r"^\s*path:\s*(.+?)\s*$", line)
            if m and tag is not None:
                out[tag[0]] = m[1].strip("'\"")
    return out, merge_dupes


def feeds_from_config(path):
    """(tag -> feed path, merge_dupes). A relative dataset path is tried next
    to the config, in the config's parent directory (a data dir's config.yml
    names the feeds as they were given to `motis import`, relative to the
    directory the import ran in) and in the working directory."""
    datasets, merge_dupes = datasets_from_config(path)
    base = os.path.dirname(os.path.abspath(path))
    out = {}
    for tag, p in datasets.items():
        if os.path.isabs(p):
            out[tag] = p
            continue
        cands = [os.path.join(d, p) for d in
                 (base, os.path.dirname(base), os.getcwd())]
        out[tag] = next((c for c in cands if os.path.exists(c)), cands[0])
    return out, merge_dupes


def check(lines, feeds_cfg, rt_flags=None, max_report=50, log=print,
          merge_dupes=False):
    """Check the responses in `lines` (one JSON document per query, as `motis
    batch` writes them) against the feeds in `feeds_cfg` (tag -> path).
    `rt_flags`: per line, whether real-time data was applied (see
    check_transit_leg). Prints the findings with `log` and returns the
    Report; its `errors` decide the verdict."""
    rep = Report()
    if merge_dupes:
        log("check_journeys: note: merge_dupes is on: a merged duplicate trip "
            "keeps the stops of one trip and the id of another, so stop checks "
            "can fail without a routing error")
    responses = parse_responses(lines)
    needed = collections.defaultdict(set)
    for _, r in responses:
        for kind in ("itineraries", "direct"):
            for it in (r or {}).get(kind) or []:
                for leg in it.get("legs") or []:
                    if leg.get("tripId"):
                        try:
                            _, _, tag, tid = split_trip_id(leg["tripId"])
                            needed[tag].add(tid)
                        except ValueError:
                            pass

    feeds = {}
    for tag in sorted(needed):
        if tag not in feeds_cfg:
            log("check_journeys: no feed for tag %r (%d trips referenced)"
                % (tag, len(needed[tag])))
            continue
        feeds[tag] = Feed(tag, feeds_cfg[tag])
        feeds[tag].load(needed[tag])
        log("check_journeys: %s: %d trips referenced, %d loaded from %s"
            % (tag, len(needed[tag]),
               sum(1 for t in needed[tag] if t in feeds[tag].stop_times),
               feeds_cfg[tag]))

    for qi, (lineno, r) in enumerate(responses, 1):
        if r is None:
            head = lines[lineno - 1].strip()[:120]
            rep.error("query %d" % qi, "response", "line %d is no plan "
                      "response: %s" % (lineno, head or "(empty)"))
            continue
        rep.counts["responses"] += 1
        rt = bool(rt_flags[qi - 1]) if rt_flags else False
        params = r.get("requestParameters") or {}
        for kind in ("itineraries", "direct"):
            for ii, it in enumerate(r.get(kind) or []):
                check_itinerary(it, "query %d %s %d" % (qi, kind, ii), params,
                                feeds, rep, rt)

    for sev, items in (("ERROR", rep.errors), ("WARNING", rep.warnings)):
        for where, cat, msg in items[:max_report]:
            log("%s %s [%s] %s" % (sev, where, cat, msg))
        if len(items) > max_report:
            log("... %d more %ss" % (len(items) - max_report, sev.lower()))
    log("check_journeys: " + ", ".join(
        "%s: %d" % kv for kv in sorted(rep.counts.items())))
    log("check_journeys: %d errors, %d warnings"
        % (len(rep.errors), len(rep.warnings)))
    return rep


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-r", "--responses", default="responses.txt")
    ap.add_argument("-c", "--config", default=None,
                    help="motis config.yml with timetable.datasets "
                         "(default: config.yml if it exists)")
    ap.add_argument("--gtfs", action="append", default=[], metavar="TAG=PATH",
                    help="feed for a dataset tag (zip or directory)")
    ap.add_argument("--rt", action="store_true",
                    help="real-time data was applied: legs with real-time "
                         "data may belong to added trips or changed routes, "
                         "their mismatches are warnings")
    ap.add_argument("--max-report", type=int, default=50,
                    help="findings to print (default 50)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on warnings too")
    args = ap.parse_args()

    feeds_cfg, merge_dupes = {}, False
    config = args.config or ("config.yml" if os.path.exists("config.yml")
                             else None)
    if config:
        feeds_cfg, merge_dupes = feeds_from_config(config)
    for x in args.gtfs:
        tag, _, path = x.partition("=")
        feeds_cfg[tag] = path
    if not feeds_cfg:
        sys.exit("check_journeys: no feeds (use -c config.yml or --gtfs)")

    with open(args.responses) as f:
        lines = f.read().splitlines()
    rep = check(lines, feeds_cfg, rt_flags=[args.rt] * len(lines),
                max_report=args.max_report, merge_dupes=merge_dupes)
    sys.exit(1 if rep.errors or (args.strict and rep.warnings) else 0)


if __name__ == "__main__":
    main()
