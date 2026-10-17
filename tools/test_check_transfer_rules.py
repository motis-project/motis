#!/usr/bin/env python3

"""Test for transfer_rules.py and check-transfer-rules.py.

Run with `python3 tools/test_check_transfer_rules.py` (or pytest).
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from transfer_rules import FORBIDDEN, TOO_SHORT, Checker  # noqa: E402

TOOL = os.path.join(HERE, "check-transfer-rules.py")

# S (station ST with the stops S and S2) and the stop S9. Every change of
# vehicles at S takes 10 minutes; T1 -> T2 is guaranteed, although the row
# states 5 minutes; T1 -> T4 is forbidden; two rows of the same specificity
# disagree on T1 -> T5 at S2.
FEED = {
    "stops.txt": "stop_id,stop_name,stop_lat,stop_lon,location_type,"
                 "parent_station\n"
                 "ST,ST,50.0,8.0,1,\n"
                 "S,S,50.0,8.0,0,ST\n"
                 "S2,S2,50.0,8.0,0,ST\n"
                 "S9,S9,50.1,8.0,0,\n",
    "trips.txt": "route_id,service_id,trip_id\n"
                 "R1,X,T1\n"
                 "R2,X,T2\n"
                 "R3,X,T3\n"
                 "R4,X,T4\n"
                 "R5,X,T5\n",
    "transfers.txt": "from_stop_id,to_stop_id,transfer_type,min_transfer_time,"
                     "from_trip_id,to_trip_id\n"
                     "S,S,2,600,,\n"
                     "S,S,1,300,T1,T2\n"
                     "S,S,3,,T1,T4\n"
                     "ST,ST,2,900,,\n"
                     "S,S2,2,600,T1,T5\n"
                     "S,S2,2,180,T1,T5\n",
}


def leg(mode, from_stop, to_stop, start, end, trip=None):
    x = {"mode": mode,
         "from": {"stopId": from_stop} if from_stop else {},
         "to": {"stopId": to_stop} if to_stop else {},
         "startTime": start,
         "endTime": end}
    if trip:
        x["tripId"] = trip
    return x


def change(at, to, gap, departing):
    """T1 arrives at `at` 08:30, `departing` leaves `to` `gap` min later."""
    return {"legs": [
        leg("REGIONAL_RAIL", "t_S9", "t_" + at,
            "2019-05-01T08:00:00Z", "2019-05-01T08:30:00Z",
            trip="20190501_10:00_t_T1"),
        leg("REGIONAL_RAIL", "t_" + to, "t_S9",
            "2019-05-01T08:%02d:00Z" % (30 + gap), "2019-05-01T09:00:00Z",
            trip="20190501_10:30_t_" + departing)]}


class TransferRulesTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        feed = os.path.join(self.dir.name, "feed")
        os.mkdir(feed)
        for name, content in FEED.items():
            with open(os.path.join(feed, name), "w") as f:
                f.write(content)
        self.feed = feed
        self.checker = Checker({"t_": feed})

    def tearDown(self):
        self.dir.cleanup()

    def check(self, itinerary):
        return self.checker.check([json.dumps({"itineraries": [itinerary]})])[1]

    def test_too_short(self):
        v = self.check(change("S", "S", 5, "T3"))
        self.assertEqual(1, len(v))
        self.assertEqual(TOO_SHORT, v[0][3])
        self.assertEqual(10, v[0][5])

    def test_long_enough(self):
        self.assertEqual([], self.check(change("S", "S", 10, "T3")))

    def test_guarantee_holds_below_its_time(self):
        # type 1: the departing vehicle waits, 0 min like nigiri's loader
        self.assertEqual([], self.check(change("S", "S", 2, "T2")))

    def test_forbidden(self):
        v = self.check(change("S", "S", 20, "T4"))
        self.assertEqual(1, len(v))
        self.assertEqual(FORBIDDEN, v[0][3])

    def test_station_rule_reaches_its_stops(self):
        # S2 -> S2 has no row of its own: the station's 15 min apply
        v = self.check(change("S2", "S2", 10, "T3"))
        self.assertEqual(1, len(v))
        self.assertEqual(15, v[0][5])

    def test_tie_is_resolved_in_favor_of_the_journey(self):
        # two trip rules for T1 -> T5, 10 and 3 min: 5 min is not reported
        self.assertEqual([], self.check(change("S", "S2", 5, "T5")))

    def test_bike_access_is_not_a_change_of_vehicles(self):
        # Bike from the door to S, then the train T1 two minutes later. There
        # is no change between two vehicles at S, so the S,S rule does not
        # apply and nothing is violated. Through the command line tool.
        responses = os.path.join(self.dir.name, "responses.json")
        with open(responses, "w") as f:
            f.write(json.dumps({"itineraries": [{"legs": [
                leg("BIKE", None, "t_S",
                    "2019-05-01T08:00:00Z", "2019-05-01T08:28:00Z"),
                leg("REGIONAL_RAIL", "t_S", "t_S9",
                    "2019-05-01T08:30:00Z", "2019-05-01T09:00:00Z",
                    trip="20190501_10:30_t_T1")]}]}) + "\n")
        result = subprocess.run(
            [sys.executable, TOOL, "--gtfs", self.feed, "--tag", "t_",
             responses],
            capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
