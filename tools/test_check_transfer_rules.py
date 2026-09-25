#!/usr/bin/env python3

"""Regression test for check-transfer-rules.py, from the t2t-rt review.

Run with `python3 tools/test_check_transfer_rules.py` (or pytest). The test
states the correct behaviour, so it fails while the defect is present.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "check-transfer-rules.py")

FEED = {
    "stops.txt": "stop_id,stop_name,stop_lat,stop_lon,parent_station\n"
                 "S,S,50.0,8.0,\n"
                 "S9,S9,50.1,8.0,\n",
    "trips.txt": "route_id,service_id,trip_id\n"
                 "R1,X,T1\n",
    # every change of vehicles at S takes 10 minutes
    "transfers.txt": "from_stop_id,to_stop_id,transfer_type,min_transfer_time\n"
                     "S,S,2,600\n",
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


class CheckTransferRulesTest(unittest.TestCase):
    def run_tool(self, itinerary):
        with tempfile.TemporaryDirectory() as d:
            feed = os.path.join(d, "feed")
            os.mkdir(feed)
            for name, content in FEED.items():
                with open(os.path.join(feed, name), "w") as f:
                    f.write(content)
            responses = os.path.join(d, "responses.json")
            with open(responses, "w") as f:
                f.write(json.dumps({"itineraries": [itinerary]}) + "\n")
            return subprocess.run(
                [sys.executable, TOOL, "--gtfs", feed, "--tag", "t_",
                 responses],
                capture_output=True, text=True)

    def test_bike_access_is_not_a_change_of_vehicles(self):
        # Bike from the door to S, then the train T1 two minutes later. There
        # is no change between two vehicles at S, so the S,S rule does not
        # apply and nothing is violated.
        result = self.run_tool({"legs": [
            leg("BIKE", None, "t_S",
                "2019-05-01T08:00:00Z", "2019-05-01T08:28:00Z"),
            leg("REGIONAL_RAIL", "t_S", "t_S9",
                "2019-05-01T08:30:00Z", "2019-05-01T09:00:00Z",
                trip="20190501_10:30_t_T1")]})
        self.assertEqual(0, result.returncode,
                         result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
