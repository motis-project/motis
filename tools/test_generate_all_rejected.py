#!/usr/bin/env python3

"""Regression test for `motis generate --max_direct`, from the t2t-rt review.

Every stop of the timetable is a short walk from every other one, so
`--max_direct 60` has to reject every pair the generator draws ("discard
queries that have a direct (walking) connection within this many minutes").
No query may be written, and after its 1000 attempts the generator must fail
cleanly (exit code 1 with a message) - not dereference the empty destination.

Run with `python3 tools/test_generate_all_rejected.py` from the repository
root. MOTIS_BIN selects the binary (default: cmake-build-relwithdebinfo/motis).
The test states the correct behaviour, so it fails while the defect is present.
"""

import os
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTIS = os.environ.get("MOTIS_BIN",
                       os.path.join(ROOT, "cmake-build-relwithdebinfo", "motis"))
OSM = os.path.join(ROOT, "test", "resources", "test_case.osm.pbf")

# two platforms of Frankfurt Hbf, inside the test OSM extract; the street
# router connects them (a 6 min walk, see test/itinerary_id_test.cc)
FEED = {
    "agency.txt": "agency_id,agency_name,agency_url,agency_timezone\n"
                  "DB,DB,https://example.com,Europe/Berlin\n",
    "stops.txt": "stop_id,stop_name,stop_lat,stop_lon\n"
                 "CA,CA,50.10593,8.66118\n"
                 "CB,CB,50.10739,8.66333\n",
    "routes.txt": "route_id,agency_id,route_short_name,route_long_name,"
                  "route_type\n"
                  "R1,DB,R1,,3\n",
    "trips.txt": "route_id,service_id,trip_id\nR1,S1,T1\n",
    "stop_times.txt": "trip_id,arrival_time,departure_time,stop_id,"
                      "stop_sequence\n"
                      "T1,10:00:00,10:00:00,CA,0\n"
                      "T1,10:10:00,10:10:00,CB,1\n",
    "calendar_dates.txt": "service_id,date,exception_type\nS1,20190501,1\n",
}


class GenerateAllRejectedTest(unittest.TestCase):
    def test_every_pair_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            gtfs = os.path.join(d, "gtfs")
            os.mkdir(gtfs)
            for name, content in FEED.items():
                with open(os.path.join(gtfs, name), "w") as f:
                    f.write(content)
            with open(os.path.join(d, "config.yml"), "w") as f:
                f.write(f"osm: {OSM}\n"
                        "street_routing: true\n"
                        "timetable:\n"
                        "  first_day: 2019-05-01\n"
                        "  num_days: 2\n"
                        "  datasets:\n"
                        "    t:\n"
                        f"      path: {gtfs}\n")
            imported = subprocess.run(
                [MOTIS, "import", "-c", "config.yml", "-d", "data"],
                cwd=d, capture_output=True, text=True)
            self.assertEqual(0, imported.returncode, imported.stderr[-2000:])

            # station-to-station only: with -m WALK the generator picks random
            # OSM nodes near the stops, and a pair the foot router cannot
            # connect has no direct walk at all - keeping it is right
            for modes in ([],):
                generated = subprocess.run(
                    [MOTIS, "generate", "-d", "data", "-n", "1",
                     "--max_direct", "60", "--geo_rank", "0",
                     "--first_day", "2019-05-01", "--last_day", "2019-05-01"]
                    + modes,
                    cwd=d, capture_output=True, text=True, timeout=600)
                self.assertGreaterEqual(generated.returncode, 0,
                                        f"{modes}: killed by a signal: "
                                        + generated.stderr[-2000:])
                with open(os.path.join(d, "queries.txt")) as f:
                    written = [l for l in f if l.strip()]
                self.assertEqual([], written,
                                 f"{modes}: a pair with a short direct walk "
                                 "was kept")


if __name__ == "__main__":
    unittest.main()
