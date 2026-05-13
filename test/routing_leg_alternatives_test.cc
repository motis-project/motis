#include "gtest/gtest.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <system_error>

#include "date/date.h"

#include "utl/init_from.h"

#include "motis-api/motis-api.h"
#include "motis/compute_footpaths.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/elevators/elevators.h"
#include "motis/elevators/parse_fasta.h"
#include "motis/endpoints/routing.h"
#include "motis/import.h"
#include "motis/update_rtt_td_footpaths.h"

using namespace date;

using namespace std::string_view_literals;
using namespace motis;
namespace n = nigiri;

// Three-leg journey A → B → C → D:
//   leg 1 (route R1):  T1   09:00 → 09:15  (chosen by router)
//     earlier alternatives:
//       T1_E1          08:30 → 08:45
//       T1_E2          08:00 → 08:15
//     duplicate (parallel route R1_DUP):
//       T1_DUP         09:00 → 09:15
//   leg 2 (route R2):  T2   09:30 → 10:00  (chosen by router)
//     duplicate (parallel route R2_DUP):
//       T2_DUP         09:30 → 10:00  (same dep+arr, different trip)
//     one later alternative (parallel route R2_LATE):
//       T2_LATE        09:45 → 10:15
//   leg 3 (route R3):  T3   10:30 → 11:00  (chosen by router)
//     later alternatives:
//       T3_L1          11:00 → 11:30
//       T3_L2          11:30 → 12:00
//
// The duplicate trip on the middle leg's parallel route also produces a
// journey-level duplicate itinerary (same start/end time, different trip
// in the middle).
//
// Stop coordinates lie within `test/resources/test_case.osm.pbf` so the
// intermodal-bike variant can use OSR pre/post transit routing.
constexpr auto const kThreeLegGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,Station A,49.87336,8.62926,0,
B,Station B,49.99359,8.65677,0,
C,Station C,50.10593,8.66118,0,
D,Station D,50.11403,8.67835,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,R1,,,3
R1_DUP,DB,R1d,,,3
R2,DB,R2,,,3
R2_DUP,DB,R2d,,,3
R2_LATE,DB,R2l,,,3
R3,DB,R3,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,bikes_allowed
R1,S,T1_E2,,1
R1,S,T1_E1,,1
R1,S,T1,,1
R1_DUP,S,T1_DUP,,1
R2,S,T2,,1
R2_DUP,S,T2_DUP,,1
R2_LATE,S,T2_LATE,,1
R3,S,T3,,1
R3,S,T3_L1,,1
R3,S,T3_L2,,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1_E2,08:00:00,08:00:00,A,0
T1_E2,08:15:00,08:15:00,B,1
T1_E1,08:30:00,08:30:00,A,0
T1_E1,08:45:00,08:45:00,B,1
T1,09:00:00,09:00:00,A,0
T1,09:15:00,09:15:00,B,1
T1_DUP,09:00:00,09:00:00,A,0
T1_DUP,09:15:00,09:15:00,B,1
T2,09:30:00,09:30:00,B,0
T2,10:00:00,10:00:00,C,1
T2_DUP,09:30:00,09:30:00,B,0
T2_DUP,10:00:00,10:00:00,C,1
T2_LATE,09:45:00,09:45:00,B,0
T2_LATE,10:15:00,10:15:00,C,1
T3,10:30:00,10:30:00,C,0
T3,11:00:00,11:00:00,D,1
T3_L1,11:00:00,11:00:00,C,0
T3_L1,11:30:00,11:30:00,D,1
T3_L2,11:30:00,11:30:00,C,0
T3_L2,12:00:00,12:00:00,D,1

# calendar_dates.txt
service_id,date,exception_type
S,20190501,1
)"sv;

namespace {

bool is_transit(api::Leg const& l) { return l.displayName_.has_value(); }

// Strip the `YYYYMMDD_HH:MM_<dataset>_` prefix the timetable importer
// prepends to every trip id so the rendered string stays compact.
std::string short_trip_id(std::string const& tid) {
  static constexpr auto kSep = std::string_view{"_test_"};
  auto const pos = tid.find(kSep);
  return pos == std::string::npos ? tid : tid.substr(pos + kSep.size());
}

void print_time(std::ostream& out, openapi::offset_time const& t) {
  out << date::format("%H:%M", *t);
}

void print_leg(std::ostream& out, api::Leg const& l) {
  out << l.mode_;
  if (is_transit(l)) {
    out << ' ' << short_trip_id(l.tripId_.value_or(""));
  }
  out << ' ';
  print_time(out, l.startTime_);
  out << "->";
  print_time(out, l.endTime_);
  if (is_transit(l) && l.interlineWithPreviousLeg_.value_or(false)) {
    out << " [interlined]";
  }
}

std::string to_str(api::Itinerary const& it) {
  auto out = std::stringstream{};
  // Leading newline so the raw-string literals in tests can put each
  // transit leg on its own line (visually aligned under `R"(`).
  out << '\n';
  for (auto const& leg : it.legs_) {
    if (!is_transit(leg)) {
      continue;
    }
    print_leg(out, leg);
    out << '\n';
    if (!leg.alternatives_.has_value()) {
      out << "  (no alternatives field)\n";
      continue;
    }
    if (leg.alternatives_->empty()) {
      out << "  (no alternatives)\n";
      continue;
    }
    for (auto const& alt : *leg.alternatives_) {
      out << "  alt [";
      auto first = true;
      for (auto const& al : alt) {
        if (!first) {
          out << " | ";
        }
        first = false;
        print_leg(out, al);
      }
      out << "]\n";
    }
  }
  return out.str();
}

config make_base_config(bool const with_osr) {
  auto c = config{
      .server_ = {{.web_folder_ = "ui/build", .n_threads_ = 1U}},
      .timetable_ = config::timetable{
          .first_day_ = "2019-05-01",
          .num_days_ = 2,
          .datasets_ = {{"test", {.path_ = std::string{kThreeLegGTFS}}}}}};
  if (with_osr) {
    c.osm_ = "test/resources/test_case.osm.pbf";
    c.street_routing_ = true;
  }
  return c;
}

config make_inline_gtfs_config(std::string_view const gtfs) {
  return config{.server_ = {{.web_folder_ = "ui/build", .n_threads_ = 1U}},
                .timetable_ = config::timetable{
                    .first_day_ = "2019-05-01",
                    .num_days_ = 2,
                    .datasets_ = {{"test", {.path_ = std::string{gtfs}}}}}};
}

}  // namespace

// Station-to-station query, OSR not loaded — match mode kEquivalent,
// access offsets reduce to {stop, 0, 0}.
TEST(motis, routing_leg_alternatives_station_to_station_no_osr) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_leg_alts_no_osr", ec);

  auto const c = make_base_config(/*with_osr=*/false);
  import(c, "test/data_leg_alts_no_osr");
  auto d = data{"test/data_leg_alts_no_osr", c};
  auto const routing = utl::init_from<ep::routing>(d).value();

  auto const res = routing(
      "?fromPlace=test_A"
      "&toPlace=test_D"
      "&time=2019-05-01T05:00Z"
      "&searchWindow=7200"
      "&numLegAlternatives=5");

  ASSERT_EQ(res.itineraries_.size(), 1U);
  EXPECT_EQ(R"(
BUS T1_DUP 07:00->07:15
  alt [WALK 07:00->07:00 | BUS T1 07:00->07:15 | WALK 07:15->07:15]
  alt [WALK 06:30->06:30 | BUS T1_E1 06:30->06:45 | WALK 06:45->06:45]
  alt [WALK 06:00->06:00 | BUS T1_E2 06:00->06:15 | WALK 06:15->06:15]
BUS T2_DUP 07:30->08:00
  alt [WALK 07:30->07:30 | BUS T2 07:30->08:00 | WALK 08:00->08:00]
  alt [WALK 07:45->07:45 | BUS T2_LATE 07:45->08:15 | WALK 08:15->08:15]
BUS T3 08:30->09:00
  alt [WALK 09:00->09:00 | BUS T3_L1 09:00->09:30 | WALK 09:30->09:30]
  alt [WALK 09:30->09:30 | BUS T3_L2 09:30->10:00 | WALK 10:00->10:00]
)",
            to_str(res.itineraries_.front()));
}

// Station-to-station query, OSR loaded but pre/postTransitModes empty —
// `get_offsets` returns nothing for the access side; the `tt_location`
// fallback inside `routing::get_offsets` still seeds the query with all
// kEquivalent stops at offset 0, so alternatives should still be found.
TEST(motis, routing_leg_alternatives_station_to_station_osr_no_pre_post) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_leg_alts_osr_no_pre_post", ec);

  auto const c = make_base_config(/*with_osr=*/true);
  import(c, "test/data_leg_alts_osr_no_pre_post");
  auto d = data{"test/data_leg_alts_osr_no_pre_post", c};
  auto const routing = utl::init_from<ep::routing>(d).value();

  auto const res = routing(
      "?fromPlace=test_A"
      "&toPlace=test_D"
      "&time=2019-05-01T05:00Z"
      "&searchWindow=7200"
      "&preTransitModes="
      "&postTransitModes="
      "&numLegAlternatives=5");

  ASSERT_EQ(res.itineraries_.size(), 1U);
  // Open boundary on this query is intermodal (kIntermodal — OSR
  // loaded), so the 0-min offset legs at journey origin/destination
  // are dropped by `drop_at_boundary`. Inner-boundary footpaths stay.
  EXPECT_EQ(R"(
BUS T1_DUP 07:00->07:15
  alt [BUS T1 07:00->07:15 | WALK 07:15->07:15]
  alt [BUS T1_E1 06:30->06:45 | WALK 06:45->06:45]
  alt [BUS T1_E2 06:00->06:15 | WALK 06:15->06:15]
BUS T2_DUP 07:30->08:00
  alt [WALK 07:30->07:30 | BUS T2 07:30->08:00 | WALK 08:00->08:00]
  alt [WALK 07:45->07:45 | BUS T2_LATE 07:45->08:15 | WALK 08:15->08:15]
BUS T3 08:30->09:00
  alt [WALK 09:00->09:00 | BUS T3_L1 09:00->09:30]
  alt [WALK 09:30->09:30 | BUS T3_L2 09:30->10:00]
)",
            to_str(res.itineraries_.front()));
}

// Intermodal coordinate-to-coordinate query with BIKE pre/post-transit.
// FIXME: this test is currently expected to FAIL — `get_leg_alternatives`
// in nigiri rewrites the synthetic ingress/egress legs of every
// alternative to plain `n::footpath`s, losing the original transport
// mode. The intermodal-aware ingress of the first transit leg and the
// egress of the last transit leg should preserve BIKE (matching the
// surrounding journey), but right now they come back as WALK.
TEST(motis, routing_leg_alternatives_intermodal_bike) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_leg_alts_intermodal_bike", ec);

  auto const c = make_base_config(/*with_osr=*/true);
  import(c, "test/data_leg_alts_intermodal_bike");
  auto d = data{"test/data_leg_alts_intermodal_bike", c};
  auto const routing = utl::init_from<ep::routing>(d).value();

  auto const res = routing(
      "?fromPlace=49.87526849014631,8.62771903392948"
      "&toPlace=50.11347,8.67664"
      "&time=2019-05-01T05:00Z"
      "&searchWindow=7200"
      "&preTransitModes=BIKE"
      "&postTransitModes=BIKE"
      "&requireBikeTransport=true"
      "&numLegAlternatives=5");

  ASSERT_EQ(res.itineraries_.size(), 1U);
  // FIXME: the first leg's first alt-leg and the last leg's third
  // alt-leg are expected to be `BIKE` (matching the surrounding
  // journey's intermodal pre/post-transit access). The leg-alternative
  // code currently rewrites every alternative ingress/egress to a
  // plain footpath, returning `WALK` instead — so this test is
  // intentionally failing until the rewrite preserves the original
  // mode on the "open" boundary of first / last legs.
  EXPECT_EQ(R"(
BUS T1_DUP 07:00->07:15
  alt [BIKE 06:58->07:00 | BUS T1 07:00->07:15 | WALK 07:15->07:15]
  alt [BIKE 06:28->06:30 | BUS T1_E1 06:30->06:45 | WALK 06:45->06:45]
  alt [BIKE 05:58->06:00 | BUS T1_E2 06:00->06:15 | WALK 06:15->06:15]
BUS T2_DUP 07:30->08:00
  alt [WALK 07:30->07:30 | BUS T2 07:30->08:00 | WALK 08:00->08:00]
  alt [WALK 07:45->07:45 | BUS T2_LATE 07:45->08:15 | WALK 08:15->08:15]
BUS T3 08:30->09:00
  alt [WALK 09:00->09:00 | BUS T3_L1 09:00->09:30 | BIKE 09:30->09:32]
  alt [WALK 09:30->09:30 | BUS T3_L2 09:30->10:00 | BIKE 10:00->10:02]
)",
            to_str(res.itineraries_.front()));
}

// Block-id-concatenated trip: T_PART1 (A → MID) and T_PART2 (MID → B)
// share the same block_id, so a passenger stays on board across both.
// nigiri models this as a single transit leg; motis splits it into two
// api legs when `joinInterlinedLegs=false`.
//
//   Block trip (block BLK):
//     T_PART1   A   09:00 → MID 09:15
//     T_PART2   MID 09:15 → B   09:30
//   Earlier alternatives for the block leg (parallel route R_ALT):
//     T_ALT_E1  A   08:30 → B   08:45
//     T_ALT_E2  A   08:00 → B   08:15
//   Second leg:
//     T2        B   09:45 → D   10:15
//     T2_L      B   10:15 → D   10:45  (later alternative)
constexpr auto const kBlockIdGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,Station A,49.87000,8.62000,0,
MID,Station MID,49.93000,8.64000,0,
B,Station B,49.99000,8.65000,0,
D,Station D,50.10000,8.66000,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R_BLOCK_A,DB,RB_A,,,3
R_BLOCK_B,DB,RB_B,,,3
R_ALT,DB,RA,,,3
R2,DB,R2,,,3
R2_LATE,DB,R2L,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R_BLOCK_A,S,T_PART1,,BLK
R_BLOCK_B,S,T_PART2,,BLK
R_ALT,S,T_ALT_E1,,
R_ALT,S,T_ALT_E2,,
R2,S,T2,,
R2_LATE,S,T2_L,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T_PART1,09:00:00,09:00:00,A,0
T_PART1,09:15:00,09:15:00,MID,1
T_PART2,09:15:00,09:15:00,MID,0
T_PART2,09:30:00,09:30:00,B,1
T_ALT_E1,08:30:00,08:30:00,A,0
T_ALT_E1,08:45:00,08:45:00,B,1
T_ALT_E2,08:00:00,08:00:00,A,0
T_ALT_E2,08:15:00,08:15:00,B,1
T2,09:45:00,09:45:00,B,0
T2,10:15:00,10:15:00,D,1
T2_L,10:15:00,10:15:00,B,0
T2_L,10:45:00,10:45:00,D,1

# calendar_dates.txt
service_id,date,exception_type
S,20190501,1
)"sv;

// Like kBlockIdGTFS but with two additional alternatives covering the
// same A → B segment at the same time as the original block trip:
//   * `T_DUP_PART1 + T_DUP_PART2` — a duplicate block-chained trip on
//     parallel routes (block BLK_DUP). When `joinInterlinedLegs=false`
//     this alternative is itself rendered as two interlined api legs.
//   * `T_SINGLE_DUP` — a single trip covering the whole A → B segment
//     alone. Always rendered as one transit leg.
//
// To avoid the new alternatives dominating the router's chosen journey
// (which would change the parent-leg shape away from the block trip),
// the original block trip departs five minutes earlier and a slightly
// earlier R2 trip is added so the block + T2_EARLY combo strictly
// dominates the duplicates on `start_time`.
constexpr auto const kBlockIdInterlinedAltsGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,Station A,49.87000,8.62000,0,
MID,Station MID,49.93000,8.64000,0,
B,Station B,49.99000,8.65000,0,
D,Station D,50.10000,8.66000,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R_BLOCK_A,DB,RB_A,,,3
R_BLOCK_B,DB,RB_B,,,3
R_BLOCK_A_DUP,DB,RB_A2,,,3
R_BLOCK_B_DUP,DB,RB_B2,,,3
R_SINGLE_DUP,DB,RS,,,3
R_SINGLE_DUP_2,DB,RS2,,,3
R2_EARLY,DB,R2E,,,3
R2,DB,R2,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R_BLOCK_A,S,T_PART1,,BLK
R_BLOCK_B,S,T_PART2,,BLK
R_BLOCK_A_DUP,S,T_DUP_PART1,,BLK_DUP
R_BLOCK_B_DUP,S,T_DUP_PART2,,BLK_DUP
R_SINGLE_DUP,S,T_SINGLE_DUP,,
R_SINGLE_DUP_2,S,T_SINGLE_DUP_2,,
R2_EARLY,S,T2_EARLY,,
R2,S,T2,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T_PART1,08:55:00,08:55:00,A,0
T_PART1,09:10:00,09:10:00,MID,1
T_PART2,09:10:00,09:10:00,MID,0
T_PART2,09:25:00,09:25:00,B,1
T_DUP_PART1,09:00:00,09:00:00,A,0
T_DUP_PART1,09:15:00,09:15:00,MID,1
T_DUP_PART2,09:15:00,09:15:00,MID,0
T_DUP_PART2,09:30:00,09:30:00,B,1
T_SINGLE_DUP,09:00:00,09:00:00,A,0
T_SINGLE_DUP,09:30:00,09:30:00,B,1
T_SINGLE_DUP_2,09:05:00,09:05:00,A,0
T_SINGLE_DUP_2,09:35:00,09:35:00,B,1
T2_EARLY,09:40:00,09:40:00,B,0
T2_EARLY,10:10:00,10:10:00,D,1
T2,09:45:00,09:45:00,B,0
T2,10:15:00,10:15:00,D,1

# calendar_dates.txt
service_id,date,exception_type
S,20190501,1
)"sv;

// With `joinInterlinedLegs=true` the block trip is rendered as a single
// transit leg A → B and behaves like any other leg: alternatives are
// populated and the user sees the earlier alternatives T_ALT_E1 /
// T_ALT_E2.
TEST(motis, routing_leg_alternatives_block_id_joined) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_leg_alts_block_joined", ec);

  auto const c = make_inline_gtfs_config(kBlockIdGTFS);
  import(c, "test/data_leg_alts_block_joined");
  auto d = data{"test/data_leg_alts_block_joined", c};
  auto const routing = utl::init_from<ep::routing>(d).value();

  auto const res = routing(
      "?fromPlace=test_A"
      "&toPlace=test_D"
      "&time=2019-05-01T05:00Z"
      "&searchWindow=7200"
      "&joinInterlinedLegs=true"
      "&numLegAlternatives=5");

  ASSERT_EQ(res.itineraries_.size(), 1U);
  EXPECT_EQ(R"(
BUS T_PART1 07:00->07:30
  alt [WALK 06:30->06:30 | BUS T_ALT_E1 06:30->06:45 | WALK 06:45->06:45]
  alt [WALK 06:00->06:00 | BUS T_ALT_E2 06:00->06:15 | WALK 06:15->06:15]
BUS T2 07:45->08:15
  alt [WALK 08:15->08:15 | BUS T2_L 08:15->08:45 | WALK 08:45->08:45]
)",
            to_str(res.itineraries_.front()));
}

// With `joinInterlinedLegs=false` the block trip is rendered as two
// interlined api legs (T_PART1 + T_PART2, the latter with
// `interlineWithPreviousLeg=true`). Alternatives are computed against
// the underlying single transit segment, so only the **main** leg
// (T_PART1) carries `alternatives` — the secondary interlined leg's
// `alternatives` field stays unset.
TEST(motis, routing_leg_alternatives_block_id_unjoined) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_leg_alts_block_unjoined", ec);

  auto const c = make_inline_gtfs_config(kBlockIdGTFS);
  import(c, "test/data_leg_alts_block_unjoined");
  auto d = data{"test/data_leg_alts_block_unjoined", c};
  auto const routing = utl::init_from<ep::routing>(d).value();

  auto const res = routing(
      "?fromPlace=test_A"
      "&toPlace=test_D"
      "&time=2019-05-01T05:00Z"
      "&searchWindow=7200"
      "&joinInterlinedLegs=false"
      "&numLegAlternatives=5");

  ASSERT_EQ(res.itineraries_.size(), 1U);
  EXPECT_EQ(R"(
BUS T_PART1 07:00->07:15
  alt [WALK 06:30->06:30 | BUS T_ALT_E1 06:30->06:45 | WALK 06:45->06:45]
  alt [WALK 06:00->06:00 | BUS T_ALT_E2 06:00->06:15 | WALK 06:15->06:15]
BUS T_PART2 07:15->07:30 [interlined]
  (no alternatives field)
BUS T2 07:45->08:15
  alt [WALK 08:15->08:15 | BUS T2_L 08:15->08:45 | WALK 08:45->08:45]
)",
            to_str(res.itineraries_.front()));
}

// Alternatives themselves can be interlined: when an alternative covers
// the same A→B segment via a parallel block-chained trip, it is itself
// rendered through journey_to_response with the request's
// `joinInterlinedLegs` setting. With `joinInterlinedLegs=false` the
// alternative therefore comes back as 4 api legs (ingress, transit_a,
// transit_b interlined, egress); a single-trip alternative covering the
// same segment stays at 3 api legs.
//
// Setup ensures both shapes coexist among the main block leg's
// alternatives:
//   - duplicate block trip (T_DUP_PART1 + T_DUP_PART2, block BLK_DUP)
//     → interlined alternative (4 legs, second transit leg carries
//       interlineWithPreviousLeg=true)
//   - single-trip duplicate (T_SINGLE_DUP) covering A→B alone
//     → non-interlined alternative (3 legs)
TEST(motis, routing_leg_alternatives_block_id_interlined_alternatives) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_leg_alts_block_alt_shapes", ec);

  auto const c = make_inline_gtfs_config(kBlockIdInterlinedAltsGTFS);
  import(c, "test/data_leg_alts_block_alt_shapes");
  auto d = data{"test/data_leg_alts_block_alt_shapes", c};
  auto const routing = utl::init_from<ep::routing>(d).value();

  auto const res = routing(
      "?fromPlace=test_A"
      "&toPlace=test_D"
      "&time=2019-05-01T05:00Z"
      "&searchWindow=7200"
      "&joinInterlinedLegs=false"
      "&numLegAlternatives=5");

  ASSERT_EQ(res.itineraries_.size(), 1U);
  EXPECT_EQ(R"(
BUS T_SINGLE_DUP_2 07:05->07:35
  alt [WALK 07:00->07:00 | BUS T_DUP_PART1 07:00->07:15 | BUS T_DUP_PART2 07:15->07:30 [interlined] | WALK 07:30->07:30]
  alt [WALK 07:00->07:00 | BUS T_SINGLE_DUP 07:00->07:30 | WALK 07:30->07:30]
  alt [WALK 06:55->06:55 | BUS T_PART1 06:55->07:10 | BUS T_PART2 07:10->07:25 [interlined] | WALK 07:25->07:25]
BUS T2_EARLY 07:40->08:10
  alt [WALK 07:45->07:45 | BUS T2 07:45->08:15 | WALK 08:15->08:15]
)",
            to_str(res.itineraries_.front()));
}

// Two complementary scenarios on the same fixture, each exercising a
// different td-footpath code path in `direct.cc`:
//
//   Scenario A — DA → FFM (transfer):
//     ICE DA_10 → FFM_10, transfer (td) FFM_10 → FFM_101,
//     S3 FFM_101 → FFM_HAUPT_S, walk to FFM_HAUPT. The transfer at FFM
//     uses the wheelchair-routed footpath whose live duration is
//     materialised into `rt_timetable::td_footpaths_*` by
//     `update_rtt_td_footpaths`. Tests the timetable-internal td
//     branch in `lookup_footpath`.
//
//   Scenario B — coord → coord inside FFM (intermodal):
//     A wheelchair user near FFM_101 boards S3 directly. The boarding
//     boundary is open / intermodal, so the only "footpath" is the
//     query-level offset; td blocking has to flow through
//     `q.td_start_` and be honoured by `lookup_offset`. Tests the
//     intermodal td-offset branch.
//
// In both cases the FFM HBF Gleis 101/102 elevator is out of service
// 01:30–03:30 UTC. The expected outputs below mirror raptor 1:1: for
// boarding times whose direct walk window falls inside the outage,
// `get_td_duration` returns the inflated "leave early and wait on the
// platform" duration — the alt is shown but its implied walk spans
// arrival-by-train through to boarding. Improving the journey
// rendering to surface that wait as a gap (rather than embedding it
// in the WALK leg) is future work.
constexpr auto const kElevatorBlockedFasta = R"__([
  {
    "description": "FFM HBF zu Gleis 101/102 (S-Bahn)",
    "equipmentnumber": 10561326,
    "geocoordX": 8.6628995,
    "geocoordY": 50.1072933,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1866,
    "type": "ELEVATOR",
    "outOfService": [
      ["2019-05-01T01:30:00Z", "2019-05-01T03:30:00Z"]
    ]
  }
])__"sv;

constexpr auto const kElevatorGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
DA,DA Hbf,49.87260,8.63085,1,,
DA_10,DA Hbf,49.87336,8.62926,0,DA,10
FFM,FFM Hbf,50.10701,8.66341,1,,
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10
FFM_101,FFM Hbf,50.10739,8.66333,0,FFM,101
FFM_HAUPT,FFM Hauptwache,50.11403,8.67835,1,,
FFM_HAUPT_S,FFM Hauptwache S,50.11404,8.67824,0,FFM_HAUPT,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
ICE,DB,ICE,,,101
S3,DB,S3,,,109

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
ICE,S1,ICE,,
S3,S1,S3,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
ICE,00:35:00,00:35:00,DA_10,0,0,0
ICE,00:45:00,00:45:00,FFM_10,1,0,0
S3,01:15:00,01:15:00,FFM_101,1,0,0
S3,01:20:00,01:20:00,FFM_HAUPT_S,2,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# frequencies.txt
trip_id,start_time,end_time,headway_secs
ICE,00:35:00,06:35:00,3600
S3,01:15:00,06:15:00,3600
)"sv;

TEST(motis, routing_leg_alternatives_td_footpath_blocked) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_leg_alts_td_footpath_blocked", ec);

  // Mirrors the elevator-test config from `routing_test.cc`: same OSM
  // tile, `use_osm_stop_coordinates_ = true`, `osr_footpath_ = true`
  // and `extend_missing_footpaths_ = false` so the wheelchair route
  // FFM_10 → FFM_101 is the elevator-routed footpath rather than an
  // auto-generated direct walk. Without this the elevator outage
  // wouldn't actually block anything.
  auto const c = config{
      .server_ = {{.web_folder_ = "ui/build", .n_threads_ = 1U}},
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .tiles_ = {{.profile_ = "deps/tiles/profile/full.lua",
                  .db_size_ = 1024U * 1024U * 25U}},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .use_osm_stop_coordinates_ = true,
              .extend_missing_footpaths_ = false,
              .datasets_ = {{"test", {.path_ = std::string{kElevatorGTFS}}}}},
      .street_routing_ = true,
      .osr_footpath_ = true};
  import(c, "test/data_leg_alts_td_footpath_blocked");
  auto d = data{"test/data_leg_alts_td_footpath_blocked", c};
  d.rt_->e_ = std::make_unique<elevators>(*d.w_, nullptr, *d.elevator_nodes_,
                                          parse_fasta(kElevatorBlockedFasta));
  d.init_rtt(date::sys_days{2019_y / May / 1});

  // Rebuild the rt_timetable's td_footpaths from the elevator state.
  // Mirrors what `update_elevators` does in production: the
  // elevator-footpath map (one entry per OSR-routed footpath that
  // traverses an elevator node, written to disk by `import()`) tells
  // `update_rtt_td_footpaths` which (from, to) stop pairs to refresh
  // for the wheelchair profile.
  auto const elevator_footpath_map = cista::read<elevator_footpath_map_t>(
      "test/data_leg_alts_td_footpath_blocked/elevator_footpath_map.bin");
  update_rtt_td_footpaths(
      *d.w_, *d.l_, *d.pl_, *d.tt_, *d.location_rtree_, *d.rt_->e_,
      *elevator_footpath_map, *d.matches_, *d.rt_->rtt_,
      std::chrono::seconds{c.timetable_->max_footpath_length_ * 60});

  auto const routing = utl::init_from<ep::routing>(d).value();

  // === Scenario A: DA → FFM_HAUPT (transfer / rt_timetable td) ===
  // The chosen wheelchair journey transfers at FFM via the elevator-
  // routed td footpath FFM_10 → FFM_101. ICE/S3 alternatives whose
  // transfer would land in the 01:30–03:30 outage window must NOT
  // appear, because the rt_timetable's td_footpath at that time has
  // duration kMaxDuration.
  auto const res_a = routing(
      "?fromPlace=test_DA"
      "&toPlace=test_FFM_HAUPT"
      "&time=2019-05-01T00:30Z"
      "&pedestrianProfile=WHEELCHAIR"
      "&useRoutedTransfers=true"
      "&numLegAlternatives=5");
  ASSERT_FALSE(res_a.itineraries_.empty());
  EXPECT_EQ(R"(
HIGHSPEED_RAIL ICE 00:35->00:45
  alt [WALK 23:28->23:35 | HIGHSPEED_RAIL ICE 23:35->23:45 | WALK 23:45->23:51]
  alt [WALK 22:28->22:35 | HIGHSPEED_RAIL ICE 22:35->22:45 | WALK 22:45->22:51]
METRO S3 01:15->01:20
  alt [WALK 01:24->02:15 | METRO S3 02:15->02:20 | WALK 02:20->02:21]
  alt [WALK 01:24->03:15 | METRO S3 03:15->03:20 | WALK 03:20->03:21]
)",
            to_str(res_a.itineraries_.front()));

  // === Scenario B: coord → coord inside FFM (intermodal td_start_) ===
  auto const res_b = routing(
      "?fromPlace=50.1040763,8.6586978"  // near FFM_101
      "&toPlace=50.1132737,8.6767235"  // near FFM_HAUPT_S
      "&time=2019-05-01T01:15Z"
      "&pedestrianProfile=WHEELCHAIR"
      "&maxMatchingDistance=8"
      "&useRoutedTransfers=true"
      "&numLegAlternatives=5");

  ASSERT_FALSE(res_b.itineraries_.empty());
  EXPECT_EQ(R"(
METRO S3 02:15->02:20
  alt [WALK 01:16->03:15 | METRO S3 03:15->03:20 | WALK 03:20->03:29]
)",
            to_str(res_b.itineraries_.front()));
}

// Two-leg journey A → B → C where the first transit leg has three
// candidate "earlier" alternatives on parallel routes:
//   T1_EARLY     07:00 → 07:30   (boards at A, alights at B, all open)
//   T1_NOENTER   07:30 → 08:00   (`pickup_type=1` at A — no boarding)
//   T1_NOEXIT    07:45 → 08:15   (`drop_off_type=1` at B — no alighting)
//
// `T1_NOENTER` and `T1_NOEXIT` must not show up as leg-alternatives for
// T1: a passenger can't board on the no-pickup trip, and an alt that
// can't be alighted at the matching stop is useless. `T1_EARLY` is the
// only valid earlier alternative. direct.cc enforces this via
// `for_each_pair`'s `can_start` / `can_finish` checks (which read the
// per-stop in_allowed / out_allowed flags) — these tests guard that
// behaviour.
constexpr auto const kPickupDropoffGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,Station A,49.87336,8.62926,0,
B,Station B,49.99359,8.65677,0,
C,Station C,50.10593,8.66118,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,R1,,,3
R1_EARLY,DB,R1e,,,3
R1_NOENTER,DB,R1ne,,,3
R1_NOEXIT,DB,R1nx,,,3
R2,DB,R2,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign
R1,S,T1,
R1_EARLY,S,T1_EARLY,
R1_NOENTER,S,T1_NOENTER,
R1_NOEXIT,S,T1_NOEXIT,
R2,S,T2,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,08:00:00,08:00:00,A,0,0,0
T1,08:30:00,08:30:00,B,1,0,0
T1_EARLY,07:00:00,07:00:00,A,0,0,0
T1_EARLY,07:30:00,07:30:00,B,1,0,0
T1_NOENTER,07:30:00,07:30:00,A,0,1,0
T1_NOENTER,08:00:00,08:00:00,B,1,0,0
T1_NOEXIT,07:45:00,07:45:00,A,0,0,0
T1_NOEXIT,08:15:00,08:15:00,B,1,0,1
T2,09:00:00,09:00:00,B,0,0,0
T2,09:30:00,09:30:00,C,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S,20190501,1
)"sv;

TEST(motis, routing_leg_alternatives_no_enter_excluded) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_leg_alts_no_enter", ec);

  auto const c = make_inline_gtfs_config(kPickupDropoffGTFS);
  import(c, "test/data_leg_alts_no_enter");
  auto d = data{"test/data_leg_alts_no_enter", c};
  auto const routing = utl::init_from<ep::routing>(d).value();

  auto const res = routing(
      "?fromPlace=test_A"
      "&toPlace=test_C"
      "&time=2019-05-01T05:30Z"
      "&searchWindow=7200"
      "&numLegAlternatives=5");

  ASSERT_EQ(res.itineraries_.size(), 1U);
  // T1_EARLY is the only earlier alternative for T1; T1_NOENTER (no
  // boarding at A) and T1_NOEXIT (no alighting at B) must not appear.
  EXPECT_EQ(R"(
BUS T1 06:00->06:30
  alt [WALK 05:00->05:00 | BUS T1_EARLY 05:00->05:30 | WALK 05:30->05:30]
BUS T2 07:00->07:30
  (no alternatives)
)",
            to_str(res.itineraries_.front()));
}

TEST(motis, routing_leg_alternatives_no_exit_excluded) {
  // Same fixture / GTFS as `routing_leg_alternatives_no_enter_excluded`
  // — T1_NOEXIT is the trip exercised here, it shares the assertion
  // above but is split into its own test for clarity / reporting.
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_leg_alts_no_exit", ec);

  auto const c = make_inline_gtfs_config(kPickupDropoffGTFS);
  import(c, "test/data_leg_alts_no_exit");
  auto d = data{"test/data_leg_alts_no_exit", c};
  auto const routing = utl::init_from<ep::routing>(d).value();

  auto const res = routing(
      "?fromPlace=test_A"
      "&toPlace=test_C"
      "&time=2019-05-01T05:30Z"
      "&searchWindow=7200"
      "&numLegAlternatives=5");

  ASSERT_EQ(res.itineraries_.size(), 1U);
  // Identical expectation to the no-enter case: only T1_EARLY shows up.
  // T1_NOEXIT is excluded because alighting at B is forbidden on that
  // trip.
  EXPECT_EQ(R"(
BUS T1 06:00->06:30
  alt [WALK 05:00->05:00 | BUS T1_EARLY 05:00->05:30 | WALK 05:30->05:30]
BUS T2 07:00->07:30
  (no alternatives)
)",
            to_str(res.itineraries_.front()));
}

// Two-leg journey A → B → C with a 0-minute GTFS `transfers.txt`
// SELF-transfer at B (B → B with min_transfer_time=0). raptor honours
// such explicit 0-min self-transfers and represents them in the
// reconstructed journey; the leg-alternative path should do the same
// instead of squashing them as "synthetic same-stop dummy walks".
//
// The router picks T2_EARLY (08:45) as the main second leg because
// it arrives earliest. T2 (09:00) becomes the alternative, boarding
// at the same stop B as T1's alighting — i.e. the alt's preceding
// "walk" is the GTFS-defined 0-min self-transfer at B.
//
// direct.cc emits all footpath-shaped walks (synthetic same-stop
// dummies AND real GTFS self-transfers) and only drops 0-duration
// `offset` legs at the journey boundary — same filter as
// `journey_to_response.cc` applies to raptor-emitted journeys. So
// the self-transfer's WALK and the surrounding synthetic same-stop
// dummies (alighting at the journey destination) all flow through.
constexpr auto const kZeroMinSelfTransferGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,Station A,49.87336,8.62926,0,
B,Station B,49.99359,8.65677,0,
C,Station C,50.10593,8.66118,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,R1,,,3
R2,DB,R2,,,3
R2_EARLY,DB,R2e,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign
R1,S,T1,
R2,S,T2,
R2_EARLY,S,T2_EARLY,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,08:00:00,08:00:00,A,0
T1,08:30:00,08:30:00,B,1
T2,09:00:00,09:00:00,B,0
T2,09:30:00,09:30:00,C,1
T2_EARLY,08:45:00,08:45:00,B,0
T2_EARLY,09:15:00,09:15:00,C,1

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
B,B,2,0

# calendar_dates.txt
service_id,date,exception_type
S,20190501,1
)"sv;

TEST(motis, routing_leg_alternatives_zero_minute_self_transfer_preserved) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_leg_alts_zero_min_self_transfer", ec);

  auto const c = make_inline_gtfs_config(kZeroMinSelfTransferGTFS);
  import(c, "test/data_leg_alts_zero_min_self_transfer");
  auto d = data{"test/data_leg_alts_zero_min_self_transfer", c};
  auto const routing = utl::init_from<ep::routing>(d).value();

  auto const res = routing(
      "?fromPlace=test_A"
      "&toPlace=test_C"
      "&time=2019-05-01T05:30Z"
      "&searchWindow=7200"
      "&numLegAlternatives=5");

  ASSERT_EQ(res.itineraries_.size(), 1U);
  // Router picks T2_EARLY as main; T2 is the alternative reached via
  // the 0-min self-transfer at B. The expected leading WALK is the
  // GTFS-defined self-transfer — must appear even though `from == to`
  // and `dur == 0`.
  EXPECT_EQ(R"(
BUS T1 06:00->06:30
  (no alternatives)
BUS T2_EARLY 06:45->07:15
  alt [WALK 07:00->07:00 | BUS T2 07:00->07:30 | WALK 07:30->07:30]
)",
            to_str(res.itineraries_.front()));
}
