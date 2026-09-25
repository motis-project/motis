#include "gtest/gtest.h"

#include <filesystem>
#include <map>
#include <optional>
#include <string>

#include "utl/helpers/algorithm.h"
#include "utl/init_from.h"

#include "nigiri/routing/for_each_hub_source.h"
#include "nigiri/timetable.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/map/routes.h"
#include "motis/endpoints/map/stops.h"
#include "motis/endpoints/one_to_all.h"
#include "motis/endpoints/routing.h"
#include "motis/endpoints/stop_times.h"
#include "motis/endpoints/transfers.h"
#include "motis/import.h"
#include "motis/itinerary_id.h"

// Regression tests for motis-side defects found in the t2t-rt review
// (virtual locations created by transfers.txt rules). Every test states the
// correct behaviour, so it fails while the defect is present.

using namespace motis;
using namespace date;
namespace n = nigiri;

namespace {

// FA (RF1) is split off to a virtual location below U by the RF1 -> RF3 rule;
// FB and FB2 (RF2) leave from U itself. The RB trips at Z are split off by
// the one-sided "arriving on RB" rule; the RB2 trips stay at Z.
constexpr auto const kPlainGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
U,U,55.0,13.0,0,
L,L,55.1,13.0,0,
M,M,55.2,13.0,0,
Z,Z,52.0,10.0,0,
F,F,52.2,10.0,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
RF1,DB,RF1,,3
RF2,DB,RF2,,3
RF3,DB,RF3,,3
RB,DB,RB,,3
RB2,DB,RB2,,3

# trips.txt
route_id,service_id,trip_id
RF1,S1,FA
RF2,S1,FB
RF2,S1,FB2
RB,S1,TB2
RB,S1,TB3
RB2,S1,TB5
RB2,S1,TB6

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
FA,10:00:00,10:00:00,L,0
FA,10:30:00,10:30:00,U,1
FB,10:33:00,10:33:00,U,0
FB,11:00:00,11:00:00,M,1
FB2,10:50:00,10:50:00,U,0
FB2,11:20:00,11:20:00,M,1
TB2,12:35:00,12:35:00,Z,0
TB2,13:00:00,13:00:00,F,1
TB3,12:45:00,12:45:00,Z,0
TB3,13:15:00,13:15:00,F,1
TB5,12:36:00,12:36:00,Z,0
TB5,13:01:00,13:01:00,F,1
TB6,12:46:00,12:46:00,Z,0
TB6,13:16:00,13:16:00,F,1

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time,from_route_id,to_route_id,from_trip_id,to_trip_id
U,U,2,120,,,,
U,U,2,300,RF1,RF3,,
Z,Z,2,120,,,,
Z,Z,2,600,RB,,,

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

// With OSM and osr_footpath:
//  - V: the RV2 departures are split off by a 10 min rule, so V has a stored
//    rule footpath to their virtual location (outside the OSM extract).
//  - P1, P2: two platforms of station P at the same coordinate, outside the
//    OSM extract, so the router finds no walk and the default profile gets a
//    beeline estimate.
//  - CA, CA2, CB: stations inside the extract that the car profile connects
//    (the coordinates of itinerary_id_test's car transfer). CA and CA2 share
//    a coordinate; both car-carrying trips at CA are split off by a trip
//    rule, the ones at CA2 and CB are not.
constexpr auto const kOsmGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
V,V,60.0,30.0,0,
VA,VA,60.1,30.0,0,
VB,VB,60.2,30.0,0,
P,P,61.0,31.0,1,
P1,P 1,61.0,31.0,0,P
P2,P 2,61.0,31.0,0,P
PA,PA,61.1,31.0,0,
CA,CA,50.10739,8.66333,1,
CA2,CA2,50.10739,8.66333,1,
CB,CB,50.10593,8.66118,1,
CX,CX,49.5,8.3,0,
CY,CY,49.5,8.4,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
RV1,DB,RV1,,3
RV2,DB,RV2,,3
RP1,DB,RP1,,3
RP2,DB,RP2,,3
RC1,DB,RC1,,3
RC2,DB,RC2,,3
RC9,DB,RC9,,3
RC3,DB,RC3,,3

# trips.txt
route_id,service_id,trip_id,cars_allowed
RV1,S1,VT1,0
RV2,S1,VT2,0
RP1,S1,PT1,0
RP2,S1,PT2,0
RC1,S1,CT1,1
RC9,S1,CT9,1
RC2,S1,CT2,1
RC3,S1,CT3,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
VT1,10:00:00,10:00:00,VA,0
VT1,10:30:00,10:30:00,V,1
VT2,10:45:00,10:45:00,V,0
VT2,11:00:00,11:00:00,VB,1
PT1,10:00:00,10:00:00,PA,0
PT1,10:30:00,10:30:00,P1,1
PT2,10:40:00,10:40:00,P2,0
PT2,11:00:00,11:00:00,PA,1
CT1,10:00:00,10:00:00,CX,0
CT1,10:30:00,10:30:00,CA,1
CT9,10:40:00,10:40:00,CA,0
CT9,11:00:00,11:00:00,CY,1
CT2,10:45:00,10:45:00,CB,0
CT2,11:10:00,11:10:00,CY,1
CT3,10:00:00,10:00:00,CX,0
CT3,10:30:00,10:30:00,CA2,1

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time,from_route_id,to_route_id,from_trip_id,to_trip_id
V,V,2,120,,,,
V,V,2,600,,RV2,,
CA,CA,2,120,,,,
CA,CA,2,300,,,CT1,CT9

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

struct fixture {
  config c_;
  std::optional<data> d_{};
};

data& load(fixture& f, char const* dir) {
  auto ec = std::error_code{};
  std::filesystem::remove_all(dir, ec);
  import(f.c_, dir);
  f.d_.emplace(dir, f.c_);
  f.d_->init_rtt(date::sys_days{2019_y / May / 1});
  return *f.d_;
}

data& plain() {
  static auto f = fixture{
      .c_ = config{.timetable_ = config::timetable{
                       .first_day_ = "2019-05-01",
                       .num_days_ = 2,
                       .datasets_ = {{"test", {.path_ = kPlainGTFS}}}}}};
  static auto& d = load(f, "test/data/review_t2t_plain");
  return d;
}

data& with_osm() {
  static auto f = fixture{
      .c_ = config{
          .osm_ = {"test/resources/test_case.osm.pbf"},
          .timetable_ =
              config::timetable{.first_day_ = "2019-05-01",
                                .num_days_ = 2,
                                .datasets_ = {{"test", {.path_ = kOsmGTFS}}}},
          .street_routing_ = true,
          .osr_footpath_ = true}};
  static auto& d = load(f, "test/data/review_t2t_osm");
  return d;
}

n::location_idx_t lidx(data const& d, char const* id) {
  return d.tt_->locations_.location_id_to_idx_.at({id, n::source_idx_t{0U}});
}

// how often each stop id occurs
std::map<std::string, unsigned> count_ids(auto const& places, auto&& get) {
  auto counts = std::map<std::string, unsigned>{};
  for (auto const& x : places) {
    auto const& p = get(x);
    if (p.stopId_.has_value()) {
      ++counts[*p.stopId_];
    }
  }
  return counts;
}

api::Itinerary plan_l_to_m(data& d, std::string const& extra = "") {
  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const res = routing(
      "?fromPlace=test_L&toPlace=test_M&time=2019-05-01T07:55:00Z"
      "&timetableView=false" +
      extra);
  EXPECT_FALSE(res.itineraries_.empty());
  return res.itineraries_.empty() ? api::Itinerary{} : res.itineraries_.front();
}

}  // namespace

// ===========================================================================
// Location loops: a virtual location is its stop outside of the routing.
// ===========================================================================

TEST(t2t_review, one_to_all_lists_each_stop_once) {
  auto& d = plain();
  auto const one_to_all = utl::init_from<ep::one_to_all>(d).value();
  auto const res = one_to_all(
      "/api/v6/one-to-all?one=test_L&time=2019-05-01T07:55:00Z"
      "&maxTravelTime=90");
  ASSERT_TRUE(res.all_.has_value());
  auto const counts = count_ids(
      *res.all_, [](api::ReachablePlace const& p) -> api::Place const& {
        return *p.place_;
      });
  ASSERT_TRUE(counts.contains("test_U")) << "precondition";
  for (auto const& [id, n] : counts) {
    EXPECT_EQ(1U, n) << id;
  }
}

TEST(t2t_review, map_stops_lists_each_stop_once) {
  auto& d = plain();
  auto const stops = utl::init_from<ep::stops>(d).value();
  auto const res = stops("/api/v1/map/stops?min=54.9%2C12.9&max=55.3%2C13.1");
  auto const counts = count_ids(
      res, [](api::Place const& p) -> api::Place const& { return p; });
  ASSERT_TRUE(counts.contains("test_U")) << "precondition";
  for (auto const& [id, n] : counts) {
    EXPECT_EQ(1U, n) << id;
  }
}

TEST(t2t_review, map_routes_lists_each_stop_once) {
  auto& d = plain();
  auto const routes = utl::init_from<ep::routes>(d).value();
  auto const res = routes(
      "/api/experimental/map/routes?max=55.3%2C13.1&min=54.9%2C12.9"
      "&zoom=16");
  auto const counts = count_ids(
      res.stops_, [](api::Place const& p) -> api::Place const& { return p; });
  ASSERT_TRUE(counts.contains("test_U")) << "precondition";
  for (auto const& [id, n] : counts) {
    EXPECT_EQ(1U, n) << id;
  }
}

// exactRadius: the stop itself, but that includes the trips a rule moved to
// its virtual locations (TB2, TB3) - not only the ones left at Z (TB5, TB6).
TEST(t2t_review, stop_times_exact_radius_lists_moved_departures) {
  auto& d = plain();
  auto const stop_times = utl::init_from<ep::stop_times>(d).value();
  auto const res = stop_times(
      "/api/v5/stoptimes?stopId=test_Z&time=2019-05-01T10:00:00Z&n=10"
      "&exactRadius=true");
  EXPECT_EQ(4U, res.stopTimes_.size());
}

// ===========================================================================
// Transfers that exist only through a hub: FA's virtual location -> U.
// ===========================================================================

TEST(t2t_review, refresh_itinerary_after_split_off_trip) {
  auto& d = plain();
  auto const original = plan_l_to_m(d);
  ASSERT_FALSE(original.legs_.empty());

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const stop_times = utl::init_from<ep::stop_times>(d).value();
  auto const refreshed =
      reconstruct_itinerary(routing, stop_times, *d.rt_, original.id_);
  for (auto const& l : refreshed.legs_) {
    EXPECT_FALSE(l.cancelled_.value_or(false))
        << l.from_.stopId_.value_or("?") << " -> "
        << l.to_.stopId_.value_or("?");
  }
  EXPECT_EQ(original, refreshed);
}

TEST(t2t_review, leg_alternatives_after_split_off_trip) {
  auto& d = plain();
  auto const it = plan_l_to_m(d, "&numLegAlternatives=3");
  auto const fb = utl::find_if(it.legs_, [](api::Leg const& l) {
    return l.tripId_.has_value() && l.tripId_->ends_with("_FB");
  });
  ASSERT_NE(end(it.legs_), fb) << "precondition";
  ASSERT_TRUE(fb->alternatives_.has_value());
  EXPECT_FALSE(fb->alternatives_->empty());  // FB2
}

// ===========================================================================
// With OSM and osr_footpath.
// ===========================================================================

// The debug transfers endpoint lists stops only: CA is split by the
// CT1 -> CT9 rule, its virtual locations are no transfer targets of their own.
TEST(t2t_review, debug_transfers_name_their_targets) {
  auto& d = with_osm();
  auto const transfers = utl::init_from<ep::transfers>(d).value();
  auto const res = transfers("/api/debug/transfers?id=test_CA");
  ASSERT_TRUE(utl::any_of(d.tt_->locations_.children_[lidx(d, "CA")],
                          [&](n::location_idx_t const c) {
                            return d.tt_->locations_.types_[c] ==
                                   n::location_type::kVirt;
                          }))
      << "precondition: CA has virtual locations";
  ASSERT_FALSE(res.transfers_.empty()) << "precondition";
  for (auto const& t : res.transfers_) {
    EXPECT_TRUE(t.to_.stopId_.has_value() && !t.to_.stopId_->empty())
        << t.to_.name_;
    EXPECT_FALSE(t.to_.name_.empty());
  }
}

// The default profile walks on routed footpaths now. Changing platforms can
// not be faster than changing at the platform itself (2 min): P1 and P2 share
// a coordinate and the router cannot connect them, so the estimate is 0 min.
TEST(t2t_review, default_profile_walk_keeps_change_time_floor) {
  auto& d = with_osm();
  auto const& tt = *d.tt_;
  auto const p1 = lidx(d, "P1");
  auto const p2 = lidx(d, "P2");

  auto best = std::optional<int>{};
  auto const take = [&](n::location_idx_t const l, n::duration_t const dur) {
    if (l == p2 && (!best.has_value() || dur.count() < *best)) {
      best = dur.count();
    }
  };
  for (auto const fp : tt.locations_.footpaths_out_[n::kDefaultProfile][p1]) {
    take(fp.target(), fp.duration());
  }
  n::routing::for_each_hub_source<n::direction::kBackward>(
      tt, n::kDefaultProfile, p1, [&](n::footpath const fp) {
        take(fp.target(), fp.duration());
        return true;
      });
  ASSERT_TRUE(best.has_value()) << "precondition: P1 -> P2 walkable";
  EXPECT_GE(*best, tt.locations_.transfer_time_[p1].count());
}

// Both car-carrying trips at CA are split off by a trip rule. CA is still a
// stop cars use, so the car profile has to connect it to CB - like CA2.
TEST(t2t_review, car_profile_connects_stops_whose_trips_were_split) {
  auto& d = with_osm();
  auto const& tt = *d.tt_;
  auto const ca = lidx(d, "CA");
  auto const cb = lidx(d, "CB");
  ASSERT_FALSE(tt.locations_.footpaths_out_[n::kCarProfile].empty())
      << "precondition: car profile computed";
  // control: CA2 (same coordinate, car trips not split) is connected to CB
  ASSERT_TRUE(
      utl::any_of(tt.locations_.footpaths_out_[n::kCarProfile][lidx(d, "CA2")],
                  [&](n::footpath const fp) { return fp.target() == cb; }))
      << "precondition: car routing works here";
  EXPECT_TRUE(
      utl::any_of(tt.locations_.footpaths_out_[n::kCarProfile][ca],
                  [&](n::footpath const fp) { return fp.target() == cb; }));
}
