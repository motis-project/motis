#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/helpers/algorithm.h"
#include "utl/init_from.h"

#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/timetable.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/gtfsrt.h"
#include "motis/endpoints/map/routes.h"
#include "motis/endpoints/map/stops.h"
#include "motis/endpoints/one_to_all.h"
#include "motis/endpoints/routing.h"
#include "motis/endpoints/stop.h"
#include "motis/endpoints/stop_times.h"
#include "motis/endpoints/transfers.h"
#include "motis/import.h"
#include "motis/itinerary_id.h"

#include "./util.h"

// Endpoints on timetables with virtual locations (created by transfers.txt
// rules): outside of the routing, a virtual location is its stop.

using namespace motis;
using namespace date;
using namespace std::chrono_literals;
namespace n = nigiri;

// FA (RF1) stops at a virtual location below U (the RF1 -> RF3 rule); FB and
// FB2 (RF2) leave from U itself. The RB trips stop at a virtual location of Z
// (the rule qualified on its from side only: arriving on RB); the RB2 trips
// stay at Z.
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

// With OSM and osr_footpath: CA, CA2, CB are stations inside the extract that
// the car profile connects (the coordinates of itinerary_id_test's car
// transfer). CA and CA2 share a coordinate; both car-carrying trips at CA are
// at virtual locations (a trip rule), the ones at CA2 and CB are not.
constexpr auto const kOsmGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
CA,CA,50.10739,8.66333,1,
CA2,CA2,50.10739,8.66333,1,
CB,CB,50.10593,8.66118,1,
CX,CX,49.5,8.3,0,
CY,CY,49.5,8.4,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
RC1,DB,RC1,,3
RC2,DB,RC2,,3
RC9,DB,RC9,,3
RC3,DB,RC3,,3

# trips.txt
route_id,service_id,trip_id,cars_allowed
RC1,S1,CT1,1
RC9,S1,CT9,1
RC2,S1,CT2,1
RC3,S1,CT3,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
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
  static auto& d = load(f, "test/data/virtual_locations_plain");
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
  static auto& d = load(f, "test/data/virtual_locations_osm");
  return d;
}

// The trip-qualified transfers.txt row differs from the unqualified one of the
// stop pair, so trip A stops at a virtual location below S1 (and B below S2).
// Real time: A arrives at S1 5 min late.
constexpr auto const kStationGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
X,X,49.80000,8.60000,0,,
S,S Hbf,49.87260,8.63085,1,,
S1,S Hbf,49.87265,8.63085,0,S,1
S2,S Hbf,49.87255,8.63085,0,S,2
Y,Y,49.95000,8.70000,0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RA,DB,A,,,3
RB,DB,B,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RA,S1,A,,
RB,S1,B,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
A,10:00:00,10:00:00,X,0,0,0
A,10:10:00,10:10:00,S1,1,0,0
B,10:30:00,10:30:00,S2,0,0,0
B,10:40:00,10:40:00,Y,1,0,0

# transfers.txt
from_stop_id,to_stop_id,from_trip_id,to_trip_id,transfer_type,min_transfer_time
S1,S2,,,2,180
S1,S2,A,B,2,600

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

data& delay_a_at_s1(data& d) {
  n::rt::gtfsrt_update_msg(
      *d.tt_, *d.rt_->rtt_, n::source_idx_t{0}, "test",
      test::to_feed_msg(
          {test::trip_update{.trip_ = {.trip_id_ = "A",
                                       .start_time_ = {"10:00:00"},
                                       .date_ = {"20190501"}},
                             .stop_updates_ = {{.stop_id_ = "S1",
                                                .seq_ = std::optional{1U},
                                                .ev_type_ = n::event_type::kArr,
                                                .delay_minutes_ = 5}}}},
          date::sys_days{2019_y / May / 1} + 7h));
  return d;
}

data& station() {
  static auto f = fixture{
      .c_ = config{.timetable_ = config::timetable{
                       .first_day_ = "2019-05-01",
                       .num_days_ = 2,
                       .datasets_ = {{"test", {.path_ = kStationGTFS}}}}}};
  static auto& d =
      delay_a_at_s1(load(f, "test/data/virtual_locations_station"));
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

api::Itinerary plan_x_to_y(data& d) {
  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const res = routing(
      "?fromPlace=test_X&toPlace=test_Y&time=2019-05-01T07:55:00Z"
      "&timetableView=false");
  EXPECT_FALSE(res.itineraries_.empty());
  return res.itineraries_.empty() ? api::Itinerary{} : res.itineraries_.front();
}

// ===========================================================================
// Stop ids: a virtual location has none, its stop's id stands for it.
// ===========================================================================

TEST(virtual_locations, itinerary_names_the_stops) {
  auto& d = station();
  ASSERT_NE(0U, test::n_virts(*d.tt_)) << "precondition";
  auto const it = plan_x_to_y(d);
  for (auto const& l : it.legs_) {
    for (auto const* p : {&l.from_, &l.to_}) {
      ASSERT_TRUE(p->stopId_.has_value());
      EXPECT_FALSE(p->stopId_->ends_with('_')) << *p->stopId_;
    }
  }
  ASSERT_EQ(3U, it.legs_.size());
  EXPECT_EQ("test_S1", it.legs_[0].to_.stopId_);
  EXPECT_EQ("test_S", it.legs_[0].to_.parentId_);
  EXPECT_EQ("1", it.legs_[0].to_.track_);
  EXPECT_EQ("test_S2", it.legs_[2].from_.stopId_);

  // the delay at S1, where A stops at a virtual location, arrived
  EXPECT_TRUE(it.legs_[0].realTime_);
  EXPECT_EQ("2019-05-01 08:15",
            date::format("%F %H:%M", *it.legs_[0].to_.arrival_.value()));
  EXPECT_EQ(
      "2019-05-01 08:10",
      date::format("%F %H:%M", *it.legs_[0].to_.scheduledArrival_.value()));
}

TEST(virtual_locations, stop_times_list_the_stops_of_the_station) {
  auto& d = station();
  auto const stop_times = utl::init_from<ep::stop_times>(d).value();
  auto const res = stop_times(
      "/api/v5/stoptimes?stopId=test_S1&time=2019-05-01T07:55:00Z&n=3");
  EXPECT_EQ("test_S1", res.place_.stopId_);
  ASSERT_FALSE(res.stopTimes_.empty());
  for (auto const& st : res.stopTimes_) {
    // departures at the other stop of S (S2) are listed too
    EXPECT_TRUE(st.place_.stopId_ == "test_S1" ||
                st.place_.stopId_ == "test_S2")
        << *st.place_.stopId_;
    EXPECT_FALSE(st.tripFrom_.stopId_->ends_with('_'));
    EXPECT_FALSE(st.tripTo_.stopId_->ends_with('_'));
  }
}

// with the real-time update at the virtual location
TEST(virtual_locations, itinerary_with_real_time_is_found_again) {
  auto& d = station();
  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const stop_times = utl::init_from<ep::stop_times>(d).value();
  auto const original = plan_x_to_y(d);
  EXPECT_EQ(original,
            reconstruct_itinerary(routing, stop_times, *d.rt_, original.id_));
}

TEST(virtual_locations, stop_routes_include_trips_at_virtual_locations) {
  auto& d = station();
  auto const stop = utl::init_from<ep::stop>(d).value();
  auto const route_ids_at = [&](char const* stop_id) {
    auto ids = std::set<std::string>{};
    for (auto const& r :
         stop(std::string{"/api/v6/stop?stopId="} + stop_id).routes_) {
      ids.insert(r.routeId_);
    }
    return ids;
  };
  EXPECT_TRUE(route_ids_at("test_S1").contains("test_RA"));
  EXPECT_TRUE(route_ids_at("test_S2").contains("test_RB"));
}

// the GTFS-RT export names the stop, not the (id-less) virtual location
TEST(virtual_locations, gtfsrt_export_names_the_stop) {
  auto& d = station();
  auto const gtfsrt = utl::init_from<ep::gtfsrt>(d).value();
  auto const reply =
      gtfsrt(net::route_request{net::web_server::http_req_t{}}, false);
  auto msg = transit_realtime::FeedMessage{};
  ASSERT_TRUE(msg.ParseFromString(
      std::get<net::web_server::string_res_t>(reply).body()));
  auto n_stop_updates = 0U;
  for (auto const& e : msg.entity()) {
    for (auto const& stu : e.trip_update().stop_time_update()) {
      EXPECT_FALSE(stu.stop_id().empty());
      ++n_stop_updates;
    }
  }
  EXPECT_NE(0U, n_stop_updates);
}

// ===========================================================================
// Location loops: a virtual location is its stop outside of the routing.
// ===========================================================================

TEST(virtual_locations, one_to_all_lists_each_stop_once) {
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

TEST(virtual_locations, map_stops_lists_each_stop_once) {
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

TEST(virtual_locations, map_routes_lists_each_stop_once) {
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
TEST(virtual_locations, stop_times_exact_radius_lists_moved_departures) {
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

TEST(virtual_locations, refresh_itinerary_through_virtual_location) {
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

TEST(virtual_locations, leg_alternatives_after_trip_at_virtual_location) {
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

// The debug transfers endpoint lists stops only: CA has virtual locations (the
// CT1 -> CT9 rule), they are no transfer targets of their own.
TEST(virtual_locations, debug_transfers_name_their_targets) {
  auto& d = with_osm();
  auto const transfers = utl::init_from<ep::transfers>(d).value();
  auto const res = transfers("/api/debug/transfers?id=test_CA");
  ASSERT_TRUE(utl::any_of(
      d.tt_->locations_.children_[lidx(d, "CA")],
      [&](n::location_idx_t const c) { return d.tt_->locations_.is_virt(c); }))
      << "precondition: CA has virtual locations";
  ASSERT_FALSE(res.transfers_.empty()) << "precondition";
  for (auto const& t : res.transfers_) {
    EXPECT_TRUE(t.to_.stopId_.has_value() && !t.to_.stopId_->empty())
        << t.to_.name_;
    EXPECT_FALSE(t.to_.name_.empty());
  }
}

// Both car-carrying trips at CA stop at virtual locations (a trip rule). CA is
// still a stop cars use, so the car profile has to connect it to CB - like CA2.
TEST(virtual_locations, car_profile_connects_stops_with_virtual_locations) {
  auto& d = with_osm();
  auto const& tt = *d.tt_;
  auto const ca = lidx(d, "CA");
  auto const cb = lidx(d, "CB");
  ASSERT_FALSE(tt.locations_.footpaths_out_[n::kCarProfile].empty())
      << "precondition: car profile computed";
  // control: CA2 (same coordinate, no virtual locations) is connected to CB
  ASSERT_TRUE(
      utl::any_of(tt.locations_.footpaths_out_[n::kCarProfile][lidx(d, "CA2")],
                  [&](n::footpath const fp) { return fp.target() == cb; }))
      << "precondition: car routing works here";
  EXPECT_TRUE(
      utl::any_of(tt.locations_.footpaths_out_[n::kCarProfile][ca],
                  [&](n::footpath const fp) { return fp.target() == cb; }));
}
