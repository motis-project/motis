#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>
#include <set>

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/init_from.h"

#include "nigiri/rt/gtfsrt_update.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/gtfsrt.h"
#include "motis/endpoints/routing.h"
#include "motis/endpoints/stop.h"
#include "motis/endpoints/stop_times.h"
#include "motis/import.h"
#include "motis/itinerary_id.h"

#include "../util.h"

using namespace motis;
using namespace date;
using namespace std::chrono_literals;
using namespace test;
namespace n = nigiri;

namespace {

// The trip-qualified transfers.txt row differs from the default of the stop
// pair, so trip A stops at a virtual location below S1 (and B below S2).
constexpr auto const kGTFS = R"(
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

void expect_stop_ids(api::Itinerary const& it) {
  for (auto const& l : it.legs_) {
    for (auto const* p : {&l.from_, &l.to_}) {
      ASSERT_TRUE(p->stopId_.has_value());
      EXPECT_FALSE(p->stopId_->ends_with('_')) << *p->stopId_;
    }
  }
}

}  // namespace

TEST(motis, transfer_rule_stop_ids) {
  auto const dir = std::filesystem::path{"test/data/transfer_rule_stop_ids"};
  auto ec = std::error_code{};
  std::filesystem::remove_all(dir, ec);

  auto const c = config{.timetable_ = config::timetable{
                            .first_day_ = "2019-05-01",
                            .num_days_ = 2,
                            .datasets_ = {{"test", {.path_ = kGTFS}}}}};
  import(c, dir);
  auto d = data{dir, c};
  d.init_rtt(date::sys_days{2019_y / May / 1});

  // precondition: the rule did split off virtual locations
  auto n_virts = 0U;
  for (auto const t : d.tt_->locations_.types_) {
    n_virts += t == n::location_type::kVirt;
  }
  ASSERT_NE(0U, n_virts);

  auto const plan = [&]() {
    auto const routing = utl::init_from<ep::routing>(d).value();
    return routing(
        "?fromPlace=test_X&toPlace=test_Y"
        "&time=2019-05-01T07:55:00Z&timetableView=false");
  };

  {  // static
    auto const res = plan();
    ASSERT_FALSE(res.itineraries_.empty());
    auto const& it = res.itineraries_.front();
    expect_stop_ids(it);
    ASSERT_EQ(3U, it.legs_.size());
    EXPECT_EQ("test_S1", it.legs_[0].to_.stopId_);
    EXPECT_EQ("test_S", it.legs_[0].to_.parentId_);
    EXPECT_EQ("1", it.legs_[0].to_.track_);
    EXPECT_EQ("test_S2", it.legs_[2].from_.stopId_);
  }

  // real-time: a delay that names the platform of a rule-bound stop
  auto const stats = n::rt::gtfsrt_update_msg(
      *d.tt_, *d.rt_->rtt_, n::source_idx_t{0}, "test",
      to_feed_msg(
          {trip_update{.trip_ = {.trip_id_ = "A",
                                 .start_time_ = {"10:00:00"},
                                 .date_ = {"20190501"}},
                       .stop_updates_ = {{.stop_id_ = "S1",
                                          .seq_ = std::optional{1U},
                                          .ev_type_ = n::event_type::kArr,
                                          .delay_minutes_ = 5}}}},
          date::sys_days{2019_y / May / 1} + 7h));
  EXPECT_EQ(1U, stats.total_entities_success_);

  {
    auto const res = plan();
    ASSERT_FALSE(res.itineraries_.empty());
    auto const& it = res.itineraries_.front();
    expect_stop_ids(it);
    EXPECT_EQ("test_S1", it.legs_[0].to_.stopId_);
    EXPECT_TRUE(it.legs_[0].realTime_);
    EXPECT_EQ("2019-05-01 08:15",
              date::format("%F %H:%M", *it.legs_[0].to_.arrival_.value()));
    EXPECT_EQ(
        "2019-05-01 08:10",
        date::format("%F %H:%M", *it.legs_[0].to_.scheduledArrival_.value()));
  }

  {
    auto const stop_times = utl::init_from<ep::stop_times>(d).value();
    auto const res = stop_times(
        "/api/v5/stoptimes?stopId=test_S1"
        "&time=2019-05-01T07:55:00Z&n=3");
    EXPECT_EQ("test_S1", res.place_.stopId_);
    ASSERT_FALSE(res.stopTimes_.empty());
    for (auto const& st : res.stopTimes_) {
      // departures of the sibling platform are listed too
      EXPECT_TRUE(st.place_.stopId_ == "test_S1" ||
                  st.place_.stopId_ == "test_S2")
          << *st.place_.stopId_;
      EXPECT_FALSE(st.tripFrom_.stopId_->ends_with('_'));
      EXPECT_FALSE(st.tripTo_.stopId_->ends_with('_'));
    }
  }

  {  // an itinerary through rule-bound stops is found again by its id
    auto const routing = utl::init_from<ep::routing>(d).value();
    auto const stop_times = utl::init_from<ep::stop_times>(d).value();
    auto const original = plan().itineraries_.front();
    EXPECT_EQ(original,
              reconstruct_itinerary(routing, stop_times, *d.rt_, original.id_));
  }

  {  // the routes of a stop include those that stop at its virtual locations
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

  {  // the GTFS-RT export names the stop, not the (id-less) virtual location
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
}
