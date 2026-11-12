#include "gtest/gtest.h"

#include <optional>
#include <string>

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/init_from.h"

#include "nigiri/rt/gtfsrt_update.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/one_to_all.h"
#include "motis/import.h"

#include "../util.h"

using namespace std::string_view_literals;
using namespace motis;
using namespace date;
using namespace std::chrono_literals;
using namespace test;
namespace n = nigiri;

namespace {

// B is only reachable via T1, C is only reachable via T2. Skipping both stops
// of T1 makes B unreachable on the real-time timetable while the schedule
// still contains the trip.
constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
A,A,49.87260,8.63085,0,,
B,B,50.10701,8.66341,0,,
C,C,49.99359,8.65677,0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,R1,,,101
R2,DB,R2,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,
R2,S1,T2,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,0,0,0
T1,10:30:00,10:30:00,B,1,0,0
T2,10:05:00,10:05:00,A,0,0,0
T2,10:20:00,10:20:00,C,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

std::optional<std::int64_t> duration_to(api::Reachable const& res,
                                        std::string_view const stop_id) {
  if (!res.all_.has_value()) {
    return std::nullopt;
  }
  for (auto const& r : *res.all_) {
    if (r.place_.has_value() && r.place_->stopId_ == stop_id) {
      return r.duration_;
    }
  }
  return std::nullopt;
}

}  // namespace

TEST(motis, one_to_all_realtime_mode) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_one_to_all_rt", ec);

  auto const c = config{.timetable_ = config::timetable{
                            .first_day_ = "2019-05-01",
                            .num_days_ = 2,
                            .datasets_ = {{"test", {.path_ = kGTFS}}}}};
  import(c, "test/data_one_to_all_rt");
  auto d = data{"test/data_one_to_all_rt", c};
  d.init_rtt(date::sys_days{2019_y / May / 1});

  auto const one_to_all_ep = utl::init_from<ep::one_to_all>(d).value();
  // endpoint wired to the same rt_timetable the update below mutates:
  EXPECT_EQ(d.rt_->rtt_.get(), one_to_all_ep.rt_->rtt_.get());

  // 10:00 Europe/Berlin (CEST) = 08:00 UTC, budget covers both trips.
  constexpr auto const kQuery =
      "/api/v1/one-to-all"
      "?one=test_A"
      "&time=2019-05-01T08:00Z"
      "&maxTravelTime=45"sv;

  // Skip both stops of T1 = the trip is cancelled in the real-time timetable.
  auto const stats = n::rt::gtfsrt_update_msg(
      *d.tt_, *d.rt_->rtt_, n::source_idx_t{0}, "test",
      to_feed_msg(
          {trip_update{.trip_ = {.trip_id_ = "T1", .date_ = {"20190501"}},
                       .stop_updates_ = {{.stop_id_ = "A", .skip_ = true},
                                         {.stop_id_ = "B", .skip_ = true}}}},
          date::sys_days{2019_y / May / 1} + 7h));
  EXPECT_EQ(1U, stats.total_entities_success_);

  // OFF and REALTIME_ANNOTATION_ONLY route on the schedule: the cancelled trip
  // still counts.
  for (auto const mode :
       {""sv, "&realtimeMode=OFF"sv, "&realtimeMode=REALTIME_ANNOTATION_ONLY"sv,
        "&realtimeMode=REALTIME"sv}) {
    auto const res = one_to_all_ep(std::string{kQuery} + std::string{mode});

    // Durations include the 2 min default transfer time at the destination.
    EXPECT_EQ(std::optional{std::int64_t{0}}, duration_to(res, "test_A"))
        << "mode=" << mode;
    EXPECT_EQ(std::optional{std::int64_t{22}}, duration_to(res, "test_C"))
        << "mode=" << mode;

    if (mode.empty() || mode == "&realtimeMode=REALTIME"sv) {
      // Only REALTIME (the default) reflects the cancellation of T1.
      EXPECT_EQ(std::nullopt, duration_to(res, "test_B")) << "mode=" << mode;
    } else {
      EXPECT_EQ(std::optional{std::int64_t{32}}, duration_to(res, "test_B"))
          << "mode=" << mode;
    }
  }
}
