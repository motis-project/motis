#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/json.hpp"

#include "fmt/format.h"

#include "gtest/gtest.h"

#include "utl/helpers/algorithm.h"
#include "utl/init_from.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/elevators/elevators.h"
#include "motis/elevators/parse_fasta.h"
#include "motis/elevators/update_elevators.h"
#include "motis/endpoints/refresh_itinerary.h"
#include "motis/endpoints/routing.h"
#include "motis/endpoints/stop_times.h"
#include "motis/endpoints/trip.h"
#include "motis/gbfs/update.h"
#include "motis/import.h"
#include "motis/itinerary_id.h"
#include "motis/tag_lookup.h"

#include "net/base64.h"

#include "osr/location.h"

#include "nigiri/routing/journey.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/special_stations.h"

#include "generated/itinerary_id/itinerary_id.pb.h"

#include "./test_case.h"
#include "./util.h"

using namespace motis;
namespace fs = std::filesystem;
namespace n = nigiri;

TEST(motis, itinerary_id_distinguishes_level_zero_from_no_level) {
  auto tt = n::timetable{};
  auto tags = tag_lookup{};

  auto const dep = n::unixtime_t{};
  auto const arr = dep + std::chrono::minutes{1};
  auto j = n::routing::journey{};
  j.legs_.push_back(n::routing::journey::leg{
      n::direction::kForward,
      n::get_special_station(n::special_station::kStart),
      n::get_special_station(n::special_station::kEnd), dep, arr,
      n::routing::offset{n::get_special_station(n::special_station::kEnd),
                         n::duration_t{1}, n::transport_mode_id_t{0}}});

  auto const start =
      place_t{osr::location{geo::latlng{1.0, 2.0}, osr::level_t{0.F}}};
  auto const dest =
      place_t{osr::location{geo::latlng{3.0, 4.0}, osr::kNoLevel}};

  auto parsed = motis::ItineraryId{};
  ASSERT_TRUE(parsed.ParseFromString(net::decode_base64(
      generate_itinerary_id(j, tags, tt, nullptr, nullptr, nullptr, nullptr,
                            nullptr, nullptr, start, dest))));
  ASSERT_EQ(1, parsed.legs_size());
  EXPECT_TRUE(parsed.legs(0).has_from_level());
  EXPECT_EQ(0.0, parsed.legs(0).from_level());
  EXPECT_FALSE(parsed.legs(0).has_to_level());
}

api::RefreshItineraryPostBody make_refresh_itinerary_post_body(
    api::Itinerary const& itinerary) {
  auto const& leg = itinerary.legs_.at(0);
  auto body = api::RefreshItineraryPostBody{};
  body.id_.legs_.push_back(api::LegId{
      .displayName_ = leg.displayName_.value(),
      .tripId_ = leg.tripId_.value(),
      .fromId_ = leg.from_.stopId_.value(),
      .fromLat_ = leg.from_.lat_,
      .fromLon_ = leg.from_.lon_,
      .fromLevel_ = leg.from_.level_,
      .toId_ = leg.to_.stopId_.value(),
      .toLat_ = leg.to_.lat_,
      .toLon_ = leg.to_.lon_,
      .toLevel_ = leg.to_.level_,
      .schedStart_ = leg.scheduledStartTime_.get_unixtime_seconds(),
      .schedEnd_ = leg.scheduledEndTime_.get_unixtime_seconds(),
      .mode_ = leg.mode_,
      .scheduled_ = leg.scheduled_,
  });
  return boost::json::value_to<api::RefreshItineraryPostBody>(
      boost::json::value_from(body));
}

constexpr auto kSimpleGtfsTemplate = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
{},DA Hbf,49.87260,8.63085,1,,
{},FFM Hbf,50.10701,8.66341,1,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
ICE,DB,ICE,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
ICE,S1,ICE,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
ICE,10:00:00,10:35:00,{},0,0,0
ICE,11:00:00,11:00:00,{},1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

constexpr auto kLoopGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
A,Stop A,49.00000,8.00000,1,,
B,Stop B,49.10000,8.10000,1,,
C,Stop C,49.20000,8.20000,1,,
D,Stop D,49.30000,8.30000,1,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
LOOP,DB,LOOP,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
LOOP,S1,LOOP,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
LOOP,10:00:00,10:00:00,A,0,0,0
LOOP,10:15:00,10:15:00,B,1,0,0
LOOP,10:30:00,10:30:00,C,2,0,0
LOOP,10:45:00,10:45:00,B,3,0,0
LOOP,11:03:00,11:04:00,D,4,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

constexpr auto kDenseShiftSourceGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
OLD_A,Origin A,49.87260,8.63085,1,,
OLD_B,Origin B,50.10701,8.66341,1,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
MAIN,DB,MAIN,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
MAIN,S1,MAIN,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
MAIN,10:00:00,10:00:00,OLD_A,0,0,0
MAIN,10:20:00,10:20:00,OLD_B,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

constexpr auto kDenseShiftTargetGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
NEW_A,Origin A New,49.87260,8.63085,1,,
A_N1,From Nearby 1,49.87300,8.63085,1,,
A_N2,From Nearby 2,49.87220,8.63100,1,,
A_N3,From Nearby 3,49.87290,8.63140,1,,
A_N4,From Nearby 4,49.87210,8.63020,1,,
NEW_B,Origin B New,50.10701,8.66341,1,,
B_N1,To Nearby 1,50.10735,8.66341,1,,
B_N2,To Nearby 2,50.10670,8.66395,1,,
B_N3,To Nearby 3,50.10745,8.66295,1,,
B_N4,To Nearby 4,50.10660,8.66290,1,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
MATCH_ROUTE,DB,MATCH,,,101
DECOY_ROUTE,DB,DECOY,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
MATCH_ROUTE,S1,MATCH,,
DECOY_ROUTE,S1,DECOY_1,,
DECOY_ROUTE,S1,DECOY_2,,
DECOY_ROUTE,S1,DECOY_3,,
DECOY_ROUTE,S1,DECOY_4,,
DECOY_ROUTE,S1,DECOY_5,,
DECOY_ROUTE,S1,DECOY_6,,
DECOY_ROUTE,S1,DECOY_7,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
MATCH,10:08:00,10:08:00,NEW_A,0,0,0
MATCH,10:28:00,10:28:00,NEW_B,1,0,0
DECOY_1,10:03:00,10:03:00,A_N1,0,0,0
DECOY_1,10:23:00,10:23:00,B_N1,1,0,0
DECOY_2,10:04:00,10:04:00,A_N2,0,0,0
DECOY_2,10:24:00,10:24:00,B_N2,1,0,0
DECOY_3,10:05:00,10:05:00,A_N3,0,0,0
DECOY_3,10:25:00,10:25:00,B_N3,1,0,0
DECOY_4,10:06:00,10:06:00,A_N4,0,0,0
DECOY_4,10:26:00,10:26:00,B_N4,1,0,0
DECOY_5,10:07:00,10:07:00,A_N1,0,0,0
DECOY_5,10:27:00,10:27:00,B_N3,1,0,0
DECOY_6,10:09:00,10:09:00,A_N3,0,0,0
DECOY_6,10:29:00,10:29:00,B_N1,1,0,0
DECOY_7,10:10:00,10:10:00,A_N2,0,0,0
DECOY_7,10:30:00,10:30:00,B_N4,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

constexpr auto kHeavyBenchmarkSourceGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
SRC_A,Source A,49.87260,8.63085,1,,
SRC_B,Source B,50.10701,8.66341,1,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
ICE_ROUTE,DB,ICE,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
ICE_ROUTE,S1,ICE,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
ICE,10:00:00,10:00:00,SRC_A,0,0,0
ICE,10:20:00,10:20:00,SRC_B,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

std::string make_heavy_target_gtfs(std::size_t const n_decoys) {
  auto stops = std::string{
      "\n# stops.txt\n"
      "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,"
      "platform_code\n"
      "MATCH_A,Match A,49.87260,8.63085,1,,\n"
      "MATCH_B,Match B,50.10701,8.66341,1,,\n"};
  auto trips = std::string{
      "\n# trips.txt\n"
      "route_id,service_id,trip_id,trip_headsign,block_id\n"
      "ICE_ROUTE,S1,ICE,,\n"};
  auto stop_times = std::string{
      "\n# stop_times.txt\n"
      "trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,"
      "drop_off_type\n"
      "ICE,10:05:00,10:05:00,MATCH_A,0,0,0\n"
      "ICE,10:25:00,10:25:00,MATCH_B,1,0,0\n"};

  for (auto i = std::size_t{0}; i < n_decoys; ++i) {
    auto const mag_a_lat = static_cast<int>((i * 3U) % 8U) + 2;
    auto const mag_a_lon = static_cast<int>((i * 5U) % 8U) + 2;
    auto const mag_b_lat = static_cast<int>((i * 7U) % 8U) + 2;
    auto const mag_b_lon = static_cast<int>((i * 11U) % 8U) + 2;
    auto const sign_a_lat = (i & 1U) != 0U ? 1 : -1;
    auto const sign_a_lon = (i & 2U) != 0U ? 1 : -1;
    auto const sign_b_lat = (i & 4U) != 0U ? 1 : -1;
    auto const sign_b_lon = (i & 8U) != 0U ? 1 : -1;
    auto const lat_off_a = 0.00005 * mag_a_lat * sign_a_lat;
    auto const lon_off_a = 0.00008 * mag_a_lon * sign_a_lon;
    auto const lat_off_b = 0.00005 * mag_b_lat * sign_b_lat;
    auto const lon_off_b = 0.00008 * mag_b_lon * sign_b_lon;
    auto const rt = static_cast<int>((i * 13U) % 33U);
    auto const time_off_min = rt < 17 ? (rt - 20) : (rt - 12);
    auto const dep_total = 10 * 60 + time_off_min;
    auto const arr_total = dep_total + 20;

    fmt::format_to(std::back_inserter(stops),
                   "DEC_A_{},DA {},{:.5f},{:.5f},1,,\n"
                   "DEC_B_{},DB {},{:.5f},{:.5f},1,,\n",
                   i, i, 49.87260 + lat_off_a, 8.63085 + lon_off_a, i, i,
                   50.10701 + lat_off_b, 8.66341 + lon_off_b);
    fmt::format_to(std::back_inserter(trips), "DECOY_ROUTE,S1,DEC_{},,\n", i);
    fmt::format_to(std::back_inserter(stop_times),
                   "DEC_{},{:02}:{:02}:00,{:02}:{:02}:00,DEC_A_{},0,0,0\n"
                   "DEC_{},{:02}:{:02}:00,{:02}:{:02}:00,DEC_B_{},1,0,0\n",
                   i, dep_total / 60, dep_total % 60, dep_total / 60,
                   dep_total % 60, i, i, arr_total / 60, arr_total % 60,
                   arr_total / 60, arr_total % 60, i);
  }

  return fmt::format(
      "\n# agency.txt\n"
      "agency_id,agency_name,agency_url,agency_timezone\n"
      "DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin\n"
      "{}"
      "\n# routes.txt\n"
      "route_id,agency_id,route_short_name,route_long_name,route_desc,"
      "route_type\n"
      "ICE_ROUTE,DB,ICE,,,101\n"
      "DECOY_ROUTE,DB,DECOY,,,101\n"
      "{}"
      "{}"
      "\n# calendar_dates.txt\n"
      "service_id,date,exception_type\n"
      "S1,20190501,1\n",
      stops, trips, stop_times);
}

config make_config(std::string const& gtfs) {
  return config{
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test", {.path_ = gtfs}}},
          },
  };
}

data import_test_data(config const& cfg, std::string_view const sub_dir) {
  auto const path = fs::path{"test/data/itinerary_id"} / sub_dir;
  auto ec = std::error_code{};
  fs::remove_all(path, ec);
  import(cfg, path);
  return data{path, cfg};
}

api::Itinerary route_first_itinerary(data& data,
                                     std::string_view const from_place,
                                     std::string_view const to_place,
                                     std::string_view const time,
                                     bool const detailed_legs = true,
                                     bool const detailed_transfers = false) {
  auto const routing = utl::init_from<ep::routing>(data).value();
  auto const query = fmt::format(
      "?fromPlace={}&toPlace={}&time={}&timetableView=true"
      "&directModes=WALK,RENTAL&detailedLegs={}&detailedTransfers={}",
      from_place, to_place, time, detailed_legs ? "true" : "false",
      detailed_transfers ? "true" : "false");
  return routing(query).itineraries_.at(0);
}

transit_realtime::FeedMessage make_added_trip_update(
    date::sys_seconds const msg_time) {
  auto msg = transit_realtime::FeedMessage{};
  auto* const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(test::to_unix(msg_time));

  auto* const e = msg.add_entity();
  e->set_id("1");
  auto* const tu = e->mutable_trip_update();
  auto* const td = tu->mutable_trip();
  td->set_trip_id("ADDED_ICE");
  td->set_route_id("ICE");
  td->set_start_date("20190501");
  td->set_start_time("11:00:00");
  td->set_schedule_relationship(
      transit_realtime::TripDescriptor_ScheduleRelationship_ADDED);

  auto* const from_stu = tu->add_stop_time_update();
  from_stu->set_stop_id("DA");
  from_stu->set_stop_sequence(0U);
  from_stu->mutable_departure()->set_time(static_cast<std::int64_t>(
      test::to_unix(msg_time + std::chrono::minutes{2})));

  auto* const to_stu = tu->add_stop_time_update();
  to_stu->set_stop_id("FFM");
  to_stu->set_stop_sequence(1U);
  to_stu->mutable_arrival()->set_time(static_cast<std::int64_t>(
      test::to_unix(msg_time + std::chrono::minutes{17})));

  return msg;
}

TEST(motis, itinerary_id_reconstruct_with_changed_stop_ids) {
  auto const source_cfg = make_config(
      std::string{fmt::format(kSimpleGtfsTemplate, "DA", "FFM", "DA", "FFM")});
  auto source_data = import_test_data(source_cfg, "changed_stop_ids_source");
  auto const original = route_first_itinerary(source_data, "test_DA",
                                              "test_FFM", "2019-05-01T02:00Z");
  auto const& id = original.id_;

  auto const target_cfg = make_config(
      std::string{fmt::format(kSimpleGtfsTemplate, "FFM", "DA", "FFM", "DA")});
  auto target_data = import_test_data(target_cfg, "changed_stop_ids_target");
  auto expected = route_first_itinerary(target_data, "test_FFM", "test_DA",
                                        "2019-05-01T02:00Z");
  expected.id_ = id;
  auto const stop_times = utl::init_from<ep::stop_times>(target_data).value();
  auto const routing = utl::init_from<ep::routing>(target_data).value();

  EXPECT_EQ(expected, reconstruct_itinerary(routing, stop_times, {}, id));
}

TEST(motis, itinerary_id_reconstruct_with_repeated_stop_in_trip) {
  auto const cfg = make_config(kLoopGtfs);
  auto data = import_test_data(cfg, "repeated_stop_route");

  auto const original =
      route_first_itinerary(data, "test_B", "test_D", "2019-05-01T08:00Z");
  ASSERT_EQ(1U, original.legs_.size());

  auto const& id = original.id_;
  auto const stop_times = utl::init_from<ep::stop_times>(data).value();
  auto const routing = utl::init_from<ep::routing>(data).value();
  EXPECT_EQ(original, reconstruct_itinerary(routing, stop_times, {}, id));
}

// The production id is generated from the nigiri journey (not the api legs).
// Verify that this id round-trips through reconstruction.
TEST(motis, itinerary_id_reconstruct_from_production_journey_id) {
  auto const cfg = make_config(
      std::string{fmt::format(kSimpleGtfsTemplate, "DA", "FFM", "DA", "FFM")});
  auto data = import_test_data(cfg, "production_journey_id");

  auto const original =
      route_first_itinerary(data, "test_DA", "test_FFM", "2019-05-01T02:00Z");
  ASSERT_FALSE(original.id_.empty());

  auto const stop_times = utl::init_from<ep::stop_times>(data).value();
  auto const routing = utl::init_from<ep::routing>(data).value();
  EXPECT_EQ(original,
            reconstruct_itinerary(routing, stop_times, {}, original.id_));
}

TEST(motis,
     itinerary_id_reconstruct_dense_nearby_with_changed_ids_and_shifted_times) {
  auto const source_cfg = make_config(kDenseShiftSourceGtfs);
  auto source_data = import_test_data(source_cfg, "dense_shift_source");
  auto const original = route_first_itinerary(
      source_data, "test_OLD_A", "test_OLD_B", "2019-05-01T02:00Z");
  ASSERT_EQ(1U, original.legs_.size());
  auto const& id = original.id_;

  auto const target_cfg = make_config(kDenseShiftTargetGtfs);
  auto target_data = import_test_data(target_cfg, "dense_shift_target");
  auto const stop_times = utl::init_from<ep::stop_times>(target_data).value();
  auto const routing = utl::init_from<ep::routing>(target_data).value();

  auto const from_candidates = stop_times(
      "?center=49.87260,8.63085"
      "&time=2019-05-01T00:00:00.000Z"
      "&arriveBy=false"
      "&direction=LATER"
      "&n=10"
      "&radius=100"
      "&exactRadius=true"
      "&mode=HIGHSPEED_RAIL");
  EXPECT_GE(from_candidates.stopTimes_.size(), 8U);

  auto const to_candidates = stop_times(
      "?center=50.10701,8.66341"
      "&time=2019-05-01T00:00:00.000Z"
      "&arriveBy=true"
      "&direction=LATER"
      "&n=10"
      "&radius=100"
      "&exactRadius=true"
      "&mode=HIGHSPEED_RAIL");
  EXPECT_GE(to_candidates.stopTimes_.size(), 8U);

  auto const reconstructed =
      reconstruct_itinerary(routing, stop_times, {}, id, false);
  ASSERT_EQ(1U, reconstructed.legs_.size());
  auto const& reconstructed_leg = reconstructed.legs_.front();
  auto const& original_leg = original.legs_.front();

  ASSERT_TRUE(reconstructed_leg.from_.stopId_.has_value());
  ASSERT_TRUE(reconstructed_leg.to_.stopId_.has_value());
  EXPECT_EQ("test_NEW_A", *reconstructed_leg.from_.stopId_);
  EXPECT_EQ("test_NEW_B", *reconstructed_leg.to_.stopId_);

  ASSERT_TRUE(reconstructed_leg.tripId_.has_value());
  EXPECT_NE(std::string::npos, reconstructed_leg.tripId_->find("test_MATCH"));

  EXPECT_EQ(original_leg.scheduledStartTime_.get_unixtime_seconds() + 8 * 60,
            reconstructed_leg.scheduledStartTime_.get_unixtime_seconds());
  EXPECT_EQ(original_leg.scheduledEndTime_.get_unixtime_seconds() + 8 * 60,
            reconstructed_leg.scheduledEndTime_.get_unixtime_seconds());
}

TEST(motis, itinerary_id_reconstruct_many_candidates_benchmark) {
  constexpr auto kNumDecoys = std::size_t{50};
  constexpr auto kIterations = 30;

  auto const source_cfg = make_config(kHeavyBenchmarkSourceGtfs);
  auto source_data = import_test_data(source_cfg, "heavy_benchmark_source");
  auto const original = route_first_itinerary(
      source_data, "test_SRC_A", "test_SRC_B", "2019-05-01T02:00Z");
  ASSERT_EQ(1U, original.legs_.size());
  auto const& id = original.id_;

  auto const target_cfg = make_config(make_heavy_target_gtfs(kNumDecoys));
  auto target_data = import_test_data(target_cfg, "heavy_benchmark_target");
  auto const stop_times = utl::init_from<ep::stop_times>(target_data).value();
  auto const routing = utl::init_from<ep::routing>(target_data).value();

  auto const original_leg = original.legs_.front();
  auto const start = std::chrono::steady_clock::now();
  for (auto i = 0; i < kIterations; ++i) {
    auto const reconstructed =
        reconstruct_itinerary(routing, stop_times, {}, id, false);
    ASSERT_EQ(1U, reconstructed.legs_.size());
    auto const& leg = reconstructed.legs_.front();
    ASSERT_TRUE(leg.from_.stopId_.has_value());
    EXPECT_EQ("test_MATCH_A", *leg.from_.stopId_);
    ASSERT_TRUE(leg.to_.stopId_.has_value());
    EXPECT_EQ("test_MATCH_B", *leg.to_.stopId_);
    EXPECT_EQ(original_leg.scheduledStartTime_.get_unixtime_seconds() + 5 * 60,
              leg.scheduledStartTime_.get_unixtime_seconds());
  }
  auto const elapsed = std::chrono::steady_clock::now() - start;
  auto const per_iter_us =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() /
      kIterations;
  std::cout << "[BENCHMARK] reconstruct_itinerary with " << kNumDecoys
            << " decoys: " << per_iter_us << " us/iter (" << kIterations
            << " iterations)" << std::endl;
}

TEST(motis, refresh_itinerary_endpoint_reconstructs_itinerary) {
  auto const source_cfg = make_config(
      std::string{fmt::format(kSimpleGtfsTemplate, "DA", "FFM", "DA", "FFM")});
  auto source_data =
      import_test_data(source_cfg, "refresh_itinerary_endpoint_source");
  auto const original = route_first_itinerary(source_data, "test_DA",
                                              "test_FFM", "2019-05-01T02:00Z");
  auto const& id = original.id_;

  auto const target_cfg = make_config(
      std::string{fmt::format(kSimpleGtfsTemplate, "FFM", "DA", "FFM", "DA")});
  auto target_data =
      import_test_data(target_cfg, "refresh_itinerary_endpoint_target");
  auto expected = route_first_itinerary(target_data, "test_FFM", "test_DA",
                                        "2019-05-01T02:00Z", false);
  expected.id_ = id;

  auto const refresh = utl::init_from<ep::refresh_itinerary>(target_data);
  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = id;
  query.detailedLegs_ = false;

  auto const get_res = (*refresh)(query.to_url("?"));
  EXPECT_EQ(expected, get_res);

  auto const refresh_post =
      utl::init_from<ep::refresh_itinerary_post>(target_data).value();
  auto const body = make_refresh_itinerary_post_body(original);
  // Routing params (detailedLegs=false) are query params on the POST url; the
  // body carries only the id.
  EXPECT_EQ(get_res, refresh_post(query.to_url("?"), body));
}

TEST(motis, refresh_itinerary_matches_scheduled_then_applies_realtime) {
  auto const cfg = make_config(
      std::string{fmt::format(kSimpleGtfsTemplate, "DA", "FFM", "DA", "FFM")});
  auto data = import_test_data(cfg, "refresh_itinerary_endpoint_realtime");

  auto const original =
      route_first_itinerary(data, "test_DA", "test_FFM", "2019-05-01T02:00Z");
  auto const& id = original.id_;

  auto const rt_base_day =
      date::sys_days{date::year{2019} / date::May / date::day{1}};
  data.init_rtt(rt_base_day);

  auto const stats = n::rt::gtfsrt_update_msg(
      *data.tt_, *data.rt_->rtt_, n::source_idx_t{0}, "test",
      test::to_feed_msg(
          {test::trip_update{.trip_ = {.trip_id_ = "ICE",
                                       .start_time_ = "10:35:00",
                                       .date_ = "20190501"},
                             .stop_updates_ = {{.stop_id_ = "DA",
                                                .seq_ = 0U,
                                                .ev_type_ = n::event_type::kDep,
                                                .delay_minutes_ = 20},
                                               {.stop_id_ = "FFM",
                                                .seq_ = 1U,
                                                .ev_type_ = n::event_type::kArr,
                                                .delay_minutes_ = 22}}}},
          rt_base_day + std::chrono::hours{10}));
  EXPECT_EQ(1U, stats.total_entities_success_);

  auto const refresh = utl::init_from<ep::refresh_itinerary>(data);
  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = id;
  auto const refreshed = (*refresh)(query.to_url("?"));

  ASSERT_EQ(1U, refreshed.legs_.size());
  auto const& original_leg = original.legs_.front();
  auto const& refreshed_leg = refreshed.legs_.front();

  EXPECT_EQ(original_leg.scheduledStartTime_,
            refreshed_leg.scheduledStartTime_);
  EXPECT_EQ(original_leg.scheduledEndTime_, refreshed_leg.scheduledEndTime_);
  EXPECT_EQ(original_leg.startTime_.get_unixtime_seconds() + 20 * 60,
            refreshed_leg.startTime_.get_unixtime_seconds());
  EXPECT_EQ(original_leg.endTime_.get_unixtime_seconds() + 22 * 60,
            refreshed_leg.endTime_.get_unixtime_seconds());
  EXPECT_TRUE(refreshed_leg.realTime_);
}

TEST(motis, refresh_itinerary_reconstructs_added_trip_by_trip_id_only) {
  auto const cfg = make_config(
      std::string{fmt::format(kSimpleGtfsTemplate, "DA", "FFM", "DA", "FFM")});
  auto data = import_test_data(cfg, "refresh_itinerary_added_trip_success");

  auto const rt_base_day =
      date::sys_days{date::year{2019} / date::May / date::day{1}};
  data.init_rtt(rt_base_day);

  auto const stats = n::rt::gtfsrt_update_msg(
      *data.tt_, *data.rt_->rtt_, n::source_idx_t{0}, "test",
      make_added_trip_update(rt_base_day + std::chrono::hours{9}));
  EXPECT_EQ(1U, stats.total_entities_success_);

  auto const added_trip_id = std::string{"20190501_09:02_test_ADDED_ICE"};
  auto const trip = utl::init_from<ep::trip>(data).value();
  auto trip_query = api::trip_params{};
  trip_query.tripId_ = added_trip_id;
  auto const added_itinerary = trip(trip_query.to_url("?"));
  auto const& itinerary_id = added_itinerary.id_;

  auto const refresh = utl::init_from<ep::refresh_itinerary>(data).value();
  auto refresh_query = api::refreshItinerary_params{};
  refresh_query.itineraryId_ = itinerary_id;
  auto const refreshed = refresh(refresh_query.to_url("?"));

  ASSERT_EQ(1U, refreshed.legs_.size());
  EXPECT_EQ(itinerary_id, refreshed.id_);
  ASSERT_TRUE(refreshed.legs_.front().tripId_.has_value());
  EXPECT_EQ(added_trip_id, *refreshed.legs_.front().tripId_);
}

// GTFS-Flex dataset: a geojson area ("da_flex", service day 2019-05-01) and a
// location group ("da_group", service day 2019-05-02), both used as a flex
// first mile to board the ICE at DA_10 -> FFM_10. Same timetable, different
// service days select the area- vs group-based flex.
constexpr auto kFlexGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
DA,DA Hbf,49.87260,8.63085,1,,
DA_10,DA Hbf,49.87336,8.62926,0,DA,10
DA_FLEX,DA Flex Pickup,49.87420,8.62940,0,,
FFM,FFM Hbf,50.10701,8.66341,1,,
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
ICE,DB,ICE,,,101
FLEXA,DB,FlexArea,,,715
FLEXG,DB,FlexGroup,,,715

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
ICE,S_ALL,ICE,,
ICE,S_ALL,ICE_LATE,,
FLEXA,S_AREA,FLEX_AREA,,
FLEXG,S_GROUP,FLEX_GROUP,,

# booking_rules.txt
booking_rule_id,booking_type,prior_notice_duration_min,prior_notice_duration_max
BR,1,0,86400

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,location_group_id,location_id,stop_sequence,start_pickup_drop_off_window,end_pickup_drop_off_window,pickup_booking_rule_id,drop_off_booking_rule_id,pickup_type,drop_off_type
ICE,10:00:00,10:00:00,DA_10,,,0,,,,,0,0
ICE,11:00:00,11:00:00,FFM_10,,,1,,,,,0,0
ICE_LATE,11:00:00,11:00:00,DA_10,,,0,,,,,0,0
ICE_LATE,12:00:00,12:00:00,FFM_10,,,1,,,,,0,0
FLEX_AREA,,,,,da_flex,0,00:00:00,24:00:00,BR,BR,2,2
FLEX_AREA,,,,,da_flex,1,00:00:00,24:00:00,BR,BR,2,2
FLEX_GROUP,,,,da_group,,0,00:00:00,24:00:00,BR,BR,2,2
FLEX_GROUP,,,,da_group,,1,00:00:00,24:00:00,BR,BR,2,2

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
S_ALL,1,1,1,1,1,1,1,20190501,20190503
S_AREA,1,1,1,1,1,1,1,20190501,20190501
S_GROUP,1,1,1,1,1,1,1,20190502,20190502

# locations.geojson
{"type":"FeatureCollection","features":[{"id":"da_flex","type":"Feature","geometry":{"type":"Polygon","coordinates":[[[8.620,49.865],[8.640,49.865],[8.640,49.880],[8.620,49.880],[8.620,49.865]]]},"properties":{"stop_name":"DA Flex Area"}}]}

# location_groups.txt
location_group_id,location_group_name
da_group,DA Flex Group

# location_group_stops.txt
location_group_id,stop_id
da_group,DA_FLEX
da_group,DA_10
)";

// Plans a FLEX first mile to board the ICE on `day` (area on 2019-05-01,
// location group on 2019-05-02), asserts the id encodes the access as a single
// FLEX leg, and that reconstruction re-routes it (round-trips) via the
// time-dependent flex offsets.
void run_flex_first_mile_test(std::string_view const sub_dir,
                              std::string_view const day) {
  auto ec = std::error_code{};
  auto const path = fs::path{"test/data/itinerary_id"} / sub_dir;
  fs::remove_all(path, ec);
  auto const cfg = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 3,
              .datasets_ = {{"test", {.path_ = std::string{kFlexGtfs}}}},
          },
      .street_routing_ = true,
      .osr_footpath_ = true,
  };
  import(cfg, path);
  auto d = data{path, cfg};
  auto const routing = utl::init_from<ep::routing>(d).value();

  // Query before the ICE departure with timetableView so the same-day flex
  // service is used (a query at the PT departure would force the access to
  // start earlier than the query and roll the journey onto the next day).
  // `numLegAlternatives` so the boarding leg also gets earlier/later departures
  // as leg alternatives -- each of those must reproduce the FLEX first mile
  // too, not just the main leg.
  auto const res =
      routing(fmt::format("/api/v6/plan"
                          "?fromPlace=49.87420,8.62940"
                          "&toPlace=50.10593,8.66118"
                          "&time={}T05:00Z"
                          "&timetableView=true"
                          "&searchWindow=10800"
                          "&preTransitModes=FLEX"
                          "&postTransitModes=WALK"
                          "&numLegAlternatives=3"
                          "&detailedLegs=true",
                          day));
  ASSERT_FALSE(res.itineraries_.empty());
  auto const& original = res.itineraries_.front();

  // The flex access is one journey offset leg, encoded as a single FLEX leg.
  auto proto = motis::ItineraryId{};
  ASSERT_TRUE(proto.ParseFromString(net::decode_base64(original.id_)));
  ASSERT_GE(proto.legs_size(), 1);
  EXPECT_EQ("FLEX", proto.legs(0).mode());

  // Sanity: the boarding leg has alternatives in the plan, so the round-trip
  // comparison below is meaningful.
  auto const boarding_it = utl::find_if(
      original.legs_, [](api::Leg const& l) { return l.tripId_.has_value(); });
  ASSERT_NE(boarding_it, end(original.legs_));
  auto const& boarding = *boarding_it;
  ASSERT_TRUE(boarding.alternatives_.has_value());
  ASSERT_FALSE(boarding.alternatives_->empty());

  // Reconstruction re-routes the flex access via get_td_offsets and renders it
  // with the flex areas, reproducing the original itinerary -- including the
  // leg alternatives. The id only encodes the chosen first-mile mode, so the
  // access modes must be supplied (preTransitModes=FLEX) for the alternatives'
  // boundary offsets to be recomputed as flex (not a bare boarding-stop walk).
  auto refresh_q = api::refreshItinerary_params{};
  refresh_q.preTransitModes_ = {api::ModeEnum::FLEX};
  auto const stop_times = utl::init_from<ep::stop_times>(d).value();
  auto const reconstructed = reconstruct_itinerary(
      routing, stop_times, *d.rt_, original.id_,
      /*require_display_name_match=*/true, /*join_interlined_legs=*/true,
      /*detailed_transfers=*/true, /*detailed_legs=*/true,
      /*with_scheduled_skipped_stops=*/false, n::lang_t{},
      /*num_leg_alternatives=*/3U, n::routing::all_clasz_allowed(),
      /*require_bike_transport=*/false, /*require_car_transport=*/false,
      /*prf_idx=*/n::profile_idx_t{0U},
      make_first_last_mile_options(refresh_q));
  EXPECT_EQ(original, reconstructed);
}

TEST(motis, itinerary_id_reconstruct_flex_area_first_mile) {
  run_flex_first_mile_test("flex_area", "2019-05-01");
}

TEST(motis, itinerary_id_reconstruct_flex_location_group_first_mile) {
  run_flex_first_mile_test("flex_group", "2019-05-02");
}

constexpr auto kMultiHopGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
DA,DA Hbf,49.87260,8.63085,1,,
DA_10,DA Hbf,49.87336,8.62926,0,DA,10
LANGEN,Langen,49.99359,8.65677,1,,
FFM,FFM Hbf,50.10701,8.66341,1,,
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10
FFM_101,FFM Hbf,50.10739,8.66333,0,FFM,101
FFM_HAUPT,FFM Hauptwache,50.11403,8.67835,1,,
FFM_HAUPT_S,FFM Hauptwache S,50.11404,8.67824,0,FFM_HAUPT,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RE1,DB,RE1,,,106
RE2,DB,RE2,,,106
S3,DB,S3,,,109

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RE1,S1,RE1,,
RE2,S1,RE2,,
S3,S1,S3,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
RE1,10:00:00,10:00:00,DA_10,0,0,0
RE1,10:08:00,10:08:00,LANGEN,1,0,0
RE2,10:10:00,10:10:00,LANGEN,0,0,0
RE2,10:25:00,10:25:00,FFM_10,1,0,0
S3,10:30:00,10:30:00,FFM_101,0,0,0
S3,10:38:00,10:38:00,FFM_HAUPT_S,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

TEST(motis, itinerary_id_reconstruct_multi_leg_with_three_pt_hops) {
  auto ec = std::error_code{};
  auto const path = fs::path{"test/data/itinerary_id"} / "multi_leg_three_hops";
  fs::remove_all(path, ec);

  auto const cfg = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test", {.path_ = std::string{kMultiHopGtfs}}}},
          },
      .street_routing_ = true,
      .osr_footpath_ = true,
  };
  import(cfg, path);
  auto d = data{path, cfg};

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const res = routing(
      "/api/v6/plan"
      "?fromPlace=49.87420,8.62940"
      "&toPlace=50.11450,8.67900"
      "&time=2019-05-01T02:00Z"
      "&timetableView=true"
      "&preTransitModes=WALK"
      "&postTransitModes=WALK"
      "&detailedLegs=true");

  ASSERT_FALSE(res.itineraries_.empty());
  auto const& original = res.itineraries_.front();

  auto pt_count = std::size_t{0};
  auto walk_count = std::size_t{0};
  for (auto const& l : original.legs_) {
    if (l.mode_ == api::ModeEnum::WALK) {
      ++walk_count;
    } else {
      ++pt_count;
    }
  }
  ASSERT_GE(pt_count, 3U) << "expected at least 3 PT legs, got " << pt_count;
  ASSERT_GE(walk_count, 4U) << "expected at least 4 walk legs (first-mile + "
                               "transfers + last-mile), got "
                            << walk_count;
  ASSERT_EQ(api::ModeEnum::WALK, original.legs_.front().mode_)
      << "expected first leg to be a first-mile walk";
  ASSERT_EQ(api::ModeEnum::WALK, original.legs_.back().mode_)
      << "expected last leg to be a last-mile walk";

  auto const& id = original.id_;
  auto const stop_times = utl::init_from<ep::stop_times>(d).value();
  auto const reconstructed =
      reconstruct_itinerary(routing, stop_times, *d.rt_, id, true, true, true);

  EXPECT_EQ(original, reconstructed);
}

// Two car-carrying trains (cars_allowed=1) meeting at an interchange where the
// alighting and boarding stops are distinct stations a short drive apart. With
// requireCarTransport + useRoutedTransfers the transfer between the two trains
// is routed with the CAR profile (Autoverlad / car-ferry scenario), so the
// itinerary contains a CAR transfer leg between the two PT legs.
constexpr auto kCarTransferGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
ORIG,Origin,49.87336,8.62926,1,,
FERRY_N,Ferry North,50.10593,8.66118,1,,
FERRY_S,Ferry South,50.10739,8.66333,1,,
DEST,Destination,50.11404,8.67824,1,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
CT1_R,DB,CT1,,,106
CT2_R,DB,CT2,,,106

# trips.txt
route_id,service_id,trip_id,trip_headsign,cars_allowed
CT1_R,S1,CT1,,1
CT2_R,S1,CT2,,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
CT1,10:00:00,10:00:00,ORIG,0
CT1,10:30:00,10:30:00,FERRY_S,1
CT2,10:45:00,10:45:00,FERRY_N,0
CT2,11:00:00,11:00:00,DEST,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

TEST(motis, itinerary_id_reconstruct_car_transfer) {
  auto ec = std::error_code{};
  auto const path = fs::path{"test/data/itinerary_id"} / "car_transfer";
  fs::remove_all(path, ec);

  auto const cfg = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test", {.path_ = std::string{kCarTransferGtfs}}}},
          },
      .street_routing_ = true,
      .osr_footpath_ = true,
  };
  import(cfg, path);
  auto d = data{path, cfg};

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const plan = routing(
      "/api/v6/plan"
      "?fromPlace=49.87420,8.62940"
      "&toPlace=50.11450,8.67900"
      "&time=2019-05-01T02:00Z"
      "&timetableView=true"
      "&requireCarTransport=true"
      "&useRoutedTransfers=true"
      "&preTransitModes=WALK"
      "&postTransitModes=WALK"
      "&detailedTransfers=true"
      "&detailedLegs=true");

  ASSERT_FALSE(plan.itineraries_.empty());
  auto const& original = plan.itineraries_.front();

  auto const pt_legs = utl::count_if(
      original.legs_, [](api::Leg const& l) { return l.tripId_.has_value(); });
  ASSERT_EQ(2, pt_legs) << "expected the two car trains as PT legs";
  auto const car_transfers =
      utl::count_if(original.legs_, [](api::Leg const& l) {
        return !l.tripId_.has_value() && l.mode_ == api::ModeEnum::CAR;
      });
  ASSERT_GE(car_transfers, 1)
      << "expected a CAR transfer between the two car trains";

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = original.id_;
  query.requireCarTransport_ = true;
  query.useRoutedTransfers_ = true;
  query.detailedTransfers_ = true;
  query.detailedLegs_ = true;
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));

  EXPECT_EQ(original, refreshed);
}

// First/last mile with a non-walking access mode (BIKE) must be encoded in the
// id and re-routed on reconstruction (rather than replayed as a fixed walk).
TEST(motis, itinerary_id_reconstruct_bike_first_last_mile) {
  auto ec = std::error_code{};
  auto const path = fs::path{"test/data/itinerary_id"} / "bike_first_last_mile";
  fs::remove_all(path, ec);

  auto const cfg = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test", {.path_ = std::string{kMultiHopGtfs}}}},
          },
      .street_routing_ = true,
      .osr_footpath_ = true,
  };
  import(cfg, path);
  auto d = data{path, cfg};

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const res = routing(
      "/api/v6/plan"
      "?fromPlace=49.87420,8.62940"
      "&toPlace=50.11450,8.67900"
      "&time=2019-05-01T02:00Z"
      "&timetableView=true"
      "&preTransitModes=BIKE"
      "&postTransitModes=BIKE"
      "&detailedLegs=true");
  ASSERT_FALSE(res.itineraries_.empty());
  auto const& original = res.itineraries_.front();
  ASSERT_EQ(api::ModeEnum::BIKE, original.legs_.front().mode_)
      << "expected a BIKE first mile";
  ASSERT_EQ(api::ModeEnum::BIKE, original.legs_.back().mode_)
      << "expected a BIKE last mile";

  // The id encodes the access/egress mode (BIKE), not WALK.
  auto proto = motis::ItineraryId{};
  ASSERT_TRUE(proto.ParseFromString(net::decode_base64(original.id_)));
  ASSERT_GE(proto.legs_size(), 2);
  EXPECT_EQ("BIKE", proto.legs(0).mode());
  EXPECT_EQ("BIKE", proto.legs(proto.legs_size() - 1).mode());

  // Reconstruction re-routes the first/last mile by BIKE (mode taken from the
  // id); the default first/last-mile options (15-min limits, default speeds)
  // suffice here.
  auto const stop_times = utl::init_from<ep::stop_times>(d).value();
  auto const reconstructed = reconstruct_itinerary(
      routing, stop_times, *d.rt_, original.id_, true, true, true);
  ASSERT_FALSE(reconstructed.legs_.empty());
  EXPECT_EQ(api::ModeEnum::BIKE, reconstructed.legs_.front().mode_);
  EXPECT_EQ(api::ModeEnum::BIKE, reconstructed.legs_.back().mode_);
}

// Shared-mobility (gbfs) first mile renders as WALK -> RENTAL -> WALK from a
// single journey offset leg. The id stores one access leg with mode RENTAL,
// and reconstruction re-routes it via gbfs (re-expanding to multiple legs).
TEST(motis, itinerary_id_reconstruct_rental_first_mile) {
  auto ec = std::error_code{};
  auto const path = fs::path{"test/data/itinerary_id"} / "rental_first_mile";
  fs::remove_all(path, ec);

  auto const cfg = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test", {.path_ = std::string{kMultiHopGtfs}}}},
          },
      .gbfs_ = {{.feeds_ = {{"CAB", {.url_ = "./test/resources/gbfs"}}}}},
      .street_routing_ = true,
      .osr_footpath_ = true,
  };
  import(cfg, path);
  auto d = data{path, cfg};

  // Load the gbfs feed (otherwise no rental vehicles are available).
  {
    auto ioc = boost::asio::io_context{};
    boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<void> {
          co_await gbfs::update(cfg, *d.w_, *d.l_, d.gbfs_, d.metrics_.get());
        },
        boost::asio::detached);
    ioc.run();
  }

  auto const routing = utl::init_from<ep::routing>(d).value();
  // Start right at the shared bike (from free_bike_status.json).
  auto const res = routing(
      "/api/v6/plan"
      "?fromPlace=49.875308,8.6276673"
      "&toPlace=50.11450,8.67900"
      "&time=2019-05-01T02:00Z"
      "&timetableView=true"
      "&preTransitModes=RENTAL"
      "&postTransitModes=WALK"
      "&detailedLegs=true");
  ASSERT_FALSE(res.itineraries_.empty());
  auto const& original = res.itineraries_.front();

  auto const has_rental = [](api::Itinerary const& i) {
    return utl::any_of(i.legs_, [](api::Leg const& l) {
      return l.mode_ == api::ModeEnum::RENTAL;
    });
  };
  ASSERT_TRUE(has_rental(original)) << "expected a RENTAL first mile";

  // The id encodes a single access leg with mode RENTAL (not the multiple
  // rendered WALK/RENTAL/WALK legs).
  auto proto = motis::ItineraryId{};
  ASSERT_TRUE(proto.ParseFromString(net::decode_base64(original.id_)));
  EXPECT_EQ("RENTAL", proto.legs(0).mode());

  auto const stop_times = utl::init_from<ep::stop_times>(d).value();
  auto const reconstructed = reconstruct_itinerary(
      routing, stop_times, *d.rt_, original.id_, true, true, true);
  EXPECT_TRUE(has_rental(reconstructed))
      << "reconstruction must re-route the RENTAL access (WALK/RENTAL/WALK)";
}

TEST(motis, itinerary_id_reconstruct_multi_leg_replaces_unmatched_leg_dummy) {
  auto ec = std::error_code{};
  auto const path = fs::path{"test/data/itinerary_id"} / "multi_leg_dummy_leg";
  fs::remove_all(path, ec);

  auto const cfg = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test", {.path_ = std::string{kMultiHopGtfs}}}},
          },
      .street_routing_ = true,
      .osr_footpath_ = true,
  };
  import(cfg, path);
  auto d = data{path, cfg};

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const res = routing(
      "/api/v5/plan"
      "?fromPlace=49.87420,8.62940"
      "&toPlace=50.11450,8.67900"
      "&time=2019-05-01T02:00Z"
      "&timetableView=true"
      "&preTransitModes=WALK"
      "&postTransitModes=WALK"
      "&detailedLegs=true");
  ASSERT_FALSE(res.itineraries_.empty());

  auto const original = res.itineraries_.front();
  auto pt_indices = std::vector<std::size_t>{};
  for (auto i = std::size_t{0}; i < original.legs_.size(); ++i) {
    if (original.legs_[i].mode_ != api::ModeEnum::WALK) {
      pt_indices.push_back(i);
    }
  }
  ASSERT_GE(pt_indices.size(), 3U);

  // Make the middle PT leg impossible to reconstruct.
  auto const broken_idx = pt_indices.at(1);
  ASSERT_TRUE(original.legs_.at(broken_idx).tripId_.has_value());
  auto const& broken_trip_id = *original.legs_.at(broken_idx).tripId_;

  auto proto = motis::ItineraryId{};
  ASSERT_TRUE(proto.ParseFromString(net::decode_base64(original.id_)));
  auto broken_found = false;
  for (auto i = 0; i < proto.legs_size(); ++i) {
    if (proto.legs(i).trip_id() == broken_trip_id) {
      auto* const l = proto.mutable_legs(i);
      l->set_from_lat(0.0);
      l->set_from_lon(0.0);
      l->set_to_lat(0.0);
      l->set_to_lon(0.0);
      broken_found = true;
    }
  }
  ASSERT_TRUE(broken_found);
  auto proto_data = std::string{};
  ASSERT_TRUE(proto.SerializeToString(&proto_data));
  auto const id = net::encode_base64(proto_data);
  auto const stop_times = utl::init_from<ep::stop_times>(d).value();
  auto const reconstructed =
      reconstruct_itinerary(routing, stop_times, *d.rt_, id, true, true, true);

  // Every leg is still present, in order, and the itinerary id is preserved.
  ASSERT_EQ(original.legs_.size(), reconstructed.legs_.size());
  EXPECT_EQ(id, reconstructed.id_);

  // The unreconstructable leg became a cancelled dummy carrying an alert with
  // the failure reason, keeping the original transit mode.
  auto const& dummy = reconstructed.legs_.at(broken_idx);
  EXPECT_TRUE(dummy.cancelled_.value_or(false));
  EXPECT_TRUE(dummy.from_.cancelled_.value_or(false));
  EXPECT_TRUE(dummy.to_.cancelled_.value_or(false));
  EXPECT_EQ(original.legs_.at(broken_idx).mode_, dummy.mode_);
  ASSERT_TRUE(dummy.alerts_.has_value());
  ASSERT_EQ(1U, dummy.alerts_->size());
  EXPECT_FALSE(dummy.alerts_->front().headerText_.empty());

  // The surrounding PT legs were reconstructed normally (not cancelled).
  for (auto const pt_idx : pt_indices) {
    if (pt_idx != broken_idx) {
      EXPECT_FALSE(reconstructed.legs_.at(pt_idx).cancelled_.value_or(false));
    }
  }
}

constexpr auto kLegAltTransitModesGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,A,49.80000,8.60000,0,
B,B,50.20000,8.70000,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
REG_MAIN_R,DB,RM,,,106
BUS_ALT_R,DB,BA,,,3
REG_ALT_R,DB,RA,,,106

# trips.txt
route_id,service_id,trip_id,trip_headsign
REG_MAIN_R,S1,REG_MAIN,
BUS_ALT_R,S1,BUS_ALT,
REG_ALT_R,S1,REG_ALT,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
REG_MAIN,09:00:00,09:00:00,A,0
REG_MAIN,10:00:00,10:00:00,B,1
BUS_ALT,09:30:00,09:30:00,A,0
BUS_ALT,10:30:00,10:30:00,B,1
REG_ALT,10:00:00,10:00:00,A,0
REG_ALT,11:00:00,11:00:00,B,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

// First transit leg of the itinerary (the leg that carries `alternatives`).
api::Leg const& first_transit_leg(api::Itinerary const& it) {
  auto const leg = utl::find_if(
      it.legs_, [](api::Leg const& l) { return l.tripId_.has_value(); });
  utl::verify(leg != end(it.legs_), "itinerary has no transit leg");
  return *leg;
}

// All transit legs of an itinerary, in order.
std::vector<api::Leg const*> transit_legs(api::Itinerary const& it) {
  auto legs = std::vector<api::Leg const*>{};
  for (auto const& l : it.legs_) {
    if (l.tripId_.has_value()) {
      legs.push_back(&l);
    }
  }
  return legs;
}

// Sorted trip ids of every transit leg across a leg's computed alternatives.
std::vector<std::string> alt_transit_trip_ids(api::Leg const& leg) {
  auto ids = std::vector<std::string>{};
  if (leg.alternatives_.has_value()) {
    for (auto const& alt : *leg.alternatives_) {
      for (auto const& al : alt) {
        if (al.tripId_.has_value() && !al.tripId_->empty()) {
          ids.push_back(*al.tripId_);
        }
      }
    }
  }
  std::sort(begin(ids), end(ids));
  return ids;
}

// True if any leg of any computed alternative is a transit leg whose trip id
// contains `needle` (trip ids are prefixed by the importer, hence substring).
bool any_alt_trip_contains(api::Leg const& leg, std::string_view const needle) {
  if (!leg.alternatives_.has_value()) {
    return false;
  }
  for (auto const& alt : *leg.alternatives_) {
    for (auto const& al : alt) {
      if (al.tripId_.has_value() &&
          al.tripId_->find(needle) != std::string::npos) {
        return true;
      }
    }
  }
  return false;
}

constexpr auto kLegAltBikeGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,A,49.80000,8.60000,0,
B,B,50.20000,8.70000,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
BIKE_MAIN_R,DB,BM,,,106
BIKE_NO_R,DB,BN,,,106
BIKE_YES_R,DB,BY,,,106

# trips.txt
route_id,service_id,trip_id,trip_headsign,bikes_allowed
BIKE_MAIN_R,S1,BIKE_MAIN,,1
BIKE_NO_R,S1,BIKE_NO,,2
BIKE_YES_R,S1,BIKE_YES,,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
BIKE_MAIN,09:00:00,09:00:00,A,0
BIKE_MAIN,10:00:00,10:00:00,B,1
BIKE_NO,09:30:00,09:30:00,A,0
BIKE_NO,10:30:00,10:30:00,B,1
BIKE_YES,10:00:00,10:00:00,A,0
BIKE_YES,11:00:00,11:00:00,B,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

constexpr auto kLegAltCarGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,A,49.80000,8.60000,0,
B,B,50.20000,8.70000,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
CAR_MAIN_R,DB,CM,,,106
CAR_NO_R,DB,CN,,,106
CAR_YES_R,DB,CY,,,106

# trips.txt
route_id,service_id,trip_id,trip_headsign,cars_allowed
CAR_MAIN_R,S1,CAR_MAIN,,1
CAR_NO_R,S1,CAR_NO,,2
CAR_YES_R,S1,CAR_YES,,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
CAR_MAIN,09:00:00,09:00:00,A,0
CAR_MAIN,10:00:00,10:00:00,B,1
CAR_NO,09:30:00,09:30:00,A,0
CAR_NO,10:30:00,10:30:00,B,1
CAR_YES,10:00:00,10:00:00,A,0
CAR_YES,11:00:00,11:00:00,B,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

// `pedestrianProfile=WHEELCHAIR` only changes leg alternatives when the
// timetable actually has a (non-empty) wheelchair footpath profile - otherwise
// the `plan` endpoint silently falls back to the default profile. `A`/`B`
// carry the actual route; `FP`/`FP_1`/`FP_2` are an unrelated parent station
// whose two platforms (placed on `test_case.osm.pbf`) make OSR generate a
// wheelchair footpath, keeping the wheelchair profile non-empty.
constexpr auto kLegAltWheelchairGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,A,50.10701,8.66341,0,
B,B,50.11403,8.67835,0,
FP,Footpath Station,49.87260,8.63085,1,
FP_1,Footpath Platform 1,49.87260,8.63085,0,FP
FP_2,Footpath Platform 2,49.87336,8.62926,0,FP

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
WC_MAIN_R,DB,WM,,,106
WC_NO_R,DB,WN,,,106
WC_YES_R,DB,WY,,,106
FP_R,DB,FP,,,106

# trips.txt
route_id,service_id,trip_id,trip_headsign,wheelchair_accessible
WC_MAIN_R,S1,WC_MAIN,,1
WC_NO_R,S1,WC_NO,,2
WC_YES_R,S1,WC_YES,,1
FP_R,S1,FP_TRIP,,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
WC_MAIN,09:00:00,09:00:00,A,0
WC_MAIN,10:00:00,10:00:00,B,1
WC_NO,09:30:00,09:30:00,A,0
WC_NO,10:30:00,10:30:00,B,1
WC_YES,10:00:00,10:00:00,A,0
WC_YES,11:00:00,11:00:00,B,1
FP_TRIP,07:00:00,07:00:00,FP_1,0
FP_TRIP,07:30:00,07:30:00,FP_2,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

// Three-transit-leg journey A -> B -> C -> D. The middle leg B -> C has a
// transit leg both before and after it, so its alternatives are computed with
// exact endpoint matching (`prev_it->to_` / `next_it->from_`) regardless of
// the query's endpoint match mode - this fixture exercises that path through
// the refresh endpoint. `L2_REG` arrives one minute before `L3` departs, so
// it is a valid leg alternative but cannot itself form the main journey.
constexpr auto kLegAltIntermediateGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,A,49.80000,8.60000,0,
B,B,49.90000,8.62000,0,
C,C,50.00000,8.64000,0,
D,D,50.20000,8.70000,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
L1_R,DB,L1,,,106
L2_MAIN_R,DB,L2M,,,106
L2_BUS_R,DB,L2B,,,3
L2_REG_R,DB,L2R,,,106
L3_R,DB,L3,,,106

# trips.txt
route_id,service_id,trip_id,trip_headsign
L1_R,S1,L1,
L2_MAIN_R,S1,L2_MAIN,
L2_BUS_R,S1,L2_BUS,
L2_REG_R,S1,L2_REG,
L3_R,S1,L3,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
L1,09:00:00,09:00:00,A,0
L1,09:15:00,09:15:00,B,1
L2_MAIN,09:30:00,09:30:00,B,0
L2_MAIN,10:00:00,10:00:00,C,1
L2_BUS,09:45:00,09:45:00,B,0
L2_BUS,10:15:00,10:15:00,C,1
L2_REG,10:00:00,10:00:00,B,0
L2_REG,10:43:00,10:43:00,C,1
L3,10:45:00,10:45:00,C,0
L3,11:15:00,11:15:00,D,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

// Intermodal journey with two competing access points. `START` can walk to
// stop `A` (trip `AC`) or stop `B` (trip `BD`); `DEST` can be reached from `C`
// or `D`. The plan computes leg alternatives over both access stops, so the
// `AC` leg gets `BD` as an alternative even though it boards / alights at
// entirely different stops. The refresh endpoint must re-derive those access
// offsets from the `START`/`DEST` coordinates to reproduce this.
// `A`/`B`/`C`/`D` lie on `test/resources/test_case.osm.pbf`.
constexpr auto kLegAltIntermodalGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,A,49.87336,8.62926,0,
B,B,49.87260,8.63085,0,
C,C,50.10593,8.66118,0,
D,D,50.10739,8.66333,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
AC_R,DB,AC,,,106
BD_R,DB,BD,,,106

# trips.txt
route_id,service_id,trip_id,trip_headsign
AC_R,S1,AC,
BD_R,S1,BD,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
AC,09:00:00,09:00:00,A,0
AC,10:00:00,10:00:00,C,1
BD,09:30:00,09:30:00,B,0
BD,10:30:00,10:30:00,D,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

// Intermodal access offsets: the refresh endpoint must re-derive the set of
// reachable access stops from the START / DEST coordinates so that a leg
// alternative may board / alight at a different stop than the chosen journey.
TEST(motis, itinerary_id_refresh_leg_alternatives_intermodal_offsets) {
  auto ec = std::error_code{};
  auto const path =
      fs::path{"test/data/itinerary_id"} / "leg_alt_intermodal_offsets";
  fs::remove_all(path, ec);

  auto const cfg = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test",
                             {.path_ = std::string{kLegAltIntermodalGtfs}}}},
          },
      .street_routing_ = true,
  };
  import(cfg, path);
  auto d = data{path, cfg};

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const plan = routing(
      "?fromPlace=49.87298,8.63006&toPlace=50.10666,8.66225"
      "&time=2019-05-01T05:00Z&searchWindow=14400&detailedLegs=true"
      "&numLegAlternatives=5");
  ASSERT_FALSE(plan.itineraries_.empty());
  auto const& plan_itin = plan.itineraries_.front();
  auto const& plan_leg = first_transit_leg(plan_itin);
  ASSERT_TRUE(plan_leg.alternatives_.has_value());
  ASSERT_FALSE(plan_leg.alternatives_->empty());
  // The chosen leg is `AC`; the plan offers `BD` (different access stops) as
  // an alternative because both are reachable from the START/DEST coordinates.
  EXPECT_TRUE(any_alt_trip_contains(plan_leg, "BD"));

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = plan_itin.id_;
  query.numLegAlternatives_ = 5;
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));
  auto const& refresh_leg = first_transit_leg(refreshed);

  // Re-derived access offsets let the refresh endpoint reproduce the `BD`
  // alternative that boards / alights at different stops than the journey.
  EXPECT_TRUE(any_alt_trip_contains(refresh_leg, "BD"));
  // Compare the alternative trips: the surrounding access-walk legs may differ
  // by sub-minute rounding, but the set of alternative transit trips must not.
  EXPECT_EQ(alt_transit_trip_ids(plan_leg), alt_transit_trip_ids(refresh_leg));
}

constexpr auto kLegAltFirstAccessGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,A,49.87336,8.62926,0,
B,B,49.87260,8.63085,0,
C,C,49.99000,8.65000,0,
D,D,50.10739,8.66333,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
AC_R,DB,AC,,,106
BC_R,DB,BC,,,106
CD_R,DB,CD,,,106

# trips.txt
route_id,service_id,trip_id,trip_headsign
AC_R,S1,AC,
BC_R,S1,BC,
CD_R,S1,CD,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
AC,09:00:00,09:00:00,A,0
AC,09:30:00,09:30:00,C,1
BC,09:05:00,09:05:00,B,0
BC,09:35:00,09:35:00,C,1
CD,10:00:00,10:00:00,C,0
CD,10:30:00,10:30:00,D,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

// First leg of a MULTI-leg journey with two competing access stops: the chosen
// journey boards at B (BC -> CD), but A is also reachable from START, so the
// first leg offers the `AC` alternative -- which boards at a different stop AND
// departs *earlier* than the chosen `BC`. `plan` passes the whole journey to
// `get_leg_alternatives`, so the first leg (a successor exists) uses the
// "earlier arrivals" branch and finds `AC`. The refresh endpoint reconstructs
// legs one at a time, so it must hand `get_leg_alternatives` the surrounding
// transit context to reproduce the same alternative set (otherwise it only
// sees "later departures" and drops the earlier `AC`).
TEST(motis, itinerary_id_refresh_leg_alternatives_first_leg_other_access) {
  auto ec = std::error_code{};
  auto const path = fs::path{"test/data/itinerary_id"} / "leg_alt_first_access";
  fs::remove_all(path, ec);

  auto const cfg = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test",
                             {.path_ = std::string{kLegAltFirstAccessGtfs}}}},
          },
      .street_routing_ = true,
  };
  import(cfg, path);
  auto d = data{path, cfg};

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const plan = routing(
      "?fromPlace=49.87298,8.63006&toPlace=50.10720,8.66320"
      "&time=2019-05-01T05:00Z&searchWindow=14400&detailedLegs=true"
      "&numLegAlternatives=5");
  ASSERT_FALSE(plan.itineraries_.empty());
  auto const& plan_itin = plan.itineraries_.front();
  auto const& plan_leg = first_transit_leg(plan_itin);
  ASSERT_TRUE(plan_leg.alternatives_.has_value());
  EXPECT_TRUE(any_alt_trip_contains(plan_leg, "AC"))
      << "plan should offer the earlier AC alternative that boards at a "
         "different stop";

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = plan_itin.id_;
  query.numLegAlternatives_ = 5;
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));
  auto const& refresh_leg = first_transit_leg(refreshed);

  EXPECT_TRUE(any_alt_trip_contains(refresh_leg, "AC"))
      << "refresh must reproduce the earlier, different-access-stop "
         "alternative";
  EXPECT_EQ(alt_transit_trip_ids(plan_leg), alt_transit_trip_ids(refresh_leg));
}

// transitModes: the alternative bus departs before the alternative regional
// train, so without the constraint the bus would be the first alternative.
TEST(motis, itinerary_id_refresh_leg_alternatives_respect_transit_modes) {
  auto const cfg = make_config(kLegAltTransitModesGtfs);
  auto d = import_test_data(cfg, "leg_alt_transit_modes");

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const plan = routing(
      "?fromPlace=test_A&toPlace=test_B"
      "&time=2019-05-01T05:00Z&searchWindow=7200&detailedLegs=true"
      "&transitModes=REGIONAL_RAIL&numLegAlternatives=5");
  ASSERT_FALSE(plan.itineraries_.empty());
  auto const& plan_itin = plan.itineraries_.front();
  auto const& plan_leg = first_transit_leg(plan_itin);
  ASSERT_TRUE(plan_leg.alternatives_.has_value());
  ASSERT_FALSE(plan_leg.alternatives_->empty());
  EXPECT_TRUE(any_alt_trip_contains(plan_leg, "REG_ALT"));
  EXPECT_FALSE(any_alt_trip_contains(plan_leg, "BUS_ALT"));

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = plan_itin.id_;
  query.numLegAlternatives_ = 5;
  query.transitModes_ = {api::ModeEnum::REGIONAL_RAIL};
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));
  auto const& refresh_leg = first_transit_leg(refreshed);

  // The refreshed alternatives must match the plan response exactly: only the
  // regional train, never the (earlier-departing) bus.
  EXPECT_TRUE(any_alt_trip_contains(refresh_leg, "REG_ALT"));
  EXPECT_FALSE(any_alt_trip_contains(refresh_leg, "BUS_ALT"));
  // With the leg-alternatives query mirroring the `plan` endpoint's endpoint
  // match mode, the refreshed alternatives match the plan response exactly.
  EXPECT_EQ(plan_leg.alternatives_, refresh_leg.alternatives_);
}

// requireBikeTransport: the alternative without bike carriage departs first.
TEST(motis, itinerary_id_refresh_leg_alternatives_respect_require_bike) {
  auto const cfg = make_config(kLegAltBikeGtfs);
  auto d = import_test_data(cfg, "leg_alt_require_bike");

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const plan = routing(
      "?fromPlace=test_A&toPlace=test_B"
      "&time=2019-05-01T05:00Z&searchWindow=7200&detailedLegs=true"
      "&requireBikeTransport=true&numLegAlternatives=5");
  ASSERT_FALSE(plan.itineraries_.empty());
  auto const& plan_itin = plan.itineraries_.front();
  auto const& plan_leg = first_transit_leg(plan_itin);
  ASSERT_TRUE(plan_leg.alternatives_.has_value());
  ASSERT_FALSE(plan_leg.alternatives_->empty());
  EXPECT_TRUE(any_alt_trip_contains(plan_leg, "BIKE_YES"));
  EXPECT_FALSE(any_alt_trip_contains(plan_leg, "BIKE_NO"));

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = plan_itin.id_;
  query.numLegAlternatives_ = 5;
  query.requireBikeTransport_ = true;
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));
  auto const& refresh_leg = first_transit_leg(refreshed);

  EXPECT_TRUE(any_alt_trip_contains(refresh_leg, "BIKE_YES"));
  EXPECT_FALSE(any_alt_trip_contains(refresh_leg, "BIKE_NO"));
  // With the leg-alternatives query mirroring the `plan` endpoint's endpoint
  // match mode, the refreshed alternatives match the plan response exactly.
  EXPECT_EQ(plan_leg.alternatives_, refresh_leg.alternatives_);
}

// requireCarTransport: the alternative without car carriage departs first.
TEST(motis, itinerary_id_refresh_leg_alternatives_respect_require_car) {
  auto const cfg = make_config(kLegAltCarGtfs);
  auto d = import_test_data(cfg, "leg_alt_require_car");

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const plan = routing(
      "?fromPlace=test_A&toPlace=test_B"
      "&time=2019-05-01T05:00Z&searchWindow=7200&detailedLegs=true"
      "&requireCarTransport=true&numLegAlternatives=5");
  ASSERT_FALSE(plan.itineraries_.empty());
  auto const& plan_itin = plan.itineraries_.front();
  auto const& plan_leg = first_transit_leg(plan_itin);
  ASSERT_TRUE(plan_leg.alternatives_.has_value());
  ASSERT_FALSE(plan_leg.alternatives_->empty());
  EXPECT_TRUE(any_alt_trip_contains(plan_leg, "CAR_YES"));
  EXPECT_FALSE(any_alt_trip_contains(plan_leg, "CAR_NO"));

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = plan_itin.id_;
  query.numLegAlternatives_ = 5;
  query.requireCarTransport_ = true;
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));
  auto const& refresh_leg = first_transit_leg(refreshed);

  EXPECT_TRUE(any_alt_trip_contains(refresh_leg, "CAR_YES"));
  EXPECT_FALSE(any_alt_trip_contains(refresh_leg, "CAR_NO"));
  // With the leg-alternatives query mirroring the `plan` endpoint's endpoint
  // match mode, the refreshed alternatives match the plan response exactly.
  EXPECT_EQ(plan_leg.alternatives_, refresh_leg.alternatives_);
}

// pedestrianProfile=WHEELCHAIR: the wheelchair-inaccessible alternative
// departs first. Needs OSR so the wheelchair footpath profile is non-empty.
TEST(motis, itinerary_id_refresh_leg_alternatives_respect_wheelchair) {
  auto ec = std::error_code{};
  auto const path = fs::path{"test/data/itinerary_id"} / "leg_alt_wheelchair";
  fs::remove_all(path, ec);

  auto const cfg = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test",
                             {.path_ = std::string{kLegAltWheelchairGtfs}}}},
          },
      .street_routing_ = true,
      .osr_footpath_ = true,
  };
  import(cfg, path);
  auto d = data{path, cfg};

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const plan = routing(
      "?fromPlace=test_A&toPlace=test_B"
      "&time=2019-05-01T05:00Z&searchWindow=7200&detailedLegs=true"
      "&pedestrianProfile=WHEELCHAIR&useRoutedTransfers=true"
      "&numLegAlternatives=5");
  ASSERT_FALSE(plan.itineraries_.empty());
  auto const& plan_itin = plan.itineraries_.front();
  auto const& plan_leg = first_transit_leg(plan_itin);
  ASSERT_TRUE(plan_leg.alternatives_.has_value());
  ASSERT_FALSE(plan_leg.alternatives_->empty());
  EXPECT_TRUE(any_alt_trip_contains(plan_leg, "WC_YES"));
  EXPECT_FALSE(any_alt_trip_contains(plan_leg, "WC_NO"));

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = plan_itin.id_;
  query.numLegAlternatives_ = 5;
  query.pedestrianProfile_ = api::PedestrianProfileEnum::WHEELCHAIR;
  query.useRoutedTransfers_ = true;
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));
  auto const& refresh_leg = first_transit_leg(refreshed);

  EXPECT_TRUE(any_alt_trip_contains(refresh_leg, "WC_YES"));
  EXPECT_FALSE(any_alt_trip_contains(refresh_leg, "WC_NO"));
  // With the leg-alternatives query mirroring the `plan` endpoint's endpoint
  // match mode, the refreshed alternatives match the plan response exactly.
  EXPECT_EQ(plan_leg.alternatives_, refresh_leg.alternatives_);
}

// Partial-journey leg-alternatives: demonstrates that when a PT leg fails to
// reconstruct and becomes a dummy, the surviving segments compute alternatives
// without the surrounding dummy context. nigiri's get_leg_alternatives only
// sees the segment's journey, so the segment-internal first transit gets
// `has_prev=false` and the alt-query falls back to `q.start_` with
// `kIntermodal` instead of being pinned to the dummy's alighting stop with
// `kExact` (the way plan would). For a stop right next to the alighting one,
// `kIntermodal` over-extends via osr and surfaces an alternative that plan
// does NOT consider — the assertion comparing alt sets exposes the divergence.
constexpr auto kPartialAltSrcGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
ORIG,Origin,49.87336,8.62926,1,,
MID,Mid Hub,50.10593,8.66118,1,,
MID_ALT,Mid Alt,50.10739,8.66333,1,,
DEST,Destination,50.11404,8.67824,1,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R_ICE,DB,ICE1,,,106
R_L2,DB,L2,,,109
R_L2_ALT,DB,L2A,,,109

# trips.txt
route_id,service_id,trip_id,trip_headsign
R_ICE,S1,ICE1,
R_L2,S1,LOCAL2,
R_L2_ALT,S1,LOCAL2_ALT,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
ICE1,10:00:00,10:00:00,ORIG,0
ICE1,10:30:00,10:30:00,MID,1
LOCAL2,10:35:00,10:35:00,MID,0
LOCAL2,10:50:00,10:50:00,DEST,1
LOCAL2_ALT,10:40:00,10:40:00,MID_ALT,0
LOCAL2_ALT,10:55:00,10:55:00,DEST,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

constexpr auto kPartialAltTargetGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
ORIG,Origin,49.87336,8.62926,1,,
MID,Mid Hub,50.10593,8.66118,1,,
MID_ALT,Mid Alt,50.10739,8.66333,1,,
DEST,Destination,50.11404,8.67824,1,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R_L2,DB,L2,,,109
R_L2_ALT,DB,L2A,,,109

# trips.txt
route_id,service_id,trip_id,trip_headsign
R_L2,S1,LOCAL2,
R_L2_ALT,S1,LOCAL2_ALT,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
LOCAL2,10:35:00,10:35:00,MID,0
LOCAL2,10:50:00,10:50:00,DEST,1
LOCAL2_ALT,10:40:00,10:40:00,MID_ALT,0
LOCAL2_ALT,10:55:00,10:55:00,DEST,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

// Partial-journey leg-alternatives: when a PT leg fails to reconstruct and
// becomes a dummy, a surviving adjacent transit leg cannot bound its
// leg-alternatives query against the dummy side (the surrounding transit
// context is incomplete), so it produces no alternatives -- even though the
// full plan (where the dummy leg exists as a real transit leg) does offer them.
TEST(motis, itinerary_id_refresh_leg_alternatives_partial_journey_overreach) {
  auto const make_partial_cfg = [](std::string const& gtfs) {
    return config{
        .osm_ = {"test/resources/test_case.osm.pbf"},
        .timetable_ =
            config::timetable{
                .first_day_ = "2019-05-01",
                .num_days_ = 2,
                .datasets_ = {{"test", {.path_ = gtfs}}},
            },
        .street_routing_ = true,
        .osr_footpath_ = true,
    };
  };

  auto ec = std::error_code{};
  auto const src_path = fs::path{"test/data/itinerary_id"} / "partial_alt_src";
  auto const tgt_path =
      fs::path{"test/data/itinerary_id"} / "partial_alt_target";
  fs::remove_all(src_path, ec);
  fs::remove_all(tgt_path, ec);

  auto const src_cfg = make_partial_cfg(std::string{kPartialAltSrcGtfs});
  auto const tgt_cfg = make_partial_cfg(std::string{kPartialAltTargetGtfs});
  import(src_cfg, src_path);
  import(tgt_cfg, tgt_path);
  auto src_data = data{src_path, src_cfg};
  auto tgt_data = data{tgt_path, tgt_cfg};

  // Plan in source (where ICE1 exists) — yields journey with ICE1 → LOCAL2.
  auto const src_routing = utl::init_from<ep::routing>(src_data).value();
  auto const plan = src_routing(
      "/api/v6/plan"
      "?fromPlace=49.87420,8.62940"
      "&toPlace=50.11450,8.67900"
      "&time=2019-05-01T02:00Z"
      "&timetableView=true"
      "&useRoutedTransfers=true"
      "&numLegAlternatives=5"
      "&detailedLegs=true");
  ASSERT_FALSE(plan.itineraries_.empty());
  auto const& plan_itin = plan.itineraries_.front();
  auto const plan_pt_legs = transit_legs(plan_itin);
  ASSERT_EQ(2U, plan_pt_legs.size()) << "expected ICE1 + LOCAL2 in source plan";
  auto const& plan_local2 = *plan_pt_legs[1];

  // Reconstruct the same id in target (no ICE1) — ICE1 becomes a dummy and
  // LOCAL2 is reconstructed inside a partial-journey segment.
  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = plan_itin.id_;
  query.numLegAlternatives_ = 5;
  query.useRoutedTransfers_ = true;
  auto const tgt_refresh =
      utl::init_from<ep::refresh_itinerary>(tgt_data).value();
  auto const refreshed = tgt_refresh(query.to_url("?"));
  auto const refresh_pt_legs = transit_legs(refreshed);
  ASSERT_FALSE(refresh_pt_legs.empty());
  // Find LOCAL2 (the reconstructed-from-segment PT leg) in refreshed output.
  auto const refresh_local2_it =
      utl::find_if(refresh_pt_legs, [](api::Leg const* l) {
        return l->tripId_.has_value() &&
               l->tripId_->find("LOCAL2") != std::string::npos;
      });
  ASSERT_NE(refresh_local2_it, end(refresh_pt_legs));
  auto const& refresh_local2 = **refresh_local2_it;

  EXPECT_FALSE(alt_transit_trip_ids(plan_local2).empty())
      << "plan should offer LOCAL2_ALT as an alternative";
  EXPECT_TRUE(alt_transit_trip_ids(refresh_local2).empty())
      << "a leg adjacent to a dummy must not compute alternatives";
}

constexpr auto const kElevatorsActive = R"__([
  {"description":"FFM HBF Gleis 101/102","equipmentnumber":1010,
   "geocoordX":8.6628995,"geocoordY":50.1072933,"state":"ACTIVE",
   "type":"ELEVATOR"},
  {"description":"DA HBF Gleis 9/10","equipmentnumber":910,
   "geocoordX":8.6293117,"geocoordY":49.8725263,"state":"ACTIVE",
   "type":"ELEVATOR"}
])__";

constexpr auto const kElevatorsBlocked = R"__([
  {"description":"FFM HBF Gleis 101/102","equipmentnumber":1010,
   "geocoordX":8.6628995,"geocoordY":50.1072933,"state":"INACTIVE",
   "type":"ELEVATOR",
   "outOfService":[["2019-04-30T00:00:00Z","2019-05-02T00:00:00Z"]]},
  {"description":"DA HBF Gleis 9/10","equipmentnumber":910,
   "geocoordX":8.6293117,"geocoordY":49.8725263,"state":"INACTIVE",
   "type":"ELEVATOR",
   "outOfService":[["2019-04-30T00:00:00Z","2019-05-02T00:00:00Z"]]}
])__";

constexpr auto const kElevatorsFfmShortOutage = R"__([
  {"description":"FFM HBF Gleis 101/102","equipmentnumber":1010,
   "geocoordX":8.6628995,"geocoordY":50.1072933,"state":"INACTIVE",
   "type":"ELEVATOR",
   "outOfService":[["2019-04-30T22:30:00Z","2019-04-30T23:00:00Z"]]},
  {"description":"DA HBF Gleis 9/10","equipmentnumber":910,
   "geocoordX":8.6293117,"geocoordY":49.8725263,"state":"ACTIVE",
   "type":"ELEVATOR"}
])__";

constexpr auto const kElevatorGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code,wheelchair_boarding
DA,DA Hbf,49.87260,8.63085,1,,,1
DA_10,DA Hbf,49.87336,8.62926,0,DA,10,1
FFM,FFM Hbf,50.10701,8.66341,1,,,1
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10,1
FFM_101,FFM Hbf,50.10739,8.66333,0,FFM,101,1
FFM_HAUPT,FFM Hauptwache,50.11403,8.67835,1,,,1
FFM_HAUPT_S,FFM Hauptwache S,50.11404,8.67824,0,FFM_HAUPT,,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
ICE,DB,ICE,,,101
S3,DB,S3,,,109

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,wheelchair_accessible
ICE,S1,ICE,,,1
S3,S1,S3,,,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
ICE,00:35:00,00:35:00,DA_10,0,0,0
ICE,00:45:00,00:45:00,FFM_10,1,0,0
S3,01:15:00,01:15:00,FFM_101,1,0,0
S3,01:20:00,01:20:00,FFM_HAUPT_S,2,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

data& get_elevator_test_case() {
  static auto c = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .use_osm_stop_coordinates_ = true,
              .extend_missing_footpaths_ = false,
              .datasets_ = {{"test", {.path_ = std::string{kElevatorGTFS}}}}},
      .street_routing_ = true,
      .osr_footpath_ = true};
  static auto d = []() {
    auto ec = std::error_code{};
    fs::remove_all("test/data_itinerary_elevator", ec);
    import(c, "test/data_itinerary_elevator");
    return data{"test/data_itinerary_elevator", c};
  }();
  return d;
}

TEST(motis, itinerary_id_refresh_offset_blocked_by_elevator) {
  auto& d = get_elevator_test_case();
  d.init_rtt(date::sys_days{date::year{2019} / date::May / 1});
  d.rt_->e_ = std::make_unique<elevators>(
      *d.w_, nullptr, *d.elevator_nodes_,
      parse_fasta(std::string_view{kElevatorsActive}));

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const plan = routing(
      "/api/v6/plan"
      "?fromPlace=49.87420,8.62940"
      "&toPlace=50.10701,8.66341"
      "&time=2019-04-30T22:00Z"
      "&timetableView=true"
      "&pedestrianProfile=WHEELCHAIR"
      "&useRoutedTransfers=true"
      "&detailedLegs=true");
  ASSERT_FALSE(plan.itineraries_.empty())
      << "expected a plan with wheelchair access via DA Hbf";
  auto const& original = plan.itineraries_.front();
  ASSERT_FALSE(original.legs_.empty());
  ASSERT_EQ(api::ModeEnum::WALK, original.legs_.front().mode_)
      << "expected first leg to be a wheelchair walk";

  auto new_rtt = std::make_unique<nigiri::rt_timetable>(
      nigiri::rt_timetable{*d.rt_->rtt_});
  d.rt_->e_ = update_elevators(d.config_, d, kElevatorsBlocked, *new_rtt);
  d.rt_->rtt_ = std::move(new_rtt);

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = original.id_;
  query.pedestrianProfile_ = api::PedestrianProfileEnum::WHEELCHAIR;
  query.useRoutedTransfers_ = true;
  query.detailedLegs_ = true;
  query.maxMatchingDistance_ = 8.0;  // wheelchair requires lower matching dist
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));

  ASSERT_EQ(api::ModeEnum::WALK, refreshed.legs_.front().mode_);
  EXPECT_EQ(true, refreshed.legs_.front().cancelled_.value_or(false))
      << "first-mile access should be cancelled when the DA-HBF Gleis 9/10 "
         "elevator is out of service and the platform is unreachable";
}

TEST(motis, itinerary_id_refresh_transfer_blocked_by_elevator) {
  auto& d = get_elevator_test_case();
  d.rt_->e_ = std::make_unique<elevators>(
      *d.w_, nullptr, *d.elevator_nodes_,
      parse_fasta(std::string_view{kElevatorsActive}));
  d.init_rtt(date::sys_days{date::year{2019} / date::May / 1});

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const plan = routing(
      "/api/v6/plan"
      "?fromPlace=49.87420,8.62940"
      "&toPlace=50.11450,8.67900"
      "&time=2019-04-30T22:00Z"
      "&timetableView=true"
      "&pedestrianProfile=WHEELCHAIR"
      "&useRoutedTransfers=true"
      "&detailedTransfers=true"
      "&detailedLegs=true");
  ASSERT_FALSE(plan.itineraries_.empty())
      << "expected a wheelchair plan transferring at FFM Hbf";
  auto const& original = plan.itineraries_.front();

  // Locate the FFM_10 → FFM_101 transfer leg (a WALK after the ICE).
  auto const transfer_it = utl::find_if(original.legs_, [](api::Leg const& l) {
    return l.mode_ == api::ModeEnum::WALK &&
           l.from_.stopId_.value_or("") == "test_FFM_10" &&
           l.to_.stopId_.value_or("") == "test_FFM_101";
  });
  ASSERT_NE(transfer_it, end(original.legs_))
      << "plan does not transfer through FFM_10 → FFM_101";
  auto const original_transfer_idx =
      static_cast<std::size_t>(transfer_it - begin(original.legs_));

  // Block the FFM-HBF Gleis 101/102 elevator.
  auto new_rtt = std::make_unique<nigiri::rt_timetable>(
      nigiri::rt_timetable{*d.rt_->rtt_});
  d.rt_->e_ = update_elevators(d.config_, d, kElevatorsBlocked, *new_rtt);
  d.rt_->rtt_ = std::move(new_rtt);

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = original.id_;
  query.pedestrianProfile_ = api::PedestrianProfileEnum::WHEELCHAIR;
  query.useRoutedTransfers_ = true;
  query.detailedTransfers_ = true;
  query.detailedLegs_ = true;
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));
  ASSERT_GT(refreshed.legs_.size(), original_transfer_idx + 1U);

  auto const& refreshed_transfer = refreshed.legs_[original_transfer_idx];

  EXPECT_EQ(api::ModeEnum::WALK, refreshed_transfer.mode_);
  EXPECT_EQ(true, refreshed_transfer.cancelled_.value_or(false))
      << "wheelchair transfer FFM_10 -> FFM_101 should be cancelled when the "
         "FFM-HBF Gleis 101/102 elevator is out of service";
}

TEST(motis, itinerary_id_refresh_access_extended_by_elevator_wait) {
  auto& d = get_elevator_test_case();
  d.rt_->e_ = std::make_unique<elevators>(
      *d.w_, nullptr, *d.elevator_nodes_,
      parse_fasta(std::string_view{kElevatorsActive}));
  d.init_rtt(date::sys_days{date::year{2019} / date::May / 1});

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const plan = routing(
      "/api/v6/plan"
      "?fromPlace=49.87420,8.62940"
      "&toPlace=50.10701,8.66341"
      "&time=2019-04-30T22:00Z"
      "&timetableView=true"
      "&pedestrianProfile=WHEELCHAIR"
      "&useRoutedTransfers=true"
      "&detailedLegs=true");
  ASSERT_FALSE(plan.itineraries_.empty());
  auto const& original = plan.itineraries_.front();
  ASSERT_EQ(api::ModeEnum::WALK, original.legs_.front().mode_);
  auto const original_access_dur = original.legs_.front().duration_;
  auto const original_access_start =
      original.legs_.front().startTime_.get_unixtime_seconds();
  auto const original_access_end =
      original.legs_.front().endTime_.get_unixtime_seconds();

  auto new_rtt = std::make_unique<nigiri::rt_timetable>(
      nigiri::rt_timetable{*d.rt_->rtt_});
  d.rt_->e_ = update_elevators(d.config_, d, R"__([
  {"description":"DA HBF Gleis 9/10","equipmentnumber":910,
   "geocoordX":8.6293117,"geocoordY":49.8725263,"state":"INACTIVE",
   "type":"ELEVATOR",
   "outOfService":[["2019-04-30T22:20:00Z","2019-04-30T22:33:00Z"]]}
])__",
                               *new_rtt);
  d.rt_->rtt_ = std::move(new_rtt);

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = original.id_;
  query.pedestrianProfile_ = api::PedestrianProfileEnum::WHEELCHAIR;
  query.useRoutedTransfers_ = true;
  query.detailedLegs_ = true;
  query.maxMatchingDistance_ = 8.0;
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));
  auto const& access = refreshed.legs_.front();

  EXPECT_EQ(api::ModeEnum::WALK, access.mode_);
  EXPECT_NE(true, access.cancelled_.value_or(false))
      << "access should remain feasible while the elevator is only briefly out";
  EXPECT_GT(access.duration_, original_access_dur)
      << "the elevator outage should extend the access by the platform wait";
  EXPECT_LT(access.startTime_.get_unixtime_seconds(), original_access_start)
      << "the access should start earlier to beat the outage";
  EXPECT_EQ(original_access_end, access.endTime_.get_unixtime_seconds())
      << "the access should still board the same train";
}

TEST(motis, itinerary_id_refresh_transfer_extended_by_elevator_wait) {
  auto& d = get_elevator_test_case();
  d.rt_->e_ = std::make_unique<elevators>(
      *d.w_, nullptr, *d.elevator_nodes_,
      parse_fasta(std::string_view{kElevatorsActive}));
  d.init_rtt(date::sys_days{date::year{2019} / date::May / 1});

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const plan = routing(
      "/api/v6/plan"
      "?fromPlace=49.87420,8.62940"
      "&toPlace=50.11450,8.67900"
      "&time=2019-04-30T22:00Z"
      "&timetableView=true"
      "&pedestrianProfile=WHEELCHAIR"
      "&useRoutedTransfers=true"
      "&detailedTransfers=true"
      "&detailedLegs=true");
  ASSERT_FALSE(plan.itineraries_.empty());
  auto const& original = plan.itineraries_.front();
  auto const transfer_it = utl::find_if(original.legs_, [](api::Leg const& l) {
    return l.mode_ == api::ModeEnum::WALK &&
           l.from_.stopId_.value_or("") == "test_FFM_10" &&
           l.to_.stopId_.value_or("") == "test_FFM_101";
  });
  ASSERT_NE(transfer_it, end(original.legs_));
  auto const idx =
      static_cast<std::size_t>(transfer_it - begin(original.legs_));
  auto const original_duration = transfer_it->duration_;

  // Take the FFM elevator out only for [22:30, 23:00) UTC (covers the 22:45
  // FFM_10 arrival, reopens before the 23:15 S3 departure).
  auto new_rtt = std::make_unique<nigiri::rt_timetable>(
      nigiri::rt_timetable{*d.rt_->rtt_});
  d.rt_->e_ =
      update_elevators(d.config_, d, kElevatorsFfmShortOutage, *new_rtt);
  d.rt_->rtt_ = std::move(new_rtt);

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = original.id_;
  query.pedestrianProfile_ = api::PedestrianProfileEnum::WHEELCHAIR;
  query.useRoutedTransfers_ = true;
  query.detailedTransfers_ = true;
  query.detailedLegs_ = true;
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));
  ASSERT_GT(refreshed.legs_.size(), idx + 1U);
  auto const& transfer = refreshed.legs_[idx];
  auto const& next_pt = refreshed.legs_[idx + 1U];

  EXPECT_EQ(api::ModeEnum::WALK, transfer.mode_);
  EXPECT_NE(true, transfer.cancelled_.value_or(false))
      << "transfer should remain feasible while the elevator is only briefly "
         "out of service";
  EXPECT_GT(transfer.duration_, original_duration)
      << "the elevator outage should extend the transfer by the platform wait";
  EXPECT_LE(transfer.endTime_.get_unixtime_seconds(),
            next_pt.startTime_.get_unixtime_seconds())
      << "the extended transfer must still make the next PT departure";
}

TEST(motis, itinerary_id_refresh_leg_alternatives_intermediate_leg) {
  auto const cfg = make_config(kLegAltIntermediateGtfs);
  auto d = import_test_data(cfg, "leg_alt_intermediate");

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const plan = routing(
      "?fromPlace=test_A&toPlace=test_D"
      "&time=2019-05-01T05:00Z&searchWindow=7200&detailedLegs=true"
      "&transitModes=REGIONAL_RAIL&numLegAlternatives=5");
  ASSERT_FALSE(plan.itineraries_.empty());
  auto const& plan_itin = plan.itineraries_.front();
  auto const plan_transit = transit_legs(plan_itin);
  ASSERT_EQ(3U, plan_transit.size());

  // Middle leg: a transit leg both before (L1) and after (L3) -> exact
  // endpoint matching. Plan already drops the bus alternative.
  auto const& plan_mid = *plan_transit[1];
  ASSERT_TRUE(plan_mid.alternatives_.has_value());
  ASSERT_FALSE(plan_mid.alternatives_->empty());
  EXPECT_TRUE(any_alt_trip_contains(plan_mid, "L2_REG"));
  EXPECT_FALSE(any_alt_trip_contains(plan_mid, "L2_BUS"));

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = plan_itin.id_;
  query.numLegAlternatives_ = 5;
  query.transitModes_ = {api::ModeEnum::REGIONAL_RAIL};
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));
  auto const refresh_transit = transit_legs(refreshed);
  ASSERT_EQ(3U, refresh_transit.size());

  auto const& refresh_mid = *refresh_transit[1];
  EXPECT_TRUE(any_alt_trip_contains(refresh_mid, "L2_REG"));
  EXPECT_FALSE(any_alt_trip_contains(refresh_mid, "L2_BUS"));

  // Every transit leg's alternatives match the plan response - including the
  // intermediate leg, whose endpoints are matched exactly.
  for (auto i = std::size_t{0}; i != 3U; ++i) {
    EXPECT_EQ(plan_transit[i]->alternatives_,
              refresh_transit[i]->alternatives_);
  }
}

// Cross-timezone journey with first/last mile WALK access. The first transit
// runs in Europe/Berlin, the second in Europe/Vienna. The first-mile START and
// the last-mile END are bare coordinates and carry no timezone of their own, so
// they must inherit from the *nearest* transit stop: START -> Europe/Berlin,
// END -> Europe/Vienna. Regression test for the refresh timezone propagation
// that previously applied the *first* timezone found to every gap, leaking the
// origin timezone onto the destination's last mile.
constexpr auto kCrossTzGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin
OEBB,OEBB,https://oebb.at,Europe/Vienna

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,A,49.87336,8.62926,0,
B,B,49.99359,8.65677,0,
C,C,50.10739,8.66333,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
DE_R,DB,DE,,,3
AT_R,OEBB,AT,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign
DE_R,S1,DE_TRIP,
AT_R,S1,AT_TRIP,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
DE_TRIP,09:00:00,09:00:00,A,0
DE_TRIP,09:30:00,09:30:00,B,1
AT_TRIP,09:40:00,09:40:00,B,0
AT_TRIP,10:10:00,10:10:00,C,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

TEST(motis, itinerary_id_refresh_timezone_propagation) {
  auto ec = std::error_code{};
  auto const path = fs::path{"test/data/itinerary_id"} / "cross_tz";
  fs::remove_all(path, ec);

  auto const cfg = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test", {.path_ = std::string{kCrossTzGtfs}}}},
          },
      .street_routing_ = true,
  };
  import(cfg, path);
  auto d = data{path, cfg};

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const plan = routing(
      "?fromPlace=49.87298,8.63006&toPlace=50.10720,8.66320"
      "&time=2019-05-01T06:00Z&searchWindow=7200&detailedLegs=true");
  ASSERT_FALSE(plan.itineraries_.empty());
  auto const& plan_itin = plan.itineraries_.front();
  ASSERT_GE(plan_itin.legs_.size(), 2U);

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = plan_itin.id_;
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));
  ASSERT_GE(refreshed.legs_.size(), 2U);

  // START (first mile, before the Berlin transit) -> Europe/Berlin.
  EXPECT_EQ(std::optional{std::string{"Europe/Berlin"}},
            refreshed.legs_.front().from_.tz_);
  // END (last mile, after the Vienna transit) -> Europe/Vienna, NOT the
  // origin's Europe/Berlin.
  EXPECT_EQ(std::optional{std::string{"Europe/Vienna"}},
            refreshed.legs_.back().to_.tz_);

  // The refresh timezones match what the plan endpoint produced.
  EXPECT_EQ(plan_itin.legs_.front().from_.tz_,
            refreshed.legs_.front().from_.tz_);
  EXPECT_EQ(plan_itin.legs_.back().to_.tz_, refreshed.legs_.back().to_.tz_);
}

// Single transit leg (no previous, no next transit) reached via a first-mile
// WALK. A parallel route departs at the SAME time as the chosen trip. The plan
// offers it as an alternative. The refresh endpoint must too - which requires
// anchoring the alternatives search at the journey's ORIGIN departure (incl.
// first-mile access time), not at the transit leg's departure. Anchoring at the
// transit departure starts the forward search one access-walk too late, so it
// misses the same-time `MAIN_ALT` and only finds the later `LATE`.
constexpr auto kFirstMileAnchorGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,A,49.87336,8.62926,0,
Z,Z,50.10739,8.66333,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
MAIN_R,DB,MAIN,,,3
MAIN_ALT_R,DB,MALT,,,3
LATE_R,DB,LATE,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign
MAIN_R,S1,MAIN,
MAIN_ALT_R,S1,MAIN_ALT,
LATE_R,S1,LATE,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
MAIN,09:30:00,09:30:00,A,0
MAIN,10:00:00,10:00:00,Z,1
MAIN_ALT,09:30:00,09:30:00,A,0
MAIN_ALT,10:00:00,10:00:00,Z,1
LATE,09:45:00,09:45:00,A,0
LATE,10:15:00,10:15:00,Z,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

TEST(motis, itinerary_id_refresh_leg_alternatives_first_mile_anchor) {
  auto ec = std::error_code{};
  auto const path = fs::path{"test/data/itinerary_id"} / "first_mile_anchor";
  fs::remove_all(path, ec);

  auto const cfg = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test",
                             {.path_ = std::string{kFirstMileAnchorGtfs}}}},
          },
      .street_routing_ = true,
  };
  import(cfg, path);
  auto d = data{path, cfg};

  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const plan = routing(
      "?fromPlace=49.87298,8.63006&toPlace=50.10720,8.66320"
      "&time=2019-05-01T07:00Z&searchWindow=7200&detailedLegs=true"
      "&numLegAlternatives=5");
  ASSERT_FALSE(plan.itineraries_.empty());
  auto const& plan_itin = plan.itineraries_.front();
  auto const& plan_leg = first_transit_leg(plan_itin);
  ASSERT_TRUE(plan_leg.alternatives_.has_value());

  // True if any alternative has a transit leg departing at the same time as the
  // chosen leg - i.e. an alternative reachable as early as the original
  // journey.
  auto const has_same_time_alt = [](api::Leg const& leg) {
    return leg.alternatives_.has_value() &&
           utl::any_of(*leg.alternatives_, [&](auto const& alt) {
             return utl::any_of(alt, [&](api::Leg const& al) {
               return al.tripId_.has_value() && !al.tripId_->empty() &&
                      *al.startTime_ == *leg.startTime_;
             });
           });
  };

  // The plan offers the same-departure-time parallel trip as an alternative.
  EXPECT_TRUE(has_same_time_alt(plan_leg));

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = plan_itin.id_;
  query.numLegAlternatives_ = 5;
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));
  auto const& refresh_leg = first_transit_leg(refreshed);

  // The refresh must reproduce the same-time alternative; anchoring at the
  // transit departure (instead of the origin departure) starts the search one
  // access-walk too late and drops it.
  EXPECT_TRUE(has_same_time_alt(refresh_leg))
      << "refresh must include the same-time alternative reachable as early as "
         "the original journey";
  EXPECT_EQ(alt_transit_trip_ids(plan_leg), alt_transit_trip_ids(refresh_leg));
}

// Transit + CAR last mile over a ONE-WAY street network (test/resources/
// oneway_car.osm): the alighting stop S reaches the destination D via a short
// one-way street (S -> D), while D -> S requires a long detour. The
// post-transit (egress) CAR offset must therefore be routed *backward from the
// destination* (giving the short S -> D duration), exactly as the plan endpoint
// does. Routing it forward (D -> S) yields the long detour, so the refreshed
// last-mile CAR leg would diverge from the plan's. Regression test for the
// flipped offset direction - this DOES fail when the direction is reverted.
constexpr auto kCarEgressGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,A,49.90000,8.00000,0,
S,S,50.00000,8.00000,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
MAIN_R,DB,MAIN,,,106

# trips.txt
route_id,service_id,trip_id,trip_headsign,cars_allowed
MAIN_R,S1,MAIN,,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
MAIN,09:30:00,09:30:00,A,0
MAIN,09:40:00,09:40:00,S,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

TEST(motis, itinerary_id_refresh_car_egress_direction) {
  auto ec = std::error_code{};
  auto const path = fs::path{"test/data/itinerary_id"} / "car_egress_dir";
  fs::remove_all(path, ec);

  auto const cfg = config{
      .osm_ = {"test/resources/oneway_car.osm"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test", {.path_ = std::string{kCarEgressGtfs}}}},
          },
      .street_routing_ = true,
  };
  import(cfg, path);
  auto d = data{path, cfg};

  auto const routing = utl::init_from<ep::routing>(d).value();
  // Board at stop A, ride to S, then CAR to the destination D (50.0030,8.0).
  // A is on a disconnected street stub, so transit is the only way into the
  // egress area (no competing straight-line walk).
  auto const plan = routing(
      "?fromPlace=test_A&toPlace=50.00300,8.00000"
      "&time=2019-05-01T07:00Z&searchWindow=7200&detailedLegs=true"
      "&postTransitModes=CAR&maxPostTransitTime=3600");
  ASSERT_FALSE(plan.itineraries_.empty());
  auto const& plan_itin = plan.itineraries_.front();
  ASSERT_EQ(api::ModeEnum::CAR, plan_itin.legs_.back().mode_)
      << "expected a CAR last mile";

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = plan_itin.id_;
  query.postTransitModes_ = {api::ModeEnum::CAR};
  query.maxPostTransitTime_ = 3600;
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));
  ASSERT_EQ(api::ModeEnum::CAR, refreshed.legs_.back().mode_);

  // The refreshed CAR egress (routed backward from the destination, i.e. the
  // short one-way S -> D) matches the plan's. The flipped (forward) direction
  // would pick the long D -> S detour and give a different duration.
  EXPECT_EQ(plan_itin.legs_.back().duration_, refreshed.legs_.back().duration_);
}

// First-mile access via the DA Hbf Gleis 9/10 elevator (wheelchair), single
// transit leg, with a parallel same-time alternative trip. The id is generated
// while the elevator is ACTIVE (short first-mile access). At refresh the
// elevator has a *temporary* outage covering the boarding window, which
// inflates the first-mile access ("leave before the outage, wait on the
// platform") rather than cancelling it. The alternatives search must anchor at
// the journey's origin departure computed from this *current* (inflated) access
// - looked up via td-offsets - not from the short access span encoded in the
// id. Otherwise the search starts too late and drops the same-time `ICE_ALT`.
constexpr auto const kTdAnchorGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code,wheelchair_boarding
DA,DA Hbf,49.87260,8.63085,1,,,1
DA_10,DA Hbf,49.87336,8.62926,0,DA,10,1
FFM,FFM Hbf,50.10701,8.66341,1,,,1
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
ICE_R,DB,ICE,,,101
ICE_ALT_R,DB,ICEA,,,101
ICE_LATE_R,DB,ICEL,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,wheelchair_accessible
ICE_R,S1,ICE,,1
ICE_ALT_R,S1,ICE_ALT,,1
ICE_LATE_R,S1,ICE_LATE,,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
ICE,00:35:00,00:35:00,DA_10,0
ICE,00:45:00,00:45:00,FFM_10,1
ICE_ALT,00:35:00,00:35:00,DA_10,0
ICE_ALT,00:45:00,00:45:00,FFM_10,1
ICE_LATE,00:55:00,00:55:00,DA_10,0
ICE_LATE,01:05:00,01:05:00,FFM_10,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

constexpr auto const kElevatorsDaShortOutage = R"__([
  {"description":"FFM HBF Gleis 101/102","equipmentnumber":1010,
   "geocoordX":8.6628995,"geocoordY":50.1072933,"state":"ACTIVE",
   "type":"ELEVATOR"},
  {"description":"DA HBF Gleis 9/10","equipmentnumber":910,
   "geocoordX":8.6293117,"geocoordY":49.8725263,"state":"INACTIVE",
   "type":"ELEVATOR",
   "outOfService":[["2019-04-30T22:20:00Z","2019-04-30T22:50:00Z"]]}
])__";

TEST(motis, itinerary_id_refresh_first_mile_td_anchor) {
  auto ec = std::error_code{};
  auto const path = fs::path{"test/data/itinerary_id"} / "first_mile_td_anchor";
  fs::remove_all(path, ec);

  auto const cfg = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .use_osm_stop_coordinates_ = true,
              .extend_missing_footpaths_ = false,
              .datasets_ = {{"test", {.path_ = std::string{kTdAnchorGtfs}}}}},
      .street_routing_ = true,
      .osr_footpath_ = true};
  import(cfg, path);
  auto d = data{path, cfg};
  d.init_rtt(date::sys_days{date::year{2019} / date::May / 1});
  d.rt_->e_ = std::make_unique<elevators>(
      *d.w_, nullptr, *d.elevator_nodes_,
      parse_fasta(std::string_view{kElevatorsActive}));

  auto const routing = utl::init_from<ep::routing>(d).value();
  // While the elevator is active the first-mile access is a short wheelchair
  // walk; the id therefore encodes a short access span.
  auto const plan = routing(
      "?fromPlace=49.87420,8.62940&toPlace=50.10593,8.66118"
      "&time=2019-04-30T22:00Z&searchWindow=7200&timetableView=true"
      "&pedestrianProfile=WHEELCHAIR&useRoutedTransfers=true"
      "&maxMatchingDistance=8&detailedLegs=true&numLegAlternatives=5");
  ASSERT_FALSE(plan.itineraries_.empty());
  auto const& original = plan.itineraries_.front();
  ASSERT_EQ(api::ModeEnum::WALK, original.legs_.front().mode_)
      << "expected a wheelchair first-mile walk";

  // Take the DA Hbf Gleis 9/10 elevator out only for [22:20, 22:50) UTC: it
  // covers the 22:35 boarding but reopens later, so the first-mile access is
  // inflated ("leave before the outage and wait on the platform"), not
  // cancelled.
  auto new_rtt = std::make_unique<nigiri::rt_timetable>(
      nigiri::rt_timetable{*d.rt_->rtt_});
  d.rt_->e_ = update_elevators(d.config_, d, kElevatorsDaShortOutage, *new_rtt);
  d.rt_->rtt_ = std::move(new_rtt);

  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = original.id_;
  query.pedestrianProfile_ = api::PedestrianProfileEnum::WHEELCHAIR;
  query.useRoutedTransfers_ = true;
  query.maxMatchingDistance_ = 8.0;
  query.numLegAlternatives_ = 5;
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto const refreshed = refresh(query.to_url("?"));
  auto const& refresh_leg = first_transit_leg(refreshed);

  // Sanity: the temporary outage inflated the first-mile access well beyond the
  // ~2 min active-elevator walk.
  EXPECT_GT(refreshed.legs_.front().duration_, 600)
      << "DA elevator outage should inflate the first-mile access";

  // The same-time alternative is only found when the alternatives search
  // anchors at the journey's CURRENT (inflated) origin departure - looked up
  // via the current td-offsets - rather than the short access span encoded in
  // the id. Anchoring at the id's span starts the search too late and drops it.
  auto const has_same_time_alt = [](api::Leg const& leg) {
    return leg.alternatives_.has_value() &&
           utl::any_of(*leg.alternatives_, [&](auto const& alt) {
             return utl::any_of(alt, [&](api::Leg const& al) {
               return al.tripId_.has_value() && !al.tripId_->empty() &&
                      *al.startTime_ == *leg.startTime_;
             });
           });
  };
  EXPECT_TRUE(has_same_time_alt(refresh_leg))
      << "refresh must include the same-time alternative reachable as early as "
         "the original journey given the inflated first-mile access";
}

// MOTIS <= 2.10.x encoded block / interlined transit continuations (the vehicle
// continues as a new trip at a shared stop) as two ADJACENT transit legs, with
// no transfer leg between them. The current reconstruction requires strictly
// alternating offset / transit / transfer legs, so those ids were rejected with
// HTTP 400 ("two consecutive transit legs"). `decode_itinerary_id` now joins
// such interlined legs into a single transit leg (as the current format does),
// keeping old ids decodable. Regression test for a real 2.10.2 id: a 20-leg
// FI -> SE -> NO -> DE -> FR -> ES journey whose two `1A` bus legs (legs 3 & 4)
// share stop `fi-fintraffic_313413` and vehicle block `1001080207`.
constexpr auto const kLegacyBlockItineraryId =
    "ClUhWGw40VMVTkApEnHO44vxOEAyFGZpLWZpbnRyYWZmaWNfMzU3MzcyOTeTb7a5FU5AQfWfNT"
    "/+"
    "7jhASKSQltEGUPSVltEGWgRXQUxLcQAAAAAAAAAACqoBCgNPQjESLjIwMjYwNjA3XzE4OjE1X2"
    "ZpLWZpbnRyYWZmaWNfMTI1NzVfMjQ1MzU1LjQ4MzIaFGZpLWZpbnRyYWZmaWNfMzU3MzcyITeT"
    "b7a5FU5AKfWfNT/+7jhAMhRmaS1maW50cmFmZmljXzMxMjgwNjmfyJOkazpOQEE9sOO/"
    "QEQ2QEj0lZbRBlDYzZbRBloDQlVTYAFpAAAAAAAAAABxAAAAAAAAAAAKdBoUZmktZmludHJhZm"
    "ZpY18zMTI4MDYhn8iTpGs6TkApPbDjv0BENkAyFGZpLWZpbnRyYWZmaWNfMzEyNTE1OW40gLdA"
    "Ok5AQWBY/"
    "nxbRDZASMDQltEGULDSltEGWgRXQUxLaQAAAAAAAAAAcQAAAAAAAAAACrIBCgIxQRI3MjAyNjA"
    "2MDdfMjA6MDdfZmktZmludHJhZmZpY18xMDAwM18wMDAxMzA3M19fMTAwMTA4MDIwNxoUZmktZ"
    "mludHJhZmZpY18zMTI1MTUhbjSAt0A6TkApYFj+"
    "fFtENkAyFGZpLWZpbnRyYWZmaWNfMzEzNDEzOUbOwp52Ok5AQUasxacARDZASLDSltEGUOzSlt"
    "EGWgNCVVNgAWkAAAAAAAAAAHEAAAAAAAAAAAqyAQoCMUESNzIwMjYwNjA3XzIwOjI1X2ZpLWZp"
    "bnRyYWZmaWNfMTAwMDNfMDAwMTMwNTZfXzEwMDEwODAyMDcaFGZpLWZpbnRyYWZmaWNfMzEzND"
    "EzIUbOwp52Ok5AKUasxacARDZAMhRmaS1maW50cmFmZmljXzMxNDA0NTmrlQm/"
    "1DdOQEFMVdriGjs2QEjs0pbRBlDw2ZbRBloDQlVTYAFpAAAAAAAAAABxAAAAAAAAAAAKdBoUZm"
    "ktZmludHJhZmZpY18zMTQwNDUhq5UJv9Q3TkApTFXa4ho7NkAyFGZpLWZpbnRyYWZmaWNfMzc5"
    "OTI2OXNoke18N05AQX/"
    "7OnDOODZASIzeltEGUPTgltEGWgRXQUxLaQAAAAAAAAAAcQAAAAAAAAAACtEBCg9UdXJrdS1Td"
    "G9ja2hvbG0SRzIwMjYwNjA3XzIwOjU1X2ZpLWZpbnRyYWZmaWNfMTAxODRfZGM3Y2YxZDAtMWM"
    "1My00NThkLThkNDktNzc1NTEwZjBmOGM1GhRmaS1maW50cmFmZmljXzM3OTkyNiFzaJHtfDdOQ"
    "Cl/+zpwzjg2QDIUZmktZmludHJhZmZpY18zNzk5Mjc5/"
    "yH99nWoTUBBmggbnl4ZMkBI9OCW0QZQyIqZ0QZaBUZFUlJZYAFpAAAAAAAAAABxAAAAAAAAAAA"
    "KdhoUZmktZmludHJhZmZpY18zNzk5Mjch/"
    "yH99nWoTUApmggbnl4ZMkAyFnNlLVRyYWZpa2xhYl83NDAwNDYxNDE5PzVeukmoTUBBC89LxcY"
    "YMkBIoI+"
    "Z0QZQzJGZ0QZaBFdBTEtpAAAAAAAAAABxAAAAAAAAAAAKqQEKAjUzEioyMDI2MDYwOF8wNjo0M"
    "l9zZS1UcmFmaWtsYWJfMjI3NTAwNTMwMDMyNDMaFnNlLVRyYWZpa2xhYl83NDAwNDYxNDEhPzV"
    "eukmoTUApC89LxcYYMkAyFnNlLVRyYWZpa2xhYl83NDAwNDYwODM5t9RBXg+qTUBBf/"
    "EMGvoPMkBIzJGZ0QZQ2JeZ0QZaA0JVU2ABaQAAAAAAAAAAcQAAAAAAAAAACnoaFnNlLVRyYWZp"
    "a2xhYl83NDAwNDYwODMht9RBXg+qTUApf/"
    "EMGvoPMkAyGG5vLUVudHVyX05TUjpRdWF5OjEwMDM5MDk0ngjiPKpNQEHWjXdHxg4yQEj89JnR"
    "BlCg+"
    "JnRBloEV0FMS2kAAAAAAAAAAHEAAAAAAAAAAArcAQoCSUMSTzIwMjYwNjA4XzEwOjI0X25vLUV"
    "udHVyX1NOVDpTZXJ2aWNlSm91cm5leTo3NTUzYjZkYy0zMGVhLTVkYjYtYTUyMC0wY2YzYTlkM"
    "WY1NTcaGG5vLUVudHVyX05TUjpRdWF5OjEwMDM5MCE0ngjiPKpNQCnWjXdHxg4yQDIYbm8tRW5"
    "0dXJfTlNSOlF1YXk6MTExMTAwOZYjZCDPxkpAQVjIXBlUAyRASKD4mdEGUIy9nNEGWg1MT05HX"
    "0RJU1RBTkNFYAFpAAAAAAAAAABxAAAAAAAA8L8KexoYbm8tRW50dXJfTlNSOlF1YXk6MTExMTA"
    "wIZYjZCDPxkpAKVjIXBlUAyRAMhdkZS1ERUxGSV9kZTowMjAwMDoxMDk1MDmSsdr8v8ZKQEGcG"
    "JKTiQMkQEjE+Z3RBlDw+"
    "53RBloEV0FMS2kAAAAAAADwv3EAAAAAAADwvwq2AQoHSUNFIDU3MxIiMjAyNjA2MDlfMDQ6NDR"
    "fZGUtREVMRklfMzIyNTIzMjUyNBoXZGUtREVMRklfZGU6MDIwMDA6MTA5NTAhkrHa/L/"
    "GSkApnBiSk4kDJEAyGmRlLURFTEZJX2RlOjA4MjIyOjI0MTc6NTo1OX46HjNQvUhAQVgiUP2D8"
    "CBASPD7ndEGUMyOn9EGWg5ISUdIU1BFRURfUkFJTGABaQAAAAAAAPC/"
    "cQAAAAAAAAAACpABGhpkZS1ERUxGSV9kZTowODIyMjoyNDE3OjU6NSF+Oh4zUL1IQClYIlD9g/"
    "AgQDIqZnItaG9yYWlyZXMtc25jZl9TdG9wUG9pbnQ6T0NFSUNFLTgwMTQwMDg3OejZrPpcvUhA"
    "QTlnRGlv8CBASKCToNEGUJiUoNEGWgRXQUxLaQAAAAAAAAAAcQAAAAAAAAAACroCCgQ2NzFBEo"
    "YBMjAyNjA2MDlfMTQ6MDJfZnItaG9yYWlyZXMtc25jZl9PQ0VTTjk1ODBGMTE4N19GOklDRTpG"
    "UjpMaW5lOjo4ODRBNjYyQS05Q0M5LTQxMzMtOTRBNS04OEI0QUU1MjA2RkQ6OjgwMTEwNjg0Oj"
    "g3NzUxMDA4OjEzOjIxNTE6MjAyNjA2MjEaKmZyLWhvcmFpcmVzLXNuY2ZfU3RvcFBvaW50Ok9D"
    "RUlDRS04MDE0MDA4NyHo2az6XL1IQCk5Z0Rpb/"
    "AgQDIqZnItaG9yYWlyZXMtc25jZl9TdG9wUG9pbnQ6T0NFSUNFLTg3MzE4OTY0OedvQiEC9kVA"
    "QUHV6NUAJRNASJiUoNEGULzLodEGWg1SRUdJT05BTF9SQUlMYAFpAAAAAAAAAABxAAAAAAAA8D"
    "8KmgEaKmZyLWhvcmFpcmVzLXNuY2ZfU3RvcFBvaW50Ok9DRUlDRS04NzMxODk2NCHnb0IhAvZF"
    "QClB1ejVACUTQDIkZnItaG9yYWlyZXMtYXZlLWVzcGFnbmUtZnJhbmNlXzg3ODE0IedvQiEC9k"
    "VAQSsWvymsJBNASJSLpNEGUIyMpNEGWgRXQUxLaQAAAAAAAPA/cQAAAAAAAPA/"
    "CugBCgdBVkUgSU5UEj4yMDI2MDYxMF8wODowMV9mci1ob3JhaXJlcy1hdmUtZXNwYWduZS1mcm"
    "FuY2VfMDk3MzAxMjAyNi0wNi0wORokZnItaG9yYWlyZXMtYXZlLWVzcGFnbmUtZnJhbmNlXzg3"
    "ODE0IedvQiEC9kVAKSsWvymsJBNAMiRmci1ob3JhaXJlcy1hdmUtZXNwYWduZS1mcmFuY2VfNj"
    "AwMDA5o+TVOQY0REBBMh06Pe+"
    "GDcBIjIyk0QZQmOSl0QZaDVJFR0lPTkFMX1JBSUxgAWkAAAAAAADwP3EAAAAAAAAAAAqJARokZ"
    "nItaG9yYWlyZXMtYXZlLWVzcGFnbmUtZnJhbmNlXzYwMDAwIaPk1TkGNERAKTIdOj3vhg3AMhl"
    "lcy1DZXJjYW7DrWFzLVJlbmZlXzE4MDAwOSPzyB8MNERAQRCyLJj4gw3ASJjkpdEGUJDlpdEGW"
    "gRXQUxLaQAAAAAAAAAAcQAAAAAAAAAACr4BCgJDMxIvMjAyNjA2MTBfMTU6MzBfZXMtQ2VyY2F"
    "uw61hcy1SZW5mZV8xMDU4WDc4MDY3QzMaGWVzLUNlcmNhbsOtYXMtUmVuZmVfMTgwMDAhI/"
    "PIHww0REApELIsmPiDDcAyGWVzLUNlcmNhbsOtYXMtUmVuZmVfMTgxMDE5O2h23Vs1REBBSP1Q"
    "xH2fDcBIzOWl0QZQxOal0QZaDVJFR0lPTkFMX1JBSUxgAWkAAAAAAAAAAHEAAAAAAAAAAApaGh"
    "llcy1DZXJjYW7DrWFzLVJlbmZlXzE4MTAxITtodt1bNURAKUj9UMR9nw3AOd7H0RxZNURAQXwO"
    "LEfIoA3ASMTmpdEGUIDnpdEGWgRXQUxLaQAAAAAAAAAA";

TEST(motis, itinerary_id_decode_legacy_block_transfer) {
  auto const decoded = decode_itinerary_id(kLegacyBlockItineraryId);
  ASSERT_GT(decoded.legs_size(), 0);

  // After normalization no two consecutive transit legs remain, so
  // verify_leg_structure (and thus the reconstruction) accepts the id instead
  // of rejecting it as "two consecutive transit legs".
  for (auto i = 1; i < decoded.legs_size(); ++i) {
    auto const prev_transit = !decoded.legs(i - 1).trip_id().empty();
    auto const cur_transit = !decoded.legs(i).trip_id().empty();
    EXPECT_FALSE(prev_transit && cur_transit)
        << "consecutive transit legs at index " << (i - 1) << "/" << i;
  }

  // The two `1A` block legs were joined into one: 20 legs -> 19.
  ASSERT_EQ(19, decoded.legs_size());

  // The merged leg (index 3: WALK, 1A bus, WALK, [merged 1A]) keeps the entered
  // trip and spans from the first leg's origin to the last leg's destination.
  auto const& merged = decoded.legs(3);
  EXPECT_EQ("fi-fintraffic_312515", merged.from_id());
  EXPECT_EQ("fi-fintraffic_314045", merged.to_id());
  EXPECT_EQ("20260607_20:07_fi-fintraffic_10003_00013073__1001080207",
            merged.trip_id());
}
