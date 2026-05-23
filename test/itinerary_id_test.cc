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

#include "boost/json.hpp"

#include "fmt/format.h"

#include "gtest/gtest.h"

#include "utl/helpers/algorithm.h"
#include "utl/init_from.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/refresh_itinerary.h"
#include "motis/endpoints/routing.h"
#include "motis/endpoints/stop_times.h"
#include "motis/endpoints/trip.h"
#include "motis/import.h"
#include "motis/itinerary_id.h"

#include "net/base64.h"

#include "nigiri/rt/gtfsrt_update.h"

#include "generated/itinerary_id/itinerary_id.pb.h"

#include "./util.h"

using namespace motis;
namespace fs = std::filesystem;
namespace n = nigiri;

std::string generate_itinerary_id(api::Itinerary const& x) {
  return motis::generate_itinerary_id(x, {}, {});
}

TEST(motis, itinerary_id_distinguishes_level_zero_from_no_level) {
  auto const t0 =
      openapi::date_time_t{std::chrono::sys_seconds{std::chrono::seconds{0}}};
  auto itinerary = api::Itinerary{
      .duration_ = 0, .startTime_ = t0, .endTime_ = t0, .transfers_ = 0};
  itinerary.legs_.push_back(
      api::Leg{.mode_ = api::ModeEnum::WALK,
               .from_ = api::Place{.lat_ = 1.0, .lon_ = 2.0, .level_ = 0.0},
               .to_ = api::Place{.lat_ = 3.0, .lon_ = 4.0},
               .duration_ = 0,
               .startTime_ = t0,
               .endTime_ = t0,
               .scheduledStartTime_ = t0,
               .scheduledEndTime_ = t0,
               .scheduled_ = true});

  auto parsed = motis::ItineraryId{};
  ASSERT_TRUE(parsed.ParseFromString(
      net::decode_base64(generate_itinerary_id(itinerary))));
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
  auto const id = generate_itinerary_id(original);

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

  auto const id = generate_itinerary_id(original);
  auto const stop_times = utl::init_from<ep::stop_times>(data).value();
  auto const routing = utl::init_from<ep::routing>(data).value();
  EXPECT_EQ(original, reconstruct_itinerary(routing, stop_times, {}, id));
}

TEST(motis, itinerary_id_generate_rejects_invalid_single_leg_inputs) {
  auto const cfg = make_config(
      std::string{fmt::format(kSimpleGtfsTemplate, "DA", "FFM", "DA", "FFM")});
  auto data = import_test_data(cfg, "invalid_generate_inputs");
  auto const original =
      route_first_itinerary(data, "test_DA", "test_FFM", "2019-05-01T02:00Z");
  ASSERT_EQ(1U, original.legs_.size());

  auto invalid = original;
  invalid.legs_.clear();
  EXPECT_ANY_THROW(generate_itinerary_id(invalid));

  invalid = original;
  invalid.legs_.front().from_.stopId_ = std::nullopt;
  EXPECT_ANY_THROW(generate_itinerary_id(invalid));

  invalid = original;
  invalid.legs_.front().to_.stopId_ = std::nullopt;
  EXPECT_ANY_THROW(generate_itinerary_id(invalid));
}

TEST(motis,
     itinerary_id_reconstruct_dense_nearby_with_changed_ids_and_shifted_times) {
  auto const source_cfg = make_config(kDenseShiftSourceGtfs);
  auto source_data = import_test_data(source_cfg, "dense_shift_source");
  auto const original = route_first_itinerary(
      source_data, "test_OLD_A", "test_OLD_B", "2019-05-01T02:00Z");
  ASSERT_EQ(1U, original.legs_.size());
  auto const id = generate_itinerary_id(original);

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
  auto const id = generate_itinerary_id(original);

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
  auto const id = generate_itinerary_id(original);

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
  auto body = make_refresh_itinerary_post_body(original);
  body.detailedLegs_ = false;
  EXPECT_EQ(get_res, refresh_post(body));
}

TEST(motis, refresh_itinerary_matches_scheduled_then_applies_realtime) {
  auto const cfg = make_config(
      std::string{fmt::format(kSimpleGtfsTemplate, "DA", "FFM", "DA", "FFM")});
  auto data = import_test_data(cfg, "refresh_itinerary_endpoint_realtime");

  auto const original =
      route_first_itinerary(data, "test_DA", "test_FFM", "2019-05-01T02:00Z");
  auto const id = generate_itinerary_id(original);

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
  auto const itinerary_id = generate_itinerary_id(added_itinerary);

  auto const refresh = utl::init_from<ep::refresh_itinerary>(data).value();
  auto refresh_query = api::refreshItinerary_params{};
  refresh_query.itineraryId_ = itinerary_id;
  auto const refreshed = refresh(refresh_query.to_url("?"));

  ASSERT_EQ(1U, refreshed.legs_.size());
  EXPECT_EQ(itinerary_id, refreshed.id_);
  ASSERT_TRUE(refreshed.legs_.front().tripId_.has_value());
  EXPECT_EQ(added_trip_id, *refreshed.legs_.front().tripId_);
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

  auto const id = generate_itinerary_id(original);
  auto const stop_times = utl::init_from<ep::stop_times>(d).value();
  auto const reconstructed =
      reconstruct_itinerary(routing, stop_times, *d.rt_, id, true, true, true);

  EXPECT_EQ(original, reconstructed);
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

  auto original = res.itineraries_.front();
  auto pt_indices = std::vector<std::size_t>{};
  for (auto i = std::size_t{0}; i < original.legs_.size(); ++i) {
    if (original.legs_[i].mode_ != api::ModeEnum::WALK) {
      pt_indices.push_back(i);
    }
  }
  ASSERT_GE(pt_indices.size(), 3U);

  // Make the middle PT leg impossible to reconstruct by moving its endpoints
  // into the ocean: no stop-time candidates exist within the search radius, so
  // get_run() fails and the leg must be replaced by a cancelled dummy leg.
  auto const broken_idx = pt_indices.at(1);
  auto& broken_leg = original.legs_.at(broken_idx);
  broken_leg.from_.lat_ = 0.0;
  broken_leg.from_.lon_ = 0.0;
  broken_leg.to_.lat_ = 0.0;
  broken_leg.to_.lon_ = 0.0;

  auto const id = generate_itinerary_id(original);
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

// ===========================================================================
// Leg-alternative constraint propagation through the refresh endpoint.
//
// The `plan` endpoint restricts the leg alternatives it computes by the same
// constraints as the main search (transit modes, bike/car carriage, wheelchair
// accessibility - see `sections_violate_constraints` in nigiri's `direct.cc`).
// When the same itinerary is refreshed, the refresh endpoint must reproduce
// exactly those alternatives, so the constraints have to be forwarded to the
// leg-alternatives query as well.
//
// Each fixture below offers, next to the main trip, two later alternatives:
// one that satisfies the constraint and one that violates it. The violating
// alternative departs earlier than the satisfying one, so it would be picked
// first if the constraint were not applied.
// ===========================================================================

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
  query.itineraryId_ = generate_itinerary_id(plan_itin);
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
  query.itineraryId_ = generate_itinerary_id(plan_itin);
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
  query.itineraryId_ = generate_itinerary_id(plan_itin);
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
  query.itineraryId_ = generate_itinerary_id(plan_itin);
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
  query.itineraryId_ = generate_itinerary_id(plan_itin);
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

// Intermediate leg: the alternatives of the middle transit leg of an
// A -> B -> C -> D journey are computed with exact endpoint matching (the
// surrounding transit legs pin both endpoints). Verifies the refresh endpoint
// reproduces the plan response for that path, too.
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
  query.itineraryId_ = generate_itinerary_id(plan_itin);
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
