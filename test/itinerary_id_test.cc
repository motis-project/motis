#include "gtest/gtest.h"

#include "utl/init_from.h"

#include "motis/config.h"
#include "motis/endpoints/routing.h"
#include "motis/import.h"
#include "motis/itinerary_id.h"

using namespace motis;

namespace {
int64_t to_epoch_seconds(openapi::date_time_t const& t) {
  return std::chrono::duration_cast<std::chrono::seconds>(
             static_cast<std::chrono::sys_seconds>(t).time_since_epoch())
      .count();
};

constexpr auto const kSimpleGTFSTemplate = R"(
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

constexpr auto const kLoopGTFS = R"(
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

constexpr auto const kDenseShiftSourceGTFS = R"(
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

constexpr auto const kDenseShiftTargetGTFS = R"(
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

data import_test_data(config const& c, std::string_view const sub_dir) {
  auto const path = std::filesystem::path{"test/data/itinerary_id"} / sub_dir;
  auto ec = std::error_code{};
  std::filesystem::remove_all(path, ec);
  return import(c, path.string(), true);
}

api::Itinerary route_first_itinerary(data& d,
                                     std::string_view const from_place,
                                     std::string_view const to_place,
                                     std::string_view const time) {
  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const query = fmt::format(
      "?fromPlace={}&toPlace={}&time={}&timetableView=false"
      "&directModes=WALK,RENTAL",
      from_place, to_place, time);
  return routing(query).itineraries_.at(0);
}

}  // namespace

TEST(motis, itinerary_id_reconstruct_with_changed_stop_ids) {
  auto const source_cfg = make_config(
      std::string{fmt::format(kSimpleGTFSTemplate, "DA", "FFM", "DA", "FFM")});
  auto source_data = import_test_data(source_cfg, "changed_stop_ids_source");
  auto const original = route_first_itinerary(source_data, "test_DA",
                                              "test_FFM", "2019-05-01T02:00Z");
  auto const id = generate_itinerary_id(original);

  auto const target_cfg = make_config(
      std::string{fmt::format(kSimpleGTFSTemplate, "FFM", "DA", "FFM", "DA")});
  auto target_data = import_test_data(target_cfg, "changed_stop_ids_target");
  auto const expected = route_first_itinerary(target_data, "test_FFM",
                                              "test_DA", "2019-05-01T02:00Z");
  auto const stop_times = utl::init_from<ep::stop_times>(target_data).value();

  EXPECT_EQ(expected, reconstruct_itinerary(stop_times, id));
}

TEST(motis, itinerary_id_reconstruct_with_repeated_stop_in_trip) {
  auto const cfg = make_config(kLoopGTFS);
  auto data = import_test_data(cfg, "repeated_stop_route");

  auto const original =
      route_first_itinerary(data, "test_B", "test_D", "2019-05-01T08:00Z");
  ASSERT_EQ(1U, original.legs_.size());

  auto const id = generate_itinerary_id(original);
  auto const stop_times = utl::init_from<ep::stop_times>(data).value();
  EXPECT_EQ(original, reconstruct_itinerary(stop_times, id));
}

TEST(motis, itinerary_id_generate_rejects_invalid_single_leg_inputs) {
  auto const cfg = make_config(
      std::string{fmt::format(kSimpleGTFSTemplate, "DA", "FFM", "DA", "FFM")});
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
  invalid.legs_.front().from_.stopId_ = std::string{};
  EXPECT_ANY_THROW(generate_itinerary_id(invalid));

  invalid = original;
  invalid.legs_.front().to_.stopId_ = std::nullopt;
  EXPECT_ANY_THROW(generate_itinerary_id(invalid));

  invalid = original;
  invalid.legs_.front().to_.stopId_ = std::string{};
  EXPECT_ANY_THROW(generate_itinerary_id(invalid));

  invalid = original;
  invalid.legs_.front().scheduledStartTime_ = {};
  EXPECT_ANY_THROW(generate_itinerary_id(invalid));
}

TEST(motis,
     itinerary_id_reconstruct_dense_nearby_with_changed_ids_and_shifted_times) {
  auto const source_cfg = make_config(kDenseShiftSourceGTFS);
  auto source_data = import_test_data(source_cfg, "dense_shift_source");
  auto const original = route_first_itinerary(
      source_data, "test_OLD_A", "test_OLD_B", "2019-05-01T02:00Z");
  ASSERT_EQ(1U, original.legs_.size());
  auto const id = generate_itinerary_id(original);

  auto const target_cfg = make_config(kDenseShiftTargetGTFS);
  auto target_data = import_test_data(target_cfg, "dense_shift_target");
  auto const stop_times = utl::init_from<ep::stop_times>(target_data).value();

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

  auto const reconstructed = reconstruct_itinerary(stop_times, id);
  ASSERT_EQ(1U, reconstructed.legs_.size());
  auto const& reconstructed_leg = reconstructed.legs_.front();
  auto const& original_leg = original.legs_.front();

  ASSERT_TRUE(reconstructed_leg.from_.stopId_.has_value());
  ASSERT_TRUE(reconstructed_leg.to_.stopId_.has_value());
  EXPECT_EQ("test_NEW_A", *reconstructed_leg.from_.stopId_);
  EXPECT_EQ("test_NEW_B", *reconstructed_leg.to_.stopId_);

  ASSERT_TRUE(reconstructed_leg.tripId_.has_value());
  EXPECT_NE(std::string::npos, reconstructed_leg.tripId_->find("test_MATCH"));

  EXPECT_EQ(to_epoch_seconds(original_leg.scheduledStartTime_) + 8 * 60,
            to_epoch_seconds(reconstructed_leg.scheduledStartTime_));
  EXPECT_EQ(to_epoch_seconds(original_leg.scheduledEndTime_) + 8 * 60,
            to_epoch_seconds(reconstructed_leg.scheduledEndTime_));
}
