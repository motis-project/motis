#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <string>

#include "boost/asio/io_context.hpp"

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "net/bad_request_exception.h"
#include "openapi/bad_request_exception.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/map/vehicles.h"
#include "motis/import.h"
#include "motis/rt/vehicle_position.h"
#include "motis/rt_update.h"

using namespace motis::vehicle_positions;
using namespace testing;

namespace {

namespace fs = std::filesystem;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
Test,Test,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
stop-1,Stop 1,50.061,19.938
stop-2,Stop 2,50.071,19.948

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
route-1,Test,1,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign
route-1,S1,trip-1,Destination

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
trip-1,01:00:00,01:00:00,stop-1,1
trip-1,01:05:00,01:05:00,stop-2,2

# calendar_dates.txt
service_id,date,exception_type
S1,20260521,1
)";

struct cwd_guard {
  explicit cwd_guard(fs::path cwd) : old_cwd_{fs::current_path()} {
    fs::current_path(std::move(cwd));
  }

  ~cwd_guard() { fs::current_path(old_cwd_); }

  fs::path old_cwd_;
};

transit_realtime::FeedMessage feed_with_vehicle(
    double const lat = 50.061,
    double const lon = 19.938,
    std::string const& route_id = "route-1") {
  auto msg = transit_realtime::FeedMessage{};
  msg.mutable_header()->set_gtfs_realtime_version("2.0");
  msg.mutable_header()->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);

  auto* entity = msg.add_entity();
  entity->set_id("entity-1");
  auto* vehicle = entity->mutable_vehicle();
  vehicle->mutable_vehicle()->set_id("veh-1");
  vehicle->mutable_vehicle()->set_label("Vehicle 1");
  vehicle->mutable_vehicle()->set_license_plate("KR 12345");
  vehicle->mutable_vehicle()->set_wheelchair_accessible(
      transit_realtime::
          VehicleDescriptor_WheelchairAccessible_WHEELCHAIR_ACCESSIBLE);
  vehicle->mutable_trip()->set_trip_id("trip-1");
  vehicle->mutable_trip()->set_start_date("20260521");
  vehicle->mutable_trip()->set_start_time("12:34:00");
  vehicle->mutable_trip()->set_route_id(route_id);
  vehicle->mutable_trip()->set_direction_id(1);
  vehicle->mutable_trip()->set_schedule_relationship(
      transit_realtime::TripDescriptor_ScheduleRelationship_SCHEDULED);
  vehicle->mutable_position()->set_latitude(lat);
  vehicle->mutable_position()->set_longitude(lon);
  vehicle->mutable_position()->set_bearing(90.0F);
  vehicle->mutable_position()->set_speed(7.5F);
  vehicle->set_current_stop_sequence(23);
  vehicle->set_stop_id("stop-1");
  vehicle->set_current_status(
      transit_realtime::VehiclePosition_VehicleStopStatus_STOPPED_AT);
  vehicle->set_occupancy_status(
      transit_realtime::VehiclePosition_OccupancyStatus_MANY_SEATS_AVAILABLE);
  vehicle->set_timestamp(1789992000);
  return msg;
}

vehicle_position position(std::string feed_id,
                          std::string entity_id,
                          double const lat,
                          double const lon) {
  return vehicle_position{
      .feed_id_ = std::move(feed_id),
      .entity_id_ = std::move(entity_id),
      .reported_position_ = {.pos_ = geo::latlng{lat, lon}},
      .current_stop_sequence_ = 7U,
      .stop_id_ = "stop",
      .current_status_ = "IN_TRANSIT_TO",
      .occupancy_status_ = "NO_DATA_AVAILABLE",
      .ingested_time_ = 1};
}

}  // namespace

TEST(motis_vehicle_positions, parses_gtfsrt_vehicle_position_fields) {
  auto const vehicles =
      parse_gtfsrt_vehicle_positions("krakow", feed_with_vehicle(), 1789992010);

  ASSERT_THAT(vehicles, SizeIs(1));
  auto const& vehicle = vehicles.front();
  EXPECT_EQ(vehicle.feed_id_, "krakow");
  EXPECT_EQ(vehicle.entity_id_, "entity-1");
  EXPECT_EQ(vehicle.vehicle_.id_, "veh-1");
  EXPECT_EQ(vehicle.vehicle_.label_, "Vehicle 1");
  EXPECT_EQ(vehicle.vehicle_.license_plate_, "KR 12345");
  EXPECT_EQ(vehicle.vehicle_.wheelchair_accessible_, "WHEELCHAIR_ACCESSIBLE");
  EXPECT_EQ(vehicle.trip_.trip_id_, "trip-1");
  EXPECT_EQ(vehicle.trip_.start_date_, "20260521");
  EXPECT_EQ(vehicle.trip_.start_time_, "12:34:00");
  EXPECT_EQ(vehicle.trip_.route_id_, "route-1");
  EXPECT_EQ(vehicle.trip_.direction_id_, 1U);
  EXPECT_EQ(vehicle.trip_.schedule_relationship_, "SCHEDULED");
  EXPECT_NEAR(vehicle.reported_position_.pos_.lat_, 50.061, 0.000001);
  EXPECT_NEAR(vehicle.reported_position_.pos_.lng_, 19.938, 0.000001);
  ASSERT_TRUE(vehicle.reported_position_.bearing_.has_value());
  EXPECT_DOUBLE_EQ(*vehicle.reported_position_.bearing_, 90.0);
  ASSERT_TRUE(vehicle.reported_position_.speed_mps_.has_value());
  EXPECT_DOUBLE_EQ(*vehicle.reported_position_.speed_mps_, 7.5);
  EXPECT_EQ(vehicle.current_stop_sequence_, 23U);
  EXPECT_EQ(vehicle.stop_id_, "stop-1");
  EXPECT_EQ(vehicle.current_status_, "STOPPED_AT");
  EXPECT_EQ(vehicle.occupancy_status_, "MANY_SEATS_AVAILABLE");
  EXPECT_EQ(vehicle.reported_time_, 1789992000);
  EXPECT_EQ(vehicle.ingested_time_, 1789992010);
}

TEST(motis_vehicle_positions, drops_zero_zero_vehicle_position) {
  auto const vehicles =
      parse_gtfsrt_vehicle_positions("krakow", feed_with_vehicle(0.0, 0.0), 1);

  EXPECT_TRUE(vehicles.empty());
}

TEST(motis_vehicle_positions, replaces_only_the_selected_feed) {
  auto store = vehicle_position_store{};
  store.replace_feed("feed-a",
                     {position("feed-a", "old", 50.0, 19.9),
                      position("feed-a", "stale", 50.1, 19.9)});
  store.replace_feed("feed-b", {position("feed-b", "kept", 50.2, 19.9)});

  store.replace_feed("feed-a", {position("feed-a", "new", 50.3, 19.9)});

  auto const snapshot = store.snapshot(
      vehicle_viewport{.min_ = geo::latlng{49.9, 19.8},
                       .max_ = geo::latlng{50.4, 20.0}});
  EXPECT_THAT(snapshot,
              ElementsAre(Field(&vehicle_position::entity_id_, Eq("new")),
                          Field(&vehicle_position::entity_id_, Eq("kept"))));
}

TEST(motis_vehicle_positions, snapshots_only_viewport_matches) {
  auto store = vehicle_position_store{};
  store.replace_feed("feed",
                     {position("feed", "inside", 50.061, 19.938),
                      position("feed", "outside", 51.0, 19.938)});

  auto const snapshot = store.snapshot(
      vehicle_viewport{.min_ = geo::latlng{50.0, 19.8},
                       .max_ = geo::latlng{50.2, 20.1}});

  ASSERT_THAT(snapshot, SizeIs(1));
  EXPECT_EQ(snapshot.front().entity_id_, "inside");
}

TEST(motis_vehicle_positions, endpoint_requires_viewport) {
  auto rt = std::make_shared<motis::rt>();
  auto endpoint = motis::ep::vehicles{.rt_ = rt};

  EXPECT_THROW(endpoint("/api/v1/map/vehicles"), openapi::bad_request_exception);
}

TEST(motis_vehicle_positions, endpoint_returns_viewport_payload) {
  auto rt = std::make_shared<motis::rt>();
  rt->vehicle_positions_->replace_feed(
      "feed", {position("feed", "inside", 50.061, 19.938),
               position("feed", "outside", 51.0, 19.938)});
  auto endpoint = motis::ep::vehicles{.rt_ = rt};

  auto const res =
      endpoint("/api/v1/map/vehicles?min=50.0,19.8&max=50.2,20.1");

  ASSERT_THAT(res.vehicles_, SizeIs(1));
  EXPECT_EQ(res.vehicles_.front().feedId_, "feed");
  EXPECT_EQ(res.vehicles_.front().entityId_, "inside");
  EXPECT_DOUBLE_EQ(res.vehicles_.front().reportedPosition_.lat_, 50.061);
  EXPECT_DOUBLE_EQ(res.vehicles_.front().reportedPosition_.lon_, 19.938);
  EXPECT_EQ(res.vehicles_.front().currentStopSequence_, 7);
  EXPECT_EQ(res.vehicles_.front().stopId_, "stop");
  EXPECT_EQ(res.vehicles_.front().currentStatus_, "IN_TRANSIT_TO");
  EXPECT_EQ(res.vehicles_.front().occupancyStatus_, "NO_DATA_AVAILABLE");
}

TEST(motis_vehicle_positions, rt_update_consumes_vehicle_only_gtfsrt_feed) {
  auto const test_dir = fs::absolute("test/data/vehicle-only-rt-update");
  auto ec = std::error_code{};
  fs::remove_all(test_dir, ec);
  fs::create_directories(test_dir);
  auto cwd = cwd_guard{test_dir};

  auto const c = motis::config{
      .timetable_ = {motis::config::timetable{
          .first_day_ = "2026-05-21",
          .num_days_ = 2,
          .update_interval_ = 3600,
          .canned_rt_ = true,
          .datasets_ = {{"test",
                         {.path_ = kGTFS,
                          .rt_ = {{{.url_ =
                                         "https://example.test/"
                                         "vehicle_positions"}}}}}}}}};

  motis::import(c, "data");
  auto d = motis::data{"data", c};

  fs::create_directory("dump_rt");
  auto const feed = feed_with_vehicle(50.061, 19.938, "prefix-route-1");
  auto dump = std::ofstream{"dump_rt/test-https___example_test_"
                            "vehicle_positions",
                            std::ios::binary};
  ASSERT_TRUE(dump.good());
  auto const bytes = feed.SerializeAsString();
  dump.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  dump.close();

  auto ioc = boost::asio::io_context{};
  motis::run_rt_update(ioc, c, d);
  ioc.run_for(std::chrono::milliseconds{100});

  auto const snapshot =
      d.rt_->vehicle_positions_->snapshot(vehicle_viewport{
          .min_ = geo::latlng{50.0, 19.8},
          .max_ = geo::latlng{50.2, 20.1}});

  ASSERT_THAT(snapshot, SizeIs(1));
  EXPECT_EQ(snapshot.front().entity_id_, "entity-1");
  EXPECT_EQ(snapshot.front().trip_.trip_id_, "trip-1");
  EXPECT_EQ(snapshot.front().stop_id_, "stop-1");

  auto endpoint = motis::ep::vehicles{.tags_ = d.tags_.get(),
                                      .tt_ = d.tt_.get(),
                                      .shapes_ = d.shapes_.get(),
                                      .rt_ = d.rt_};
  auto const res =
      endpoint("/api/v1/map/vehicles?min=50.0,19.8&max=50.2,20.1");

  ASSERT_THAT(res.vehicles_, SizeIs(1));
  ASSERT_TRUE(res.vehicles_.front().route_.has_value());
  EXPECT_EQ(res.vehicles_.front().route_->shortName_, "1");
  EXPECT_EQ(res.vehicles_.front().trip_.routeId_, "prefix-route-1");
  EXPECT_EQ(res.vehicles_.front().route_->id_, "prefix-route-1");
  ASSERT_TRUE(res.vehicles_.front().mode_.has_value());
  EXPECT_EQ(res.vehicles_.front().mode_, motis::api::ModeEnum::BUS);
  ASSERT_TRUE(res.vehicles_.front().shape_.has_value());
  EXPECT_GT(res.vehicles_.front().shape_->length_, 0);
}
