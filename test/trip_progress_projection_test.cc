#include "gtest/gtest.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "nigiri/rt/frun.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/import.h"
#include "motis/rt/trip_progress_projection.h"
#include "motis/tag_lookup.h"

namespace fs = std::filesystem;
namespace n = nigiri;

namespace motis {
namespace {

constexpr auto const kStraightGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
Test,Test,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,50.0000,8.0000
B,B,50.0000,8.0100
C,C,50.0000,8.0200

# routes.txt
route_id,agency_id,route_short_name,route_type
route,Test,1,3

# trips.txt
route_id,service_id,trip_id,shape_id
route,S1,straight,straight-shape

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
straight,01:00:00,01:00:00,A,10
straight,01:05:00,01:05:00,B,20
straight,01:10:00,01:10:00,C,30

# calendar_dates.txt
service_id,date,exception_type
S1,20260521,1

# shapes.txt
shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence
straight-shape,50.0000,8.0000,1
straight-shape,50.0000,8.0050,2
straight-shape,50.0000,8.0100,3
straight-shape,50.0000,8.0150,4
straight-shape,50.0000,8.0200,5
)";

constexpr auto const kRepeatedShapeGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
Test,Test,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,50.0000,8.0000
B,B,50.0000,8.0100
C,C,50.0000,7.9900

# routes.txt
route_id,agency_id,route_short_name,route_type
route,Test,1,3

# trips.txt
route_id,service_id,trip_id,shape_id
route,S1,repeated,repeated-shape

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
repeated,01:00:00,01:00:00,A,10
repeated,01:05:00,01:05:00,B,20
repeated,01:10:00,01:10:00,A,30
repeated,01:15:00,01:15:00,C,40

# calendar_dates.txt
service_id,date,exception_type
S1,20260521,1

# shapes.txt
shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence
repeated-shape,50.0000,8.0000,1
repeated-shape,50.0000,8.0050,2
repeated-shape,50.0000,8.0100,3
repeated-shape,50.0000,8.0050,4
repeated-shape,50.0000,8.0000,5
repeated-shape,50.0000,7.9950,6
repeated-shape,50.0000,7.9900,7
)";

constexpr auto const kGeometryGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
Test,Test,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
LA,LA,50.0000,8.0000
LB,LB,50.0100,8.0000
LC,LC,50.0100,8.0100
LD,LD,50.0000,8.0100
XA,XA,50.0000,8.0000
XB,XB,50.0100,8.0100
XC,XC,50.0100,8.0000
XD,XD,50.0000,8.0100

# routes.txt
route_id,agency_id,route_short_name,route_type
route,Test,1,3

# trips.txt
route_id,service_id,trip_id,shape_id
route,S1,loop,loop-shape
route,S1,crossing,crossing-shape
route,S1,missing,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
loop,01:00:00,01:00:00,LA,10
loop,01:05:00,01:05:00,LB,20
loop,01:10:00,01:10:00,LC,30
loop,01:15:00,01:15:00,LD,40
loop,01:20:00,01:20:00,LA,50
crossing,01:00:00,01:00:00,XA,10
crossing,01:05:00,01:05:00,XB,20
crossing,01:10:00,01:10:00,XC,30
crossing,01:15:00,01:15:00,XD,40
missing,01:00:00,01:00:00,XA,10
missing,01:05:00,01:05:00,XB,20

# calendar_dates.txt
service_id,date,exception_type
S1,20260521,1

# shapes.txt
shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence
loop-shape,50.0000,8.0000,1
loop-shape,50.0100,8.0000,2
loop-shape,50.0100,8.0100,3
loop-shape,50.0000,8.0100,4
loop-shape,50.0000,8.0000,5
crossing-shape,50.0000,8.0000,1
crossing-shape,50.0100,8.0100,2
crossing-shape,50.0100,8.0000,3
crossing-shape,50.0000,8.0100,4
)";

struct projection_fixture {
  explicit projection_fixture(std::string_view const gtfs)
      : path_{fs::temp_directory_path() / "motis-trip-progress-test"},
        config_{
            .timetable_ = config::timetable{
                .first_day_ = "2026-05-21",
                .num_days_ = 2,
                .with_shapes_ = true,
                .datasets_ = {{"progress", {.path_ = std::string{gtfs}}}}}} {
    auto ec = std::error_code{};
    fs::remove_all(path_, ec);
    import(config_, path_);
    data_.emplace(path_, config_);
  }

  ~projection_fixture() {
    data_.reset();
    auto ec = std::error_code{};
    fs::remove_all(path_, ec);
  }

  n::rt::frun run(std::string trip_id) const {
    auto const [run, trip] = data_->tags_->get_trip(
        *data_->tt_, nullptr, "20260521_01:00_progress_" + std::move(trip_id));
    EXPECT_TRUE(run.valid());
    EXPECT_NE(trip, n::trip_idx_t::invalid());
    return n::rt::frun{*data_->tt_, nullptr, run};
  }

  fs::path path_;
  config config_;
  std::optional<data> data_;
};

TEST(trip_progress_projection, projects_an_ordinary_route) {
  auto fixture = projection_fixture{kStraightGtfs};
  auto projector = trip_progress_projector{*fixture.data_->shapes_};

  auto const result =
      projector.project(fixture.run("straight"), geo::latlng{50.0, 8.005});

  ASSERT_EQ(result.status_, trip_progress_projection_status::kProjected);
  ASSERT_TRUE(result.progress_.has_value());
  EXPECT_GT(result.progress_->distance_along_shape_m_, 300.0);
  EXPECT_LT(result.progress_->distance_along_shape_m_, 400.0);
  EXPECT_LT(result.progress_->lateral_error_m_, 0.1);
  EXPECT_EQ(result.progress_->next_static_stop_sequence_, 20U);
  EXPECT_GT(result.progress_->distance_to_next_stop_m_, 300.0);
  EXPECT_LT(result.progress_->distance_to_next_stop_m_, 400.0);
  EXPECT_EQ(result.progress_->monotonicity_,
            trip_progress_monotonicity::kNoPrior);
}

TEST(trip_progress_projection, converts_a_non_zero_run_range) {
  auto fixture = projection_fixture{kStraightGtfs};
  auto projector = trip_progress_projector{*fixture.data_->shapes_};
  auto run = fixture.run("straight");
  ++run.stop_range_.from_;

  auto const result = projector.project(run, geo::latlng{50.0, 8.015});

  ASSERT_EQ(result.status_, trip_progress_projection_status::kProjected);
  ASSERT_TRUE(result.progress_.has_value());
  EXPECT_GT(result.progress_->distance_along_shape_m_, 300.0);
  EXPECT_LT(result.progress_->distance_along_shape_m_, 400.0);
  EXPECT_EQ(result.progress_->next_static_stop_sequence_, 30U);
}

TEST(trip_progress_projection, rejects_a_far_off_shape_position) {
  auto fixture = projection_fixture{kStraightGtfs};
  auto projector = trip_progress_projector{*fixture.data_->shapes_};

  auto const result =
      projector.project(fixture.run("straight"), geo::latlng{51.0, 8.005});

  EXPECT_EQ(result.status_, trip_progress_projection_status::kOffShape);
  EXPECT_FALSE(result.progress_.has_value());
}

TEST(trip_progress_projection, rejects_a_backward_jump_from_prior_progress) {
  auto fixture = projection_fixture{kStraightGtfs};
  auto projector = trip_progress_projector{*fixture.data_->shapes_};
  auto const prior = trip_progress{.distance_along_shape_m_ = 1000.0};

  auto const result = projector.project(fixture.run("straight"),
                                        geo::latlng{50.0, 8.005}, prior);

  EXPECT_EQ(result.status_, trip_progress_projection_status::kImplausible);
  EXPECT_FALSE(result.progress_.has_value());
}

TEST(trip_progress_projection,
     rejects_a_repeated_shape_without_disambiguating_prior_progress) {
  auto fixture = projection_fixture{kRepeatedShapeGtfs};
  auto projector = trip_progress_projector{*fixture.data_->shapes_};

  auto const result =
      projector.project(fixture.run("repeated"), geo::latlng{50.0, 8.005});

  EXPECT_EQ(result.status_, trip_progress_projection_status::kAmbiguous);
  EXPECT_FALSE(result.progress_.has_value());
}

TEST(trip_progress_projection, prior_progress_disambiguates_a_repeated_shape) {
  auto fixture = projection_fixture{kRepeatedShapeGtfs};
  auto projector = trip_progress_projector{*fixture.data_->shapes_};
  auto const prior = trip_progress{.distance_along_shape_m_ = 900.0};

  auto const result = projector.project(fixture.run("repeated"),
                                        geo::latlng{50.0, 8.005}, prior);

  ASSERT_EQ(result.status_, trip_progress_projection_status::kProjected);
  ASSERT_TRUE(result.progress_.has_value());
  EXPECT_GT(result.progress_->distance_along_shape_m_, 1000.0);
  EXPECT_EQ(result.progress_->next_static_stop_sequence_, 30U);
  EXPECT_EQ(result.progress_->monotonicity_,
            trip_progress_monotonicity::kForward);
}

TEST(trip_progress_projection, rejects_an_ambiguous_loop) {
  auto fixture = projection_fixture{kGeometryGtfs};
  auto projector = trip_progress_projector{*fixture.data_->shapes_};

  auto const result =
      projector.project(fixture.run("loop"), geo::latlng{50.0, 8.0});

  EXPECT_EQ(result.status_, trip_progress_projection_status::kAmbiguous);
  EXPECT_FALSE(result.progress_.has_value());
}

TEST(trip_progress_projection,
     rejects_far_separated_forward_candidates_at_a_crossing) {
  auto fixture = projection_fixture{kGeometryGtfs};
  auto projector = trip_progress_projector{*fixture.data_->shapes_};
  auto const prior = trip_progress{.distance_along_shape_m_ = 100.0};

  auto const result = projector.project(fixture.run("crossing"),
                                        geo::latlng{50.005, 8.005}, prior);

  EXPECT_EQ(result.status_, trip_progress_projection_status::kAmbiguous);
  EXPECT_FALSE(result.progress_.has_value());
}

TEST(trip_progress_projection,
     maps_gapped_vehicle_stop_sequence_to_the_constrained_section) {
  auto fixture = projection_fixture{kGeometryGtfs};
  auto projector = trip_progress_projector{*fixture.data_->shapes_};
  auto const constraint = vehicle_position_progress_constraint{
      .current_static_stop_sequence_ = 40U,
      .status_ = vehicle_position_stop_status::kInTransitTo};

  auto const result =
      projector.project(fixture.run("crossing"), geo::latlng{50.005, 8.005},
                        std::nullopt, constraint);

  ASSERT_EQ(result.status_, trip_progress_projection_status::kProjected);
  ASSERT_TRUE(result.progress_.has_value());
  EXPECT_EQ(result.progress_->next_static_stop_sequence_, 40U);
  EXPECT_GT(result.progress_->distance_along_shape_m_, 2000.0);
}

TEST(trip_progress_projection, stopped_at_constraint_projects_to_that_stop) {
  auto fixture = projection_fixture{kGeometryGtfs};
  auto projector = trip_progress_projector{*fixture.data_->shapes_};
  auto const constraint = vehicle_position_progress_constraint{
      .current_static_stop_sequence_ = 30U,
      .status_ = vehicle_position_stop_status::kStoppedAt};

  auto const result =
      projector.project(fixture.run("crossing"), geo::latlng{50.01, 8.0},
                        std::nullopt, constraint);

  ASSERT_EQ(result.status_, trip_progress_projection_status::kProjected);
  ASSERT_TRUE(result.progress_.has_value());
  EXPECT_EQ(result.progress_->next_static_stop_sequence_, 30U);
  EXPECT_LT(result.progress_->lateral_error_m_, 0.1);
  EXPECT_DOUBLE_EQ(result.progress_->distance_to_next_stop_m_, 0.0);
}

TEST(trip_progress_projection, rejects_an_unknown_vehicle_stop_sequence) {
  auto fixture = projection_fixture{kGeometryGtfs};
  auto projector = trip_progress_projector{*fixture.data_->shapes_};
  auto const constraint = vehicle_position_progress_constraint{
      .current_static_stop_sequence_ = 31U,
      .status_ = vehicle_position_stop_status::kIncomingAt};

  auto const result =
      projector.project(fixture.run("crossing"), geo::latlng{50.005, 8.005},
                        std::nullopt, constraint);

  EXPECT_EQ(result.status_, trip_progress_projection_status::kImplausible);
  EXPECT_FALSE(result.progress_.has_value());
}

TEST(trip_progress_projection, fails_closed_when_the_shape_is_missing) {
  auto fixture = projection_fixture{kGeometryGtfs};
  auto projector = trip_progress_projector{*fixture.data_->shapes_};

  auto const result =
      projector.project(fixture.run("missing"), geo::latlng{50.0, 8.0});

  EXPECT_EQ(result.status_, trip_progress_projection_status::kMissingShape);
  EXPECT_FALSE(result.progress_.has_value());
}

}  // namespace
}  // namespace motis
