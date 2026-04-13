#include "gtest/gtest.h"

#ifdef NO_DATA
#undef NO_DATA
#endif

#include <chrono>
#include <optional>
#include <vector>

#include "utl/init_from.h"

#include "nigiri/common/parse_time.h"

#include "motis-api/motis-api.h"

#include "motis/config.h"
#include "motis/endpoints/one_to_many.h"
#include "motis/endpoints/one_to_many_post.h"

#include "../test_case.h"

using namespace std::string_view_literals;
using namespace motis;
using namespace date;
using namespace std::chrono_literals;

namespace n = nigiri;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
DA,DA Hbf,49.87260,8.63085,1,,
DA_3,DA Hbf,49.87355,8.63003,0,DA,3
DA_10,DA Hbf,49.87336,8.62926,0,DA,10
FFM,FFM Hbf,50.10701,8.66341,1,,
FFM_101,FFM Hbf,50.10739,8.66333,0,FFM,101
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10
FFM_12,FFM Hbf,50.10658,8.66178,0,FFM,12
de:6412:10:6:1,FFM Hbf U-Bahn,50.107577,8.6638173,0,,U4
LANGEN,Langen,49.99359,8.65677,1,,1
FFM_HAUPT,FFM Hauptwache,50.11403,8.67835,1,,
FFM_HAUPT_U,Hauptwache U1/U2/U3/U8,50.11385,8.67912,0,FFM_HAUPT,
FFM_HAUPT_S,FFM Hauptwache S,50.11404,8.67824,0,FFM_HAUPT,
PAUL1,Römer/Paulskirche,50.110979,8.682276,0,,
PAUL2,Römer/Paulskirche,50.110828,8.681587,0,,
FFM_C,FFM C,50.107812,8.664628,0,,
FFM_B,FFM B,50.107519,8.664775,0,,
DA_Bus_1,DA Hbf,49.8722160,8.6282315
DA_Bus_2,DA Hbf,49.8755778,8.6240518
DA_Tram_1,DA Hbf,49.8752926,8.6277460
DA_Tram_2,DA Hbf,49.874995,8.6313925
DA_Tram_3,DA Hbf,49.871561,8.6320181

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
S3,DB,S3,,,109
U4,DB,U4,,,402
ICE,DB,ICE,,,101
11_1,DB,11,,,0
11_2,DB,11,,,0
B1,DB,B1,,3
T1,DB,T1,,0

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
S3,S1,S3,,block_1
U4,S1,U4,,block_1
ICE,S1,ICE,,
11_1,S1,11_1_1,,
11_1,S1,11_1_2,,
11_2,S1,11_2_1,,
11_2,S1,11_2_2,,
B1,S1,B1,Bus 1,
T1,S1,T1,Tram 1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
S3,01:15:00,01:15:00,FFM_101,1,0,0
S3,01:20:00,01:20:00,FFM_HAUPT_S,2,0,0
U4,01:05:00,01:05:00,de:6412:10:6:1,0,0,0
U4,01:10:00,01:10:00,FFM_HAUPT_U,1,0,0
ICE,00:35:00,00:35:00,DA_10,0,0,0
ICE,00:45:00,00:45:00,FFM_10,1,0,0
11_1_1,12:00:00,12:00:00,PAUL1,0,0,0
11_1_1,12:10:00,12:10:00,FFM_C,1,0,0
11_1_2,12:15:00,12:15:00,PAUL1,0,0,0
11_1_2,12:25:00,12:25:00,FFM_C,1,0,0
11_2_1,12:05:00,12:05:00,FFM_B,0,0,0
11_2_1,12:15:00,12:15:00,PAUL2,1,0,0
11_2_2,12:20:00,12:20:00,FFM_B,0,0,0
11_2_2,12:30:00,12:30:00,PAUL2,1,0,0
B1,00:10:00,00:10:00,DA_Bus_1,1
B1,00:20:00,00:20:00,DA_Bus_2,2
T1,00:24:00,00:24:00,DA_Tram_1,1
T1,00:25:00,00:25:00,DA_Tram_2,2
T1,00:26:00,00:26:00,DA_Tram_3,3

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

template <>
test_case_params const import_test_case<test_case::FFM_one_to_many>() {
  auto const c =
      config{.osm_ = {"test/resources/test_case.osm.pbf"},
             .timetable_ =
                 config::timetable{.first_day_ = "2019-05-01",
                                   .num_days_ = 2,
                                   .datasets_ = {{"test", {.path_ = kGTFS}}}},
             .street_routing_ = true,
             .osr_footpath_ = true};
  return import_test_case(std::move(c), "test/test_case/ffm_one_to_many_data");
}

std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>
parse_time(std::string_view time) {
  return std::chrono::time_point_cast<std::chrono::seconds>(
      n::parse_time(time, "%FT%T%Ez"));
}

auto one_to_many_get(data& d) {
  return utl::init_from<ep::one_to_many_intermodal>(d).value();
}

auto one_to_many_post(data& d) {
  return utl::init_from<ep::one_to_many_intermodal_post>(d).value();
}

TEST(one_to_many, get_request_forward) {
  auto [d, _] = get_test_case<test_case::FFM_one_to_many>();

  auto const durations = one_to_many_get(d)(
      "/api/experimental/one-to-many-intermodal"
      "?one=49.8722439;8.6320624"  // Near DA
      "&many="
      "49.87336;8.62926,"  // DA_10
      "50.10593;8.66118,"  // FFM_10
      "50.107577;8.6638173,"  // de:6412:10:6:1
      "50.10739;8.66333,"  // FFM_101
      "50.11385;8.67912,"  // FFM_HAUPT_U
      "50.11404;8.67824,"  // FFM_HAUPT_S
      "49.872855;8.632008,"  // Near one
      "49.872504;8.628988,"  // Inside DA station
      "49.874399;8.630361,"  // Still reachable, near DA
      "49.875292;8.627746,"  // Far, near DA
      "50.106596;8.663485,"  // Inside FFM station
      "50.106209;8.668934,"  // Outside FFM station
      "50.103663;8.663666,"  // Far, not reachable, near FFM
      "50.113494;8.679129,"  // Near FFM_HAUPT_U
      "50.114080;8.677027,"  // Near FFM_HAUPT_S
      "50.114520;8.673050,"  // Too far from FFM_HAUPT_U
      "50.114773;8.672604"  // Far, near FFM_HAUPT
      "&time=2019-04-30T22:30:00.000Z"
      "&maxTravelTime=60"
      "&maxMatchingDistance=250"
      "&maxDirectTime=540"  // Updated to maxPreTransitTime == 1500
      "&maxPostTransitTime=420"
      "&arriveBy=false");

  EXPECT_EQ((api::OneToManyIntermodalResponse{
                .street_durations_ = {{{.duration_ = 281.0},
                                       {},
                                       {},
                                       {},
                                       {},
                                       {},
                                       {.duration_ = 122.0},
                                       {.duration_ = 240.0},
                                       {.duration_ = 529.0},
                                       {.duration_ = 582.0},
                                       {},
                                       {},
                                       {},
                                       {},
                                       {},
                                       {},
                                       {}}},
                .transit_durations_ = {{
                    {},
                    {{.duration_ = 1080.0, .transfers_ = 0}},
                    {// Not routed transfer => faster than realistic
                     {.duration_ = 1140.0, .transfers_ = 0}},
                    {// Not routed transfer
                     {.duration_ = 1140.0, .transfers_ = 0}},
                    {{.duration_ = 2580.0, .transfers_ = 1}},
                    {{.duration_ = 2580.0, .transfers_ = 1}},
                    {},
                    {},
                    {},
                    {},
                    {{.duration_ = 1260.0, .transfers_ = 0}},
                    {{.duration_ = 1620.0, .transfers_ = 0}},
                    {},
                    {{.duration_ = 2700.0, .transfers_ = 1}},
                    {{.duration_ = 2640.0, .transfers_ = 1}},
                    {{.duration_ = 2940.0, .transfers_ = 1}},
                    {},
                }}}),
            durations);
}

TEST(one_to_many, post_request_backward) {
  auto [d, _] = get_test_case<test_case::FFM_one_to_many>();

  auto const durations = one_to_many_post(d)(api::OneToManyIntermodalParams{
      .one_ = "50.113816,8.679421,0",  // Near FFM_HAUPT
      .many_ =
          {
              "49.87336,8.62926",  // DA_10
              "50.10593,8.66118",  // FFM_10
              "test_FFM_10",
              "50.107577,8.6638173",  // de:6412:10:6:1
              "50.10739,8.66333",  // FFM_101
              "test_FFM_101",
              "50.11385,8.67912",  // FFM_HAUPT_U
              "50.11385,8.67912,-4",  // FFM_HAUPT_U
              "test_FFM_HAUPT_U",
              "50.11404,8.67824",  // FFM_HAUPT_S
              "50.113385,8.678328,0",  // Close, near FFM_HAUPT, level 0
              "50.113385,8.678328,-2",  // Close, near FFM_HAUPT, level -2
              "50.111900,8.675208",  // Far, near FFM_HAUPT
              "50.106543,8.663474,0",  // Close, near FFM
              "50.107361,8.660478",  // Too far from de:6412:10:6:1
              "50.104298,8.660285",  // Far, near FFM
              "49.872243,8.632062",  // Near DA
              "49.875368,8.627596",  // Far, near DA
          },
      .time_ = parse_time("2019-05-01T01:25:00.000+02:00"),
      .maxTravelTime_ = 60,
      .maxMatchingDistance_ = 250.0,
      .arriveBy_ = true,
      .maxPreTransitTime_ = 300,
      .maxDirectTime_ = 300});  // Updated to maxPostTransitTime == 1500

  EXPECT_EQ(
      (api::OneToManyIntermodalResponse{
          .street_durations_ = {{
              {},
              {},
              {},
              {},
              {},
              {},
              {.duration_ = 159.0},  // No explicit level
              {.duration_ = 160.0},  // Explicit level
              {.duration_ = 160.0},
              {.duration_ = 127.0},
              {.duration_ = 103.0},
              {.duration_ = 123.0},
              {.duration_ = 355.0},
              {},
              {},
              {},
              {},
              {},
          }},
          .transit_durations_ = {{
              {{.duration_ = 3180.0, .transfers_ = 1}},
              {{.duration_ = 840.0, .transfers_ = 0}},  // Not routed transfer
              {{.duration_ = 720.0, .transfers_ = 0}},  // Not routed transfer
              {{.duration_ = 780.0, .transfers_ = 0}},
              {{.duration_ = 780.0, .transfers_ = 0}},
              {{.duration_ = 720.0, .transfers_ = 0}},
              {},
              {},
              {},
              {},
              {},
              {},
              {},
              {{.duration_ = 900.0, .transfers_ = 0}},
              {{.duration_ = 1020.0, .transfers_ = 0}},
              {},
              {{.duration_ = 3360.0, .transfers_ = 1}},  // from DA_10: 3420.0
              {},
          }}}),
      durations);
}

TEST(one_to_many,
     post_request_forward_with_routed_transfers_and_short_pre_transit) {
  auto [d, _] = get_test_case<test_case::FFM_one_to_many>();

  auto const durations = one_to_many_post(d)(api::OneToManyIntermodalParams{
      .one_ = "50.106691,8.659763",  // Near FFM
      .many_ =
          {
              "test_DA_10",
              "50.107577,8.6638173",  // de:6412:10:6:1
              "test_de:6412:10:6:1",
              "test_FFM_101",
              "test_FFM_HAUPT_S",
              "50.11385,8.67912",  // FFM_HAUPT_U
              "50.10590,8.66452",  // Near FFM
              "50.113291,8.678321,0",  // Near FFM_HAUPT
              "50.113127,8.678879,-2",  // Near FFM_HAUPT
              "50.114141,8.677025,-3",  // Near FFM_HAUPT
              "50.113589,8.679070,-4",  // Near FFM_HAUPT
          },
      .time_ = parse_time("2019-05-01T00:55:00.000+02:00"),
      .maxTravelTime_ = 60,
      .maxMatchingDistance_ = 250.0,
      .arriveBy_ = false,
      .useRoutedTransfers_ = true,
      .maxPreTransitTime_ = 360});  // Too short to reach U4

  EXPECT_EQ((api::OneToManyIntermodalResponse{
                .street_durations_ = {{
                    {},
                    {.duration_ = 443.0},
                    {.duration_ = 370.0},  // Direct connection allowed
                    {.duration_ = 321.0},  // Valid for pre transit
                    {},
                    {},
                    {.duration_ = 403.0},
                    {},
                    {},
                    {},
                    {},
                }},
                .transit_durations_ = {{
                    {},
                    {},
                    {},
                    {},
                    {{.duration_ = 1560.0, .transfers_ = 0}},  // Must take S3
                    {{.duration_ = 1680.0, .transfers_ = 0}},  // Must take S3
                    {},
                    {{.duration_ = 1800.0, .transfers_ = 0}},
                    {{.duration_ = 1740.0, .transfers_ = 0}},
                    {{.duration_ = 1740.0, .transfers_ = 0}},
                    {{.duration_ = 1680.0, .transfers_ = 0}},

                }}}),
            durations);
}

TEST(one_to_many, get_request_backward_with_wheelchair_and_short_post_transit) {
  auto [d, _] = get_test_case<test_case::FFM_one_to_many>();

  auto const durations = one_to_many_get(d)(
      "/api/experimental/one-to-many-intermodal"
      "?one=50.11385;8.67912"  // FFM_HAUPT_U
      "&many="
      "50.107577;8.6638173,"  // de:6412:10:6:1
      "50.10739;8.66333,"  // FFM_101
      "50.11404;8.67824,"  // FFM_HAUPT_S
      "50.113465;8.678477,"  // Near FFM_HAUPT
      "50.112519;8.676565"  // Far, near FFM_HAUPT
      "&time=2019-04-30T23:30:00.000Z"
      "&maxTravelTime=60"
      "&maxMatchingDistance=250"
      "&maxDirectTime=540"
      "&maxPostTransitTime=240"
      "&pedestrianProfile=WHEELCHAIR"
      "&useRoutedTransfers=true"
      "&withDistance=true"
      "&arriveBy=true");

  auto const& sd = durations.street_durations_.value();
  auto const& td = durations.transit_durations_.value();

  ASSERT_EQ(5U, sd.size());
  EXPECT_EQ(api::Duration{}, sd.at(0));
  EXPECT_EQ(api::Duration{}, sd.at(1));
  // Not valid for post transit => unreachable from FFM_101
  EXPECT_DOUBLE_EQ(333.0, sd.at(2).duration_.value());
  EXPECT_NEAR(124.1, sd.at(2).distance_.value(), 0.1);
  EXPECT_DOUBLE_EQ(517.0, sd.at(3).duration_.value());
  EXPECT_NEAR(271.8, sd.at(3).distance_.value(), 0.1);
  EXPECT_DOUBLE_EQ(771.0, sd.at(4).duration_.value());
  EXPECT_NEAR(476.0, sd.at(4).distance_.value(), 0.1);

  ASSERT_EQ(5U, td.size());
  ASSERT_EQ(1U, td.at(0).size());
  EXPECT_DOUBLE_EQ(1680.0, td.at(0).at(0).duration_);
  EXPECT_EQ(0, td.at(0).at(0).transfers_);
  // Unreachable, as FFM_HAUPT_S -> FFM_HAUPT_U not usable postTransit
  EXPECT_TRUE(td.at(1).empty());
  EXPECT_TRUE(td.at(2).empty());
  EXPECT_TRUE(td.at(3).empty());
  EXPECT_TRUE(td.at(4).empty());
}

TEST(one_to_many, oneway_get_forward_for_pre_transit_and_direct_modes) {
  auto [d, _] = get_test_case<test_case::FFM_one_to_many>();

  auto const durations = one_to_many_get(d)(
      "/api/experimental/one-to-many-intermodal"
      "?one=50.107328;8.664836"
      "&many="
      "50.107812;8.664628,"  // FFM C  (shorter path)
      "50.107519;8.664775,"  // FFM B  (longer path, due to oneway)
      "50.110828;8.681587"  // PAUL2
      "&time=2019-05-01T10:00:00.00Z"
      "&maxTravelTime=60"
      "&maxMatchingDistance=250"
      "&maxDirectTime=3600"
      "&directMode=BIKE"
      "&preTransitModes=BIKE"
      "&arriveBy=false"
      "&cyclingSpeed=2.4");

  EXPECT_EQ((api::OneToManyIntermodalResponse{
                .street_durations_ = {{
                    {.duration_ = 228.0},
                    {.duration_ = 321.0},
                    {},
                }},
                .transit_durations_ = {{
                    {},
                    {},
                    {// Must use later trip
                     {.duration_ = 1980.0, .transfers_ = 0}},
                }}}),
            durations);
}

TEST(one_to_many, oneway_post_backward_for_post_transit_and_direct_modes) {
  auto [d, _] = get_test_case<test_case::FFM_one_to_many>();

  auto const durations = one_to_many_post(d)(api::OneToManyIntermodalParams{
      .one_ = "50.107326,8.665237",
      .many_ = {"test_FFM_B", "test_FFM_C", "50.107812,8.664628", "test_PAUL1"},
      .time_ = parse_time("2019-05-01T12:30:00.000+02:00"),
      .maxTravelTime_ = 60,
      .maxMatchingDistance_ = 250.0,
      .arriveBy_ = true,
      .cyclingSpeed_ = 2.2,
      .postTransitModes_ = {api::ModeEnum::BIKE},
      .directMode_ = api::ModeEnum::BIKE,
      .withDistance_ = true});

  auto const& sd = durations.street_durations_.value();
  auto const& td = durations.transit_durations_.value();

  ASSERT_EQ(4U, sd.size());
  EXPECT_DOUBLE_EQ(228.0, sd.at(0).duration_.value());
  EXPECT_NEAR(341.3, sd.at(0).distance_.value(), 0.1);
  EXPECT_DOUBLE_EQ(335.0, sd.at(1).duration_.value());
  EXPECT_NEAR(502.1, sd.at(1).distance_.value(), 0.1);
  EXPECT_DOUBLE_EQ(335.0, sd.at(2).duration_.value());
  EXPECT_NEAR(502.1, sd.at(2).distance_.value(), 0.1);
  EXPECT_EQ(api::Duration{}, sd.at(3));

  ASSERT_EQ(4U, td.size());
  EXPECT_TRUE(td.at(0).empty());
  EXPECT_TRUE(td.at(1).empty());
  EXPECT_TRUE(td.at(2).empty());
  ASSERT_EQ(1U, td.at(3).size());
  EXPECT_DOUBLE_EQ(1920.0, td.at(3).at(0).duration_);
  EXPECT_EQ(0, td.at(3).at(0).transfers_);
}

TEST(one_to_many, oneway_post_forward_for_post_transit_modes) {
  auto [d, _] = get_test_case<test_case::FFM_one_to_many>();

  auto const durations = one_to_many_post(d)(api::OneToManyIntermodalParams{
      .one_ = "test_PAUL1",
      .many_ = {"test_FFM_C", "50.107326,8.665237"},  // includes C -> B
      .time_ = parse_time("2019-05-01T12:00:00.000+02:00"),
      .maxTravelTime_ = 30,
      .maxMatchingDistance_ = 250.0,
      .arriveBy_ = false,
      .postTransitModes_ = {api::ModeEnum::BIKE}});

  EXPECT_EQ((api::OneToManyIntermodalResponse{
                .street_durations_ = std::vector<api::Duration>(2),
                .transit_durations_ = {{
                    {{.duration_ = 720.0, .transfers_ = 0}},
                    {{.duration_ = 840.0, .transfers_ = 0}},
                }}}),
            durations);
}

TEST(one_to_many, oneway_get_backward_for_pre_transit_modes) {
  auto [d, _] = get_test_case<test_case::FFM_one_to_many>();

  auto const durations = one_to_many_get(d)(
      "/api/experimental/one-to-many-intermodal"
      "?one=50.110828;8.681587"  // PAUL2
      "&many="
      "50.107812;8.664628,"  // FFM C  (with incorrect transfer C -> B)
      "50.107519;8.664775,"  // FFM B
      "50.107328;8.664836"  // Long preTransit due to oneway  (C -> B)
      "&time=2019-05-01T10:20:00.00Z"
      "&maxTravelTime=60"
      "&maxMatchingDistance=250"
      "&preTransitModes=BIKE"
      "&arriveBy=true"
      "&cyclingSpeed=2.4");

  EXPECT_EQ((api::OneToManyIntermodalResponse{
                .street_durations_ = std::vector<api::Duration>(3),
                .transit_durations_ = {{
                    {{.duration_ = 1080.0, .transfers_ = 0}},
                    {{.duration_ = 1080.0, .transfers_ = 0}},
                    {{.duration_ = 1260.0, .transfers_ = 0}},
                }}}),
            durations);
}

TEST(one_to_many, transfer_time_settings_min_transfer_time) {
  auto [d, _] = get_test_case<test_case::FFM_one_to_many>();

  auto const durations = one_to_many_get(d)(
      "/api/experimental/one-to-many-intermodal"
      "?one=49.872710;8.631168"  // Near DA
      "&many=50.113487;8.678913"  // Near FFM_HAUPT
      "&time=2019-04-30T22:30:00.00Z"
      "&useRoutedTransfers=true"
      "&minTransferTime=21");

  // FIXME Times are also added for final footpath
  EXPECT_EQ((api::OneToManyIntermodalResponse{
                .street_durations_ = {{api::Duration{}}},
                .transit_durations_ = {{
                    {{.duration_ = 4320.0, .transfers_ = 1}},
                }}}),
            durations);
}

TEST(one_to_many, transfer_time_settings_additional_transfer_time) {
  auto [d, _] = get_test_case<test_case::FFM_one_to_many>();

  auto const durations = one_to_many_post(d)(api::OneToManyIntermodalParams{
      .one_ = "49.872710, 8.631168",  // Near DA
      .many_ = {"50.113487, 8.678913"},  // Near FFM_HAUPT
      .time_ = parse_time("2019-05-01T00:30:00.000+02:00"),
      .additionalTransferTime_ = 17,
      .useRoutedTransfers_ = true});

  // FIXME Times are also added for final footpath
  EXPECT_EQ((api::OneToManyIntermodalResponse{
                .street_durations_ = {{api::Duration{}}},
                .transit_durations_ = {{
                    {{.duration_ = 4200.0, .transfers_ = 1}},
                }}}),
            durations);
}

TEST(one_to_many, bug_additional_footpath_for_first_last_mile) {
  // Bug examples: Should not connect final footpath with first or last mile
  auto [d, _] = get_test_case<test_case::FFM_one_to_many>();

  auto const ep = one_to_many_post(d);
  auto const many = std::vector<std::string>{
      "test_FFM_HAUPT_U", "50.11385,8.67912,-4",  // FFM_HAUPT_U
      "test_FFM_HAUPT_S", "50.11404,8.67824,-3",  // FFM_HAUPT_S
      "50.114093,8.676546"};  // Test location
  {
    // Last location should not be reachable when only arriving with U4
    auto const durations = ep(api::OneToManyIntermodalParams{
        .one_ = "50.108056,8.663177,-2",  // Near de:6412:10:6:1
        .many_ = many,
        .time_ = parse_time("2019-05-01T01:00:00.000+02:00"),
        .maxTravelTime_ = 30,
        .maxMatchingDistance_ = 250.0,
        .arriveBy_ = false,
        .useRoutedTransfers_ = true,
        .pedestrianProfile_ = api::PedestrianProfileEnum::WHEELCHAIR,
        .maxPostTransitTime_ = 420});  // Too short to reach from U4

    EXPECT_EQ((api::OneToManyIntermodalResponse{
                  .street_durations_ = std::vector<api::Duration>(5),
                  .transit_durations_ = {{
                      {{.duration_ = 720.0, .transfers_ = 0}},
                      {{.duration_ = 780.0, .transfers_ = 0}},
                      {{.duration_ = 720.0, .transfers_ = 0}},
                      {{.duration_ = 1020.0, .transfers_ = 0}},
                      {// FIXME Test location should be unreachable
                       {.duration_ = 1380.0, .transfers_ = 0}},
                  }}}),
              durations);
  }

  {
    // Test that location is reachable from FFM_HAUPT_S after arrival
    auto const test_durations = ep(api::OneToManyIntermodalParams{
        .one_ = "50.10739,8.66333,-3",  // Near FFM_101
        .many_ = many,
        .time_ = parse_time("2019-05-01T01:00:00.000+02:00"),
        .maxTravelTime_ = 30,
        .maxMatchingDistance_ = 250.0,
        .arriveBy_ = false,
        .useRoutedTransfers_ = true,
        .pedestrianProfile_ = api::PedestrianProfileEnum::WHEELCHAIR,
        .maxPostTransitTime_ = 420});  // Reachable from S3
    EXPECT_EQ((api::OneToManyIntermodalResponse{
                  .street_durations_ = std::vector<api::Duration>(5),
                  .transit_durations_ = {{
                      {{.duration_ = 1260.0, .transfers_ = 0}},
                      {{.duration_ = 1620.0, .transfers_ = 0}},
                      {{.duration_ = 1260.0, .transfers_ = 0}},
                      {{.duration_ = 1380.0, .transfers_ = 0}},
                      {{.duration_ = 1740.0, .transfers_ = 0}},
                  }}}),
              test_durations);
  }

  {
    // Ensure all arriving stations are reachable
    // Also check that the correct arrival time is used
    // Should start from FFM_HAUPT_S due to post transit time constraint
    auto const walk_durations = ep(api::OneToManyIntermodalParams{
        .one_ = "50.107066,8.663604,0",
        .many_ = many,
        .time_ = parse_time("2019-05-01T01:00:00.000+02:00"),
        .maxTravelTime_ = 30,
        .maxMatchingDistance_ = 250.0,
        .arriveBy_ = false,
        .useRoutedTransfers_ = true,
        .pedestrianProfile_ = api::PedestrianProfileEnum::FOOT,
        .maxPostTransitTime_ = 240});  // Only reachable from S3
    EXPECT_EQ((api::OneToManyIntermodalResponse{
                  .street_durations_ = std::vector<api::Duration>(5),
                  .transit_durations_ = {{
                      {{.duration_ = 720.0, .transfers_ = 0}},
                      {{.duration_ = 780.0, .transfers_ = 0}},
                      {{.duration_ = 720.0, .transfers_ = 0}},
                      {{.duration_ = 780.0, .transfers_ = 0}},
                      {// FIXME Should start FFM_HAUPT_S => time > 1200
                       {.duration_ = 960.0, .transfers_ = 0}},
                  }}}),
              walk_durations);
  }
}

TEST(one_to_many, pareto_sets_with_routed_transfers_and_distances) {
  auto [d, _] = get_test_case<test_case::FFM_one_to_many>();

  auto const durations = one_to_many_post(d)(api::OneToManyIntermodalParams{
      .one_ = "49.8722160,8.6282315",  // DA_Bus_1
      .many_ = {"49.875292,8.6277460",  // near Tram_1, must be south of node
                "49.874995,8.6313925",  // near Tram_2
                "49.871561,8.6320181",  // near Tram_3
                "50.111900,8.675208"},  // near FFM_HAUPT
      .time_ = parse_time("2019-05-01T00:05:00.000+02:00"),
      .useRoutedTransfers_ = true,
      .withDistance_ = true});

  auto const& sd = durations.street_durations_.value();
  auto const& td = durations.transit_durations_.value();

  ASSERT_EQ(4U, sd.size());
  EXPECT_DOUBLE_EQ(344.0, sd.at(0).duration_.value());
  EXPECT_NEAR(351.9, sd.at(0).distance_.value(), 0.1);
  EXPECT_DOUBLE_EQ(556.0, sd.at(1).duration_.value());
  EXPECT_NEAR(607.0, sd.at(1).distance_.value(), 0.1);
  EXPECT_DOUBLE_EQ(966.0, sd.at(2).duration_.value());
  EXPECT_NEAR(1100.6, sd.at(2).distance_.value(), 0.1);
  EXPECT_EQ(api::Duration{}, sd.at(3));

  ASSERT_EQ(4U, td.size());
  ASSERT_EQ(1U, td.at(0).size());
  EXPECT_DOUBLE_EQ(1320.0, td.at(0).at(0).duration_);
  EXPECT_EQ(0, td.at(0).at(0).transfers_);
  ASSERT_EQ(1U, td.at(1).size());
  EXPECT_DOUBLE_EQ(1680.0, td.at(1).at(0).duration_);
  EXPECT_EQ(0, td.at(1).at(0).transfers_);
  ASSERT_EQ(1U, td.at(2).size());
  EXPECT_DOUBLE_EQ(1740.0, td.at(2).at(0).duration_);
  EXPECT_EQ(0, td.at(2).at(0).transfers_);
  ASSERT_EQ(1U, td.at(3).size());
  EXPECT_DOUBLE_EQ(4440.0, td.at(3).at(0).duration_);
  EXPECT_EQ(2, td.at(3).at(0).transfers_);
}

TEST(one_to_many, pareto_sets_with_multiple_entries) {
  // Long walking paths + fast connctions => multiple durations
  // Currently: Long transfer times, so that transit is faster
  // After bug fix: Slow walking speed, so that transit is faster
  // might require moving stops (B2->T1, T1->T2, T2 delete) with paths:
  // Bus1 -> Tram3, Bus1 -> Bus2 -> Tram3, Bus1 -> Bus2 -> Tram1/2 -> Tram3
  auto [d, _] = get_test_case<test_case::FFM_one_to_many>();

  auto const durations = one_to_many_post(d)(api::OneToManyIntermodalParams{
      .one_ = "49.8722160,8.6282315",  // DA_Bus_1
      .many_ = {"49.8755778,8.6240518",  // DA_Bus_2
                "49.8752926,8.6277460",  // DA_Tram_1
                "49.871561,8.6320181"},  // DA_Tram_3
      .time_ = parse_time("2019-05-01T00:05:00.000+02:00"),
      .maxPreTransitTime_ = 300});  // Prevent any pre transit to Tram_x

  // We only care about duration to DA_Tram_3, everything else is for debugging
  auto const& sd = durations.street_durations_.value();
  ASSERT_EQ(3U, sd.size());
  EXPECT_DOUBLE_EQ(966.0, sd.at(2).duration_.value());
  EXPECT_EQ((std::optional<std::vector<std::vector<api::ParetoSetEntry>>>{{
                {{.duration_ = 1080.0, .transfers_ = 0}},
                {{.duration_ = 1140.0, .transfers_ = 0}},
                {{.duration_ = 1500.0, .transfers_ = 0},
                 {.duration_ = 1440.0, .transfers_ = 1}},
            }}),
            durations.transit_durations_);
}
