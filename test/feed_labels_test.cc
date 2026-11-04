#include "gtest/gtest.h"

#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "utl/helpers/algorithm.h"
#include "utl/init_from.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/adr/geocode.h"
#include "motis/endpoints/refresh_itinerary.h"
#include "motis/endpoints/routing.h"
#include "motis/import.h"

using namespace std::string_view_literals;
using namespace motis;

namespace {

// Public bus P: A -> B, 10:00 - 10:05.
constexpr auto const kPublicGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
PUB,Musterland Bus,https://example.org,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,Musterdorf Hauptplatz,50.1062497,8.6582223
B,Beispielberg Bahnhof,50.1079161,8.6605383

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
BUS_P,PUB,P,,3

# trips.txt
route_id,service_id,trip_id
BUS_P,S1,bus_p

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
bus_p,10:00:00,10:00:00,A,1
bus_p,10:05:00,10:05:00,B,2

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

// Private hotel shuttle: bus H (A -> B -> C, 10:00 - 10:05) and flex trip
// lg_h. Stop C only exists in this dataset.
constexpr auto const kHotelGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
HOT,Hotel Beispielberg,https://example.org,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,Musterdorf Hauptplatz,50.1062497,8.6582223
B,Beispielberg Bahnhof,50.1079161,8.6605383
C,Hotel Beispielberg Eingang,50.1085000,8.6612000

# location_groups.txt
location_group_id,location_group_name
lg_h,Hotel Shuttle

# location_group_stops.txt
location_group_id,stop_id
lg_h,A
lg_h,B

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
BUS_H,HOT,H,,3
FLEX_H,HOT,,Hotel Shuttle,715

# trips.txt
route_id,service_id,trip_id
BUS_H,S1,bus_h
FLEX_H,S1,flex_h

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,location_group_id,start_pickup_drop_off_window,end_pickup_drop_off_window,pickup_type,drop_off_type
bus_h,10:00:00,10:00:00,A,1,,,,0,0
bus_h,10:03:00,10:03:00,B,2,,,,0,0
bus_h,10:05:00,10:05:00,C,3,,,,0,0
flex_h,,,,1,lg_h,08:30:00,18:00:00,2,1
flex_h,,,,2,lg_h,08:30:00,18:00:00,1,2

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

// Public on-demand bus: flex trip lg_v.
constexpr auto const kRufbusGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
RUF,Rufbus Musterland,https://example.org,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,Musterdorf Hauptplatz,50.1062497,8.6582223
B,Beispielberg Bahnhof,50.1079161,8.6605383

# location_groups.txt
location_group_id,location_group_name
lg_v,Rufbus

# location_group_stops.txt
location_group_id,stop_id
lg_v,A
lg_v,B

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
FLEX_V,RUF,,Rufbus,715

# trips.txt
route_id,service_id,trip_id
FLEX_V,S1,flex_v

# stop_times.txt
trip_id,stop_sequence,location_group_id,start_pickup_drop_off_window,end_pickup_drop_off_window,pickup_type,drop_off_type
flex_v,1,lg_v,08:30:00,18:00:00,2,1
flex_v,2,lg_v,08:30:00,18:00:00,1,2

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

constexpr auto const kAToB =
    "?fromPlace=50.1062497,8.6582223"
    "&toPlace=50.1079161,8.6605383"
    "&time=2019-05-01T07:30Z"
    "&timetableView=false"sv;

std::set<std::string> transit_routes(api::plan_response const& res) {
  auto routes = std::set<std::string>{};
  for (auto const& it : res.itineraries_) {
    for (auto const& l : it.legs_) {
      if (l.mode_ == api::ModeEnum::BUS) {
        routes.emplace(l.routeShortName_.value_or(""));
      }
    }
  }
  return routes;
}

std::set<std::string> flex_ids(api::plan_response const& res) {
  auto ids = std::set<std::string>{};
  for (auto const& it : res.direct_) {
    for (auto const& l : it.legs_) {
      if (l.mode_ == api::ModeEnum::FLEX) {
        ids.emplace(l.from_.flexId_.value_or(""));
      }
    }
  }
  return ids;
}

}  // namespace

TEST(motis, feed_labels) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_feed_labels", ec);

  auto const c = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2U,
              .datasets_ = {{"public", {.path_ = std::string{kPublicGtfs}}},
                            {"shuttle",
                             {.path_ = std::string{kHotelGtfs},
                              .labels_ = std::vector<std::string>{"hotel"},
                              .hidden_ = true}},
                            {"rufbus", {.path_ = std::string{kRufbusGtfs}}}}},
      .street_routing_ = true};
  import(c, "test/data_feed_labels");
  auto d = data{"test/data_feed_labels", c};
  auto const routing = utl::init_from<ep::routing>(d).value();

  auto const transit = [&](std::string_view const labels) {
    return transit_routes(
        routing(std::string{kAToB} + "&directModes=" + std::string{labels}));
  };
  auto const flex = [&](std::string_view const labels) {
    return flex_ids(routing(std::string{kAToB} +
                            "&transitModes=&directModes=FLEX" +
                            std::string{labels}));
  };

  // Hidden datasets are only used when requested.
  EXPECT_EQ((std::set<std::string>{"P"}), transit(""));
  EXPECT_EQ((std::set<std::string>{"H"}), transit("&includeLabels=hotel"));
  EXPECT_EQ((std::set<std::string>{"P"}), transit("&excludeLabels=shuttle"));

  EXPECT_EQ((std::set<std::string>{"lg_v"}), flex(""));
  EXPECT_EQ((std::set<std::string>{"lg_h", "lg_v"}),
            flex("&includeLabels=hotel"));

  // Named visible datasets limit the request to them.
  EXPECT_EQ((std::set<std::string>{"lg_v"}), flex("&includeLabels=rufbus"));
  EXPECT_EQ((std::set<std::string>{"lg_h"}),
            flex("&includeLabels=rufbus,hotel&excludeLabels=rufbus"));
  EXPECT_TRUE(flex("&excludeLabels=rufbus").empty());

  // Refreshing an itinerary of a hidden dataset requires its label.
  auto const res =
      routing(std::string{kAToB} + "&directModes=&includeLabels=hotel");
  ASSERT_EQ(1U, res.itineraries_.size());
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = res.itineraries_.front().id_;
  query.includeLabels_ = {"hotel"};
  EXPECT_EQ(res.itineraries_.front().legs_.size(),
            refresh(query.to_url("?")).legs_.size());
  query.includeLabels_ = {};
  EXPECT_ANY_THROW(refresh(query.to_url("?")));
}

TEST(motis, feed_labels_geocode) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_feed_labels_geocode", ec);

  auto const c = config{
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2U,
              .datasets_ = {{"public", {.path_ = std::string{kPublicGtfs}}},
                            {"shuttle",
                             {.path_ = std::string{kHotelGtfs},
                              .labels_ = std::vector<std::string>{"hotel"},
                              .hidden_ = true}}}},
      .street_routing_ = true,
      .geocoding_ = true};
  import(c, "test/data_feed_labels_geocode");
  auto d = data{"test/data_feed_labels_geocode", c};
  auto const geocode = utl::init_from<ep::geocode>(d).value();

  auto const has_stop = [&](std::string const& name,
                            std::vector<std::string> const& include,
                            std::vector<std::string> const& exclude) {
    auto query = api::geocode_params{};
    query.text_ = name;
    query.includeLabels_ = include;
    query.excludeLabels_ = exclude;
    auto const res = geocode(query.to_url("?"));
    return utl::any_of(res, [&](api::Match const& m) {
      return m.type_ == api::LocationTypeEnum::STOP && m.name_ == name;
    });
  };

  // Stops that only exist in a hidden dataset require its label.
  EXPECT_FALSE(has_stop("Hotel Beispielberg Eingang", {}, {}));
  EXPECT_TRUE(has_stop("Hotel Beispielberg Eingang", {"hotel"}, {}));

  // Stops that also exist in a visible dataset are found unless it is excluded.
  EXPECT_TRUE(has_stop("Musterdorf Hauptplatz", {}, {}));
  EXPECT_FALSE(has_stop("Musterdorf Hauptplatz", {}, {"public"}));
}
