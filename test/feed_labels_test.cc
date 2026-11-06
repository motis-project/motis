#include "gtest/gtest.h"

#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "utl/helpers/algorithm.h"
#include "utl/init_from.h"

#include "net/bad_request_exception.h"
#include "net/not_found_exception.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/adr/geocode.h"
#include "motis/endpoints/adr/reverse_geocode.h"
#include "motis/endpoints/map/route_details.h"
#include "motis/endpoints/map/routes.h"
#include "motis/endpoints/matches.h"
#include "motis/endpoints/one_to_all.h"
#include "motis/endpoints/refresh_itinerary.h"
#include "motis/endpoints/routing.h"
#include "motis/endpoints/stop.h"
#include "motis/endpoints/stop_times.h"
#include "motis/endpoints/transfers.h"
#include "motis/gbfs/routing_data.h"
#include "motis/import.h"
#include "motis/osr/parameters.h"
#include "motis/tag_lookup.h"

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
  // Without it, the bus leg can't be reconstructed and is cancelled.
  auto const res =
      routing(std::string{kAToB} + "&directModes=&includeLabels=hotel");
  ASSERT_EQ(1U, res.itineraries_.size());
  auto const refresh = utl::init_from<ep::refresh_itinerary>(d).value();
  auto query = api::refreshItinerary_params{};
  query.itineraryId_ = res.itineraries_.front().id_;
  query.includeLabels_ = {"hotel"};
  auto const with_label = refresh(query.to_url("?"));
  ASSERT_EQ(3U, with_label.legs_.size());
  EXPECT_EQ("H", with_label.legs_[1].routeShortName_);
  EXPECT_FALSE(with_label.legs_[1].cancelled_.value_or(false));

  query.includeLabels_ = {};
  auto const without_label = refresh(query.to_url("?"));
  ASSERT_EQ(3U, without_label.legs_.size());
  EXPECT_TRUE(without_label.legs_[1].cancelled_.value_or(false));
  EXPECT_FALSE(without_label.legs_[1].tripId_.has_value());
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

namespace {

constexpr auto const kHotelStop = "Hotel Beispielberg Eingang"sv;

bool is_hotel_stop(api::Place const& p) { return p.name_ == kHotelStop; }

}  // namespace

TEST(motis, feed_labels_hidden_stops) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_feed_labels_hidden_stops", ec);

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
      .geocoding_ = true,
      .reverse_geocoding_ = true};
  import(c, "test/data_feed_labels_hidden_stops");
  auto d = data{"test/data_feed_labels_hidden_stops", c};
  auto const label = "&includeLabels=hotel"sv;

  // Trips of a hidden dataset end at its stop only with the label.
  auto const routing = utl::init_from<ep::routing>(d).value();
  auto const hotel_stop_in_plan = [&](std::string_view const labels) {
    auto const res = routing(
        "?fromPlace=50.1062497,8.6582223"
        "&toPlace=50.1085000,8.6612000"
        "&time=2019-05-01T07:30Z&timetableView=false&directModes="
        "&maxPostTransitTime=60" +
        std::string{labels});
    return utl::any_of(res.itineraries_, [](api::Itinerary const& it) {
      return utl::any_of(it.legs_, [](api::Leg const& l) {
        return is_hotel_stop(l.from_) || is_hotel_stop(l.to_);
      });
    });
  };
  EXPECT_FALSE(hotel_stop_in_plan(""));
  EXPECT_TRUE(hotel_stop_in_plan(label));

  // Stops of inactive datasets are no access or egress candidates.
  auto gbfs_rd = gbfs::gbfs_routing_data{d.w_.get(), d.l_.get(), nullptr};
  auto stats = ep::stats_map_t{};
  auto const reaches_hidden_stop = [&](std::vector<std::string> const& labels) {
    auto const offsets = routing.get_offsets(
        nullptr, osr::location{{50.1085000, 8.6612000}, osr::kNoLevel},
        osr::direction::kBackward, {api::ModeEnum::WALK}, rental_options{},
        osr_parameters{}, api::PedestrianProfileEnum::FOOT,
        api::ElevationCostsEnum::NONE, std::chrono::minutes{5}, 100.0, gbfs_rd,
        d.labels_.blocked(*d.tt_, nullptr, labels, {}), stats);
    auto const hidden = d.tags_->get_src("shuttle");
    return utl::any_of(offsets, [&](nigiri::routing::offset const& o) {
      return d.tt_->locations_.src_[o.target()] == hidden;
    });
  };
  EXPECT_FALSE(reaches_hidden_stop({}));
  EXPECT_TRUE(reaches_hidden_stop({"hotel"}));

  // A hidden stop is reported like a stop id that does not exist.
  auto const plan_to = [&](std::string_view const place,
                           std::string_view const labels) {
    return routing(
        "?fromPlace=50.1062497,8.6582223&toPlace=" + std::string{place} +
        "&time=2019-05-01T07:30Z&timetableView=false&directModes=" +
        std::string{labels});
  };
  EXPECT_THROW(plan_to("shuttle_C", ""), net::not_found_exception);
  EXPECT_THROW(plan_to("shuttle_missing", ""), net::not_found_exception);
  EXPECT_NO_THROW(plan_to("shuttle_C", label));

  auto const stop = utl::init_from<ep::stop>(d).value();
  EXPECT_THROW(stop("/api/v6/stop?stopId=shuttle_C"sv),
               net::bad_request_exception);
  EXPECT_THROW(stop("/api/v6/stop?stopId=shuttle_missing"sv),
               net::bad_request_exception);
  EXPECT_EQ(
      kHotelStop,
      stop("/api/v6/stop?stopId=shuttle_C&includeLabels=hotel"sv).place_.name_);

  auto const stop_times = utl::init_from<ep::stop_times>(d).value();
  auto const st = [&](std::string_view const id,
                      std::string_view const labels) {
    return stop_times("/api/v5/stoptimes?stopId=" + std::string{id} +
                      "&time=2019-05-01T07:30Z&n=3" + std::string{labels});
  };
  EXPECT_THROW(st("shuttle_C", ""), net::not_found_exception);
  EXPECT_THROW(st("shuttle_missing", ""), net::not_found_exception);
  EXPECT_EQ(kHotelStop, st("shuttle_C", label).place_.name_);

  // Stops reached by a footpath are not reachable places.
  auto const one_to_all = utl::init_from<ep::one_to_all>(d).value();
  auto const hotel_stop_reachable = [&](std::string_view const labels) {
    auto const res = one_to_all(
        "/api/v1/one-to-all?one=50.1062497,8.6582223"
        "&time=2019-05-01T07:30Z&maxTravelTime=60" +
        std::string{labels});
    return utl::any_of(res.all_.value(), [](api::ReachablePlace const& p) {
      return is_hotel_stop(p.place_.value());
    });
  };
  EXPECT_FALSE(hotel_stop_reachable(""));
  EXPECT_TRUE(hotel_stop_reachable(label));

  auto const reverse = utl::init_from<ep::reverse_geocode>(d).value();
  auto const hotel_stop_nearby = [&](std::string_view const labels) {
    auto const res = reverse(
        "/api/v1/reverse-geocode?place=50.1085000,8.6612000"
        "&type=STOP&numResults=10" +
        std::string{labels});
    return utl::any_of(
        res, [](api::Match const& m) { return m.name_ == kHotelStop; });
  };
  EXPECT_FALSE(hotel_stop_nearby(""));
  EXPECT_TRUE(hotel_stop_nearby(label));

  auto const routes = utl::init_from<ep::routes>(d).value();
  auto const route_h = [&](std::string_view const labels) {
    auto const res = routes(
        "/api/experimental/map/routes?min=50.2,8.5&max=50.0,8.8&zoom=20" +
        std::string{labels});
    auto const it = utl::find_if(res.routes_, [](api::RouteInfo const& r) {
      return utl::any_of(r.transitRoutes_, [](api::TransitRouteInfo const& t) {
        return t.shortName_ == "H";
      });
    });
    return it == end(res.routes_) ? std::optional<std::int64_t>{}
                                  : std::optional{it->routeIdx_};
  };
  EXPECT_FALSE(route_h("").has_value());
  auto const route_idx = route_h(label);
  ASSERT_TRUE(route_idx.has_value());

  auto const route_details = utl::init_from<ep::route_details>(d).value();
  auto const details = fmt::format(
      "/api/experimental/map/route-details?routeIdx={}", *route_idx);
  EXPECT_THROW(route_details(details), net::not_found_exception);
  EXPECT_NO_THROW(route_details(details + "&includeLabels=hotel"));

  auto const matches = utl::init_from<ep::matches>(d).value();
  auto const matched = boost::json::serialize(
      matches(boost::json::value{8.65, 50.10, 8.67, 50.11}));
  EXPECT_EQ(std::string::npos, matched.find(kHotelStop));

  auto const transfers = utl::init_from<ep::transfers>(d).value();
  EXPECT_THROW(transfers("/api/debug/transfers?id=shuttle_C"sv),
               net::not_found_exception);
  auto const public_stop = transfers("/api/debug/transfers?id=public_B"sv);
  EXPECT_FALSE(utl::any_of(public_stop.transfers_, [](api::Transfer const& t) {
    return is_hotel_stop(t.to_);
  }));
}
