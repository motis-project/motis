#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "net/bad_request_exception.h"

#include "utl/init_from.h"
#include "utl/to_vec.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/map/route_search.h"
#include "motis/import.h"

using namespace motis;
using namespace testing;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
Test,Test Agency,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
DA_1,DA Hbf,49.8724891,8.6281994
DA_2,DA Sued,49.8750407,8.6312172
DA_3,DA Ost,49.8742551,8.6321063
HH_1,HH Hbf,53.5528000,10.0067000
HH_2,HH Altona,53.5575000,10.0150000

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
L9_DA,Test,L9,Línia 9,1
L9_HH,Test,L9,Línia 9,1
L91,Test,L91,Línia 91,3
N9,Test,N9,Bus L9 Nit,3
L9X,Test,L9X,Línia 9 hivern,3

# trips.txt
route_id,service_id,trip_id,trip_headsign
L9_DA,S1,T_L9_DA,Nord
L9_HH,S1,T_L9_HH,Nord
L91,S1,T_L91,Nord
N9,S1,T_N9,Nord
L9X,S_OLD,T_L9X,Nord

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T_L9_DA,01:00:00,01:00:00,DA_1,1
T_L9_DA,01:10:00,01:10:00,DA_2,2
T_L9_HH,01:00:00,01:00:00,HH_1,1
T_L9_HH,01:10:00,01:10:00,HH_2,2
T_L91,01:00:00,01:00:00,DA_2,1
T_L91,01:10:00,01:10:00,DA_3,2
T_N9,01:00:00,01:00:00,DA_3,1
T_N9,01:10:00,01:10:00,DA_1,2
T_L9X,01:00:00,01:00:00,DA_1,1
T_L9X,01:10:00,01:10:00,DA_3,2

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
S_OLD,20190401,1
)";

namespace {

std::vector<std::string> ids(api::routeSearch_response const& res) {
  return utl::to_vec(res.routes_,
                     [](api::RouteMatch const& m) { return m.transitRoute_.id_; });
}

}  // namespace

TEST(motis, map_route_search) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c = config{.timetable_ = config::timetable{
                            .first_day_ = "2019-05-01",
                            .num_days_ = 2,
                            .datasets_ = {{"test", {.path_ = kGTFS}}}}};
  import(c, "test/data");
  auto d = data{"test/data", c};

  auto const search = utl::init_from<ep::route_search>(d).value();

  {
    auto const res =
        search("/api/experimental/map/route-search?text=L9");

    EXPECT_THAT(ids(res), ElementsAre("L9_DA", "L9_HH", "L91", "N9"));
    EXPECT_THAT(ids(res), Not(Contains("L9X")));

    EXPECT_EQ(res.routes_[0].mode_, api::ModeEnum::SUBWAY);
    EXPECT_EQ(res.routes_[3].mode_, api::ModeEnum::BUS);
    EXPECT_THAT(res.routes_[0].agencyName_,
                Optional(std::string{"Test Agency"}));
    EXPECT_FALSE(res.routes_[0].routeIndexes_.empty());
    EXPECT_THAT(res.routes_[0].transitRoute_.shortName_, Eq("L9"));
    EXPECT_THAT(res.routes_[0].transitRoute_.longName_, Eq("Línia 9"));
  }

  {
    auto const res = search(
        "/api/experimental/map/route-search?text=L9&place=49.87,8.63"
        "&placeBias=5");

    EXPECT_THAT(ids(res), ElementsAre("L9_DA", "L91", "N9", "L9_HH"));
  }

  {
    auto const unbiased =
        search("/api/experimental/map/route-search?text=L9");
    auto const zero_bias = search(
        "/api/experimental/map/route-search?text=L9&place=53.55,10.0"
        "&placeBias=0");

    EXPECT_EQ(ids(unbiased), ids(zero_bias));
  }

  {
    auto const res = search("/api/experimental/map/route-search?text=L");

    EXPECT_THAT(ids(res), Contains("L9_DA"));
    EXPECT_THAT(ids(res), Not(Contains("L9X")));
  }

  {
    auto const res =
        search("/api/experimental/map/route-search?text=linia%209");

    EXPECT_THAT(ids(res), Contains("L9_DA"));
  }

  {
    auto const res =
        search("/api/experimental/map/route-search?text=L9&numResults=2");

    EXPECT_EQ(res.routes_.size(), 2U);
  }

  {
    auto const limited =
        config{.limits_ = config::limits{.route_search_max_results_ = 1}};
    auto const capped = ep::route_search{limited, *d.tt_, d.shapes_.get()};

    auto const res =
        capped("/api/experimental/map/route-search?text=L9&numResults=5");

    EXPECT_EQ(res.routes_.size(), 1U);
    EXPECT_THAT(ids(res), ElementsAre("L9_DA"));
  }

  {
    auto const no_shapes = ep::route_search{d.config_, *d.tt_, nullptr};

    auto const res = no_shapes(
        "/api/experimental/map/route-search?text=L9&place=49.87,8.63"
        "&placeBias=5");

    EXPECT_THAT(ids(res), ElementsAre("L9_DA", "L91", "N9", "L9_HH"));
  }

  {
    EXPECT_THROW(search("/api/experimental/map/route-search?text="),
                 net::bad_request_exception);
    EXPECT_THROW(
        search("/api/experimental/map/route-search?text=L9&numResults=0"),
        net::bad_request_exception);
    EXPECT_THROW(
        search("/api/experimental/map/route-search?text=L9&place=nonsense"),
        net::bad_request_exception);
  }
}
