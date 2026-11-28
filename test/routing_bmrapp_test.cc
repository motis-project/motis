#include "gtest/gtest.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "fmt/format.h"

#include "utl/init_from.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"
#include "motis/import.h"

using namespace motis;

namespace {

// Every trip leaves at 10:00 (Europe/Berlin). One trade-off per criterion:
//   W0 -> W9  bus + 10 min walk W1 -> W2 + bus, 11:00 | bus + bus at W3, 11:10
//   S0 -> S9  rail + bus at S1, 11:00 | bus + bus at S2, 11:10
//   M0 -> M9  air 10:30 | coach 10:40 | rail 10:50
// BMRAPP keeps a journey if it arrives within its anchor's travel time T plus
// clamp(T / 2, 20 min, 180 min) (relax_arr(), bmrap_bounds.h): for air's 30
// minutes 50, so all of these. W1 and W2 are 220 m apart, the other stops
// kilometres: the only footpath is W1 -> W2.
constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
D,20190501,1

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
W0,W0,50.00,8.00,,
W1,W1,50.02,8.00,,
W2,W2,50.022,8.00,,
W3,W3,50.06,8.00,,
W9,W9,50.08,8.00,,
S0,S0,50.00,8.20,,
S1,S1,50.02,8.20,,
S2,S2,50.04,8.20,,
S9,S9,50.06,8.20,,
M0,M0,50.00,8.40,,
M9,M9,50.02,8.40,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RAIL,MTA,RAIL,,,2
BUS,MTA,BUS,,,3
AIR,MTA,AIR,,,1100
COACH,MTA,COACH,,,200

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
BUS,D,W_FAST1,,
BUS,D,W_FAST2,,
BUS,D,W_DRY1,,
BUS,D,W_DRY2,,
RAIL,D,S_MIX1,,
BUS,D,S_MIX2,,
BUS,D,S_BUS1,,
BUS,D,S_BUS2,,
AIR,D,M_AIR,,
COACH,D,M_COACH,,
RAIL,D,M_RAIL,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
W_FAST1,10:00:00,10:00:00,W0,0,0,0
W_FAST1,10:20:00,10:20:00,W1,1,0,0
W_FAST2,10:35:00,10:35:00,W2,0,0,0
W_FAST2,11:00:00,11:00:00,W9,1,0,0
W_DRY1,10:00:00,10:00:00,W0,0,0,0
W_DRY1,10:20:00,10:20:00,W3,1,0,0
W_DRY2,10:35:00,10:35:00,W3,0,0,0
W_DRY2,11:10:00,11:10:00,W9,1,0,0
S_MIX1,10:00:00,10:00:00,S0,0,0,0
S_MIX1,10:20:00,10:20:00,S1,1,0,0
S_MIX2,10:35:00,10:35:00,S1,0,0,0
S_MIX2,11:00:00,11:00:00,S9,1,0,0
S_BUS1,10:00:00,10:00:00,S0,0,0,0
S_BUS1,10:20:00,10:20:00,S2,1,0,0
S_BUS2,10:35:00,10:35:00,S2,0,0,0
S_BUS2,11:10:00,11:10:00,S9,1,0,0
M_AIR,10:00:00,10:00:00,M0,0,0,0
M_AIR,10:30:00,10:30:00,M9,1,0,0
M_COACH,10:00:00,10:00:00,M0,0,0,0
M_COACH,10:40:00,10:40:00,M9,1,0,0
M_RAIL,10:00:00,10:00:00,M0,0,0,0
M_RAIL,10:50:00,10:50:00,M9,1,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
W1,W2,2,600
)";

data& bmrapp_data() {
  static auto const c = config{.timetable_ = config::timetable{
                                   .first_day_ = "2019-05-01",
                                   .num_days_ = 2,
                                   .datasets_ = {{"test", {.path_ = kGTFS}}}}};
  static auto d = [&]() {
    auto const path = std::filesystem::path{"test/data_bmrapp"};
    auto ec = std::error_code{};
    std::filesystem::remove_all(path, ec);
    import(c, path);
    return data{path, c};
  }();
  return d;
}

template <typename T>
std::string value(std::optional<T> const& v) {
  return v.has_value() ? fmt::format("{}", *v) : "-";
}

// "<minutes> t<transfers>" and the criteria, if the response has them
std::vector<std::string> rows(api::plan_response const& res) {
  auto v = std::vector<std::string>{};
  for (auto const& it : res.itineraries_) {
    auto const& c = it.criteria_;
    v.push_back(fmt::format(
        "{} t{}{}", it.duration_ / 60, it.transfers_,
        c.has_value()
            ? fmt::format(" nt={} ms={} av={}", value(c->nonTransit_),
                          value(c->modeSwitches_), value(c->avoidedModes_))
            : ""));
  }
  std::sort(begin(v), end(v));
  return v;
}

std::uint64_t algorithm(api::plan_response const& res) {
  return res.debugOutput_.at("algorithm");
}

constexpr auto const kBmrapp =
    static_cast<std::uint64_t>(api::algorithmEnum::BMRAPP);

// Departing from 09:55 or arriving until 11:45: every journey of the fixture
// is in the window.
struct plan {
  api::plan_response operator()(std::string_view const from,
                                std::string_view const to,
                                std::string_view const params) const {
    return routing_(fmt::format(
        "?fromPlace=test_{}&toPlace=test_{}&time=2019-05-01T{}&arriveBy={}{}",
        from, to, arrive_by_ ? "09:45Z" : "07:55Z", arrive_by_, params));
  }

  ep::routing const& routing_;
  bool arrive_by_;
};

}  // namespace

TEST(motis, bmrapp_without_criteria_matches_pong) {
  auto const routing = utl::init_from<ep::routing>(bmrapp_data()).value();
  for (auto const arrive_by : {false, true}) {
    auto const p = plan{routing, arrive_by};
    auto const pong = p("W0", "W9", "&algorithm=PONG");
    EXPECT_EQ((std::vector<std::string>{"60 t1"}), rows(pong)) << arrive_by;

    // the criteria are there, each one unset
    auto const bmrapp = p("W0", "W9", "&algorithm=BMRAPP");
    EXPECT_EQ(kBmrapp, algorithm(bmrapp)) << arrive_by;
    EXPECT_EQ((std::vector<std::string>{"60 t1 nt=- ms=- av=-"}), rows(bmrapp))
        << arrive_by;
  }
}

// A minimize* parameter runs BMRAPP instead of the default PONG and reports
// its own criterion only.
TEST(motis, bmrapp_minimize_non_transit) {
  auto const routing = utl::init_from<ep::routing>(bmrapp_data()).value();
  for (auto const arrive_by : {false, true}) {
    auto const res =
        plan{routing, arrive_by}("W0", "W9", "&minimizeNonTransit=true");
    EXPECT_EQ(kBmrapp, algorithm(res)) << arrive_by;
    EXPECT_EQ((std::vector<std::string>{"60 t1 nt=10 ms=- av=-",
                                        "70 t1 nt=0 ms=- av=-"}),
              rows(res))
        << arrive_by;
  }
}

TEST(motis, bmrapp_minimize_mode_switches) {
  auto const routing = utl::init_from<ep::routing>(bmrapp_data()).value();
  for (auto const arrive_by : {false, true}) {
    auto const res =
        plan{routing, arrive_by}("S0", "S9", "&minimizeModeSwitches=true");
    EXPECT_EQ(kBmrapp, algorithm(res)) << arrive_by;
    EXPECT_EQ((std::vector<std::string>{"60 t1 nt=- ms=1 av=-",
                                        "70 t1 nt=- ms=0 av=-"}),
              rows(res))
        << arrive_by;
  }
}

// Each avoided set of transit modes keeps the fastest journey and the best one
// without any mode of the set; several modes are one criterion. Street modes
// avoid nothing: no criterion, PONG answers.
TEST(motis, bmrapp_minimize_without) {
  auto const routing = utl::init_from<ep::routing>(bmrapp_data()).value();
  for (auto const arrive_by : {false, true}) {
    auto const p = plan{routing, arrive_by};
    EXPECT_EQ((std::vector<std::string>{"30 t0 nt=- ms=- av=true",
                                        "40 t0 nt=- ms=- av=false"}),
              rows(p("M0", "M9", "&minimizeWithout=AIRPLANE")))
        << arrive_by;
    EXPECT_EQ((std::vector<std::string>{"30 t0 nt=- ms=- av=false"}),
              rows(p("M0", "M9", "&minimizeWithout=COACH")))
        << arrive_by;
    EXPECT_EQ((std::vector<std::string>{"30 t0 nt=- ms=- av=true",
                                        "50 t0 nt=- ms=- av=false"}),
              rows(p("M0", "M9", "&minimizeWithout=AIRPLANE,COACH")))
        << arrive_by;
    EXPECT_EQ((std::vector<std::string>{"30 t0 nt=- ms=- av=true",
                                        "40 t0 nt=- ms=- av=false"}),
              rows(p("M0", "M9", "&minimizeWithout=AIRPLANE,RAIL")))
        << arrive_by;
    EXPECT_EQ((std::vector<std::string>{"60 t1 nt=- ms=- av=true"}),
              rows(p("S0", "S9", "&minimizeWithout=BUS")))
        << arrive_by;

    auto const walk = p("M0", "M9", "&minimizeWithout=WALK");
    EXPECT_EQ(static_cast<std::uint64_t>(api::algorithmEnum::PONG),
              algorithm(walk))
        << arrive_by;
    EXPECT_EQ((std::vector<std::string>{"30 t0"}), rows(walk)) << arrive_by;
  }
}

TEST(motis, bmrapp_criteria_combine) {
  auto const routing = utl::init_from<ep::routing>(bmrapp_data()).value();
  auto const res =
      plan{routing, false}("W0", "W9",
                           "&minimizeNonTransit=true&minimizeModeSwitches=true"
                           "&minimizeWithout=AIRPLANE");
  EXPECT_EQ(kBmrapp, algorithm(res));
  EXPECT_EQ((std::vector<std::string>{"60 t1 nt=10 ms=0 av=false",
                                      "70 t1 nt=0 ms=0 av=false"}),
            rows(res));
}

// Via stops are beyond the multicriteria engine: the requested algorithm
// answers (PONG by default and for BMRAPP), without criteria.
TEST(motis, bmrapp_falls_back_to_requested_algorithm) {
  auto const routing = utl::init_from<ep::routing>(bmrapp_data()).value();
  for (auto const& [params, expected] :
       {std::pair{"&minimizeNonTransit=true", api::algorithmEnum::PONG},
        std::pair{"&algorithm=BMRAPP", api::algorithmEnum::PONG},
        std::pair{"&algorithm=RAPTOR&minimizeNonTransit=true",
                  api::algorithmEnum::RAPTOR}}) {
    auto const res =
        plan{routing, false}("W0", "W9", fmt::format("{}&via=test_W3", params));
    EXPECT_EQ(static_cast<std::uint64_t>(expected), algorithm(res)) << params;
    EXPECT_EQ((std::vector<std::string>{"70 t1"}), rows(res)) << params;
  }
}
