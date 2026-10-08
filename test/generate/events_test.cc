#include "gtest/gtest.h"

#include <set>
#include <string>
#include <vector>

#include "boost/program_options.hpp"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

#include "generate/events.h"

namespace n = nigiri;
namespace po = boost::program_options;
using namespace motis;
using namespace date;

namespace {

// T1 serves A -> B -> C on March 4 and 5, T2 serves A -> B on March 4 only,
// D is not served:
//   departures: A 2 + 1 = 3, B 2, C 0 (last stop), D 0
//   arrivals:   A 0 (first stop), B 2 + 1 = 3, C 2, D 0
n::loader::mem_dir tt_files() {
  return n::loader::mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
LINZ,Linz AG,https://linzag.at,Europe/Vienna

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,48.30000,14.28000
B,B,48.30000,14.32000
C,C,48.33000,14.28000
D,D,48.36000,14.28000

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,LINZ,R,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R,TWO_DAYS,T1,,
R,ONE_DAY,T2,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,10:00:00,10:00:00,A,0
T1,10:10:00,10:10:00,B,1
T1,10:20:00,10:20:00,C,2
T2,11:00:00,11:00:00,A,0
T2,11:10:00,11:10:00,B,1

# calendar_dates.txt
service_id,date,exception_type
TWO_DAYS,20240304,1
TWO_DAYS,20240305,1
ONE_DAY,20240304,1
)");
}

constexpr auto kDraws = 5000U;

struct event_sampler_test : public ::testing::Test {
  void SetUp() override {
    tt_.date_range_ = {date::sys_days{2024_y / March / 4},
                       date::sys_days{2024_y / March / 8}};
    n::loader::register_special_stations(tt_);
    n::loader::gtfs::load_timetable({.default_tz_ = "Europe/Vienna"},
                                    n::source_idx_t{0}, tt_files(), tt_);
    n::loader::finalize(tt_);
  }

  n::location_idx_t loc(std::string_view const id) const {
    return tt_.locations_.location_id_to_idx_.at(
        {.id_ = id, .src_ = n::source_idx_t{0}});
  }

  // every location, including special stations without routes, like the
  // stops generate passes without bounds
  std::vector<n::location_idx_t> all_locations() const {
    auto v = std::vector<n::location_idx_t>{};
    for (auto i = 0U; i != tt_.n_locations(); ++i) {
      v.emplace_back(n::location_idx_t{i});
    }
    return v;
  }

  event_sampler counted(date::sys_days const first_day,
                        date::sys_days const last_day) const {
    auto s = event_sampler{};
    s.enabled_ = true;
    s.count_events(tt_, all_locations(), first_day, last_day);
    return s;
  }

  double dep(event_sampler const& s, std::string_view const id) const {
    return s.dep_weight_[to_idx(loc(id))];
  }

  double arr(event_sampler const& s, std::string_view const id) const {
    return s.arr_weight_[to_idx(loc(id))];
  }

  n::timetable tt_;
};

void parse(event_sampler& s, std::vector<std::string> const& args) {
  auto desc = po::options_description{};
  s.add_options(desc);
  auto vm = po::variables_map{};
  po::store(po::command_line_parser(args).options(desc).run(), vm);
  po::notify(vm);
}

}  // namespace

TEST_F(event_sampler_test, count_events) {
  auto const s =
      counted(sys_days{2024_y / March / 4}, sys_days{2024_y / March / 6});

  EXPECT_EQ(3.0, dep(s, "A"));
  EXPECT_EQ(2.0, dep(s, "B"));
  EXPECT_EQ(0.0, dep(s, "C"));
  EXPECT_EQ(0.0, dep(s, "D"));

  EXPECT_EQ(0.0, arr(s, "A"));
  EXPECT_EQ(3.0, arr(s, "B"));
  EXPECT_EQ(2.0, arr(s, "C"));
  EXPECT_EQ(0.0, arr(s, "D"));

  ASSERT_EQ(tt_.n_locations(), s.dep_cum_.size());
  EXPECT_EQ(5.0, s.dep_cum_.back());
}

TEST_F(event_sampler_test, count_events_within_day_range) {
  // only T1 runs on March 5
  auto const s =
      counted(sys_days{2024_y / March / 5}, sys_days{2024_y / March / 6});

  EXPECT_EQ(1.0, dep(s, "A"));
  EXPECT_EQ(1.0, dep(s, "B"));
  EXPECT_EQ(1.0, arr(s, "B"));
  EXPECT_EQ(1.0, arr(s, "C"));
}

TEST_F(event_sampler_test, count_events_disabled_is_noop) {
  auto s = event_sampler{};
  s.count_events(tt_, all_locations(), sys_days{2024_y / March / 4},
                 sys_days{2024_y / March / 6});
  EXPECT_TRUE(s.dep_weight_.empty());
  EXPECT_TRUE(s.arr_weight_.empty());
  EXPECT_TRUE(s.dep_cum_.empty());
}

TEST_F(event_sampler_test, random_from_weighted_by_departures) {
  auto const s =
      counted(sys_days{2024_y / March / 4}, sys_days{2024_y / March / 6});

  auto n_a = 0U;
  for (auto i = 0U; i != kDraws; ++i) {
    auto const l = s.random_from(tt_, all_locations());
    ASSERT_TRUE(l == loc("A") || l == loc("B"));
    n_a += l == loc("A") ? 1U : 0U;
  }
  // A has 3 of 5 departures
  EXPECT_NEAR(0.6, static_cast<double>(n_a) / kDraws, 0.05);
}

TEST_F(event_sampler_test, random_from_without_departures_is_uniform) {
  // no service from March 6 on
  auto const s =
      counted(sys_days{2024_y / March / 6}, sys_days{2024_y / March / 8});
  ASSERT_EQ(0.0, s.dep_cum_.back());

  auto drawn = std::set<n::location_idx_t>{};
  for (auto i = 0U; i != kDraws; ++i) {
    drawn.insert(s.random_from(tt_, all_locations()));
  }
  // any stop with a route
  EXPECT_EQ((std::set{loc("A"), loc("B"), loc("C")}), drawn);
}

TEST_F(event_sampler_test, random_to_weighted_by_arrivals_within_range) {
  auto const s =
      counted(sys_days{2024_y / March / 4}, sys_days{2024_y / March / 6});
  auto const stops = std::vector{loc("C"), loc("D"), loc("A"), loc("B")};

  auto n_b = 0U;
  for (auto i = 0U; i != kDraws; ++i) {
    // D and A have no arrivals
    EXPECT_EQ(loc("C"), s.random_to(stops, 0U, 2U));
    EXPECT_EQ(loc("B"), s.random_to(stops, 2U, 100U));  // hi is clamped

    auto const l = s.random_to(stops, 0U, stops.size());
    ASSERT_TRUE(l == loc("B") || l == loc("C"));
    n_b += l == loc("B") ? 1U : 0U;
  }
  // B has 3 of 5 arrivals
  EXPECT_NEAR(0.6, static_cast<double>(n_b) / kDraws, 0.05);
}

TEST_F(event_sampler_test, random_to_without_arrivals_is_uniform_in_range) {
  auto const s =
      counted(sys_days{2024_y / March / 4}, sys_days{2024_y / March / 6});
  auto const stops = std::vector{loc("C"), loc("D"), loc("A"), loc("B")};

  auto drawn = std::set<n::location_idx_t>{};
  for (auto i = 0U; i != kDraws; ++i) {
    drawn.insert(s.random_to(stops, 1U, 3U));
  }
  EXPECT_EQ((std::set{loc("D"), loc("A")}), drawn);
}

TEST(event_sampler, random_from_resolves_small_weights) {
  // with a total weight of 10^6, a draw with only 10^6 distinct positions
  // lands on whole numbers and never on a stop in [k + 0.25, k + 0.75)
  // between stops in [k, k + 0.25) and [k + 0.75, k + 1); the first 10^5
  // units follow this pattern, so these stops hold 5% of the weight
  constexpr auto kUnits = 100'000U;
  constexpr auto kTotal = 1'000'000.0;

  auto s = event_sampler{};
  auto acc = 0.0;
  auto const add_stop = [&](double const weight) {
    s.stops_.emplace_back(s.stops_.size());
    s.dep_cum_.push_back(acc += weight);
  };
  for (auto i = 0U; i != kUnits; ++i) {
    add_stop(0.25);
    add_stop(0.5);
    add_stop(0.25);
  }
  add_stop(kTotal - kUnits);
  ASSERT_EQ(kTotal, s.dep_cum_.back());

  auto const tt = n::timetable{};
  auto n_between = 0U;
  for (auto i = 0U; i != kDraws; ++i) {
    auto const l = to_idx(s.random_from(tt, s.stops_));
    n_between += l < 3U * kUnits && l % 3U == 1U ? 1U : 0U;
  }
  EXPECT_NEAR(0.05, static_cast<double>(n_between) / kDraws, 0.02);
}

TEST(event_sampler, verify) {
  auto s = event_sampler{};
  EXPECT_NO_THROW(s.verify(false, false));
  EXPECT_NO_THROW(s.verify(true, false));
  EXPECT_NO_THROW(s.verify(false, true));

  s.enabled_ = true;
  EXPECT_NO_THROW(s.verify(false, false));
  EXPECT_THROW(s.verify(true, false), std::runtime_error);
  EXPECT_THROW(s.verify(false, true), std::runtime_error);
}

TEST(event_sampler, options) {
  auto defaulted = event_sampler{};
  parse(defaulted, {});
  EXPECT_FALSE(defaulted.enabled_);

  auto enabled = event_sampler{};
  parse(enabled, {"--event_weighted", "1"});
  EXPECT_TRUE(enabled.enabled_);
}
