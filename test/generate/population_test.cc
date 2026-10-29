#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "boost/program_options.hpp"

#include "fmt/core.h"

#include "geo/grid.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

#include "utl/raii.h"

#include "generate/population.h"

namespace n = nigiri;
namespace fs = std::filesystem;
namespace po = boost::program_options;
using namespace motis;
using namespace date;
using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

namespace {

constexpr auto kResolution = 1000;

// A, A2, B and C are served by a trip, D is not;
// A2 lies about 30 m from A, in the same 1 km cell
n::loader::mem_dir tt_files() {
  return n::loader::mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
LINZ,Linz AG,https://linzag.at,Europe/Vienna

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,48.30000,14.28000
A2,A2,48.30020,14.28020
B,B,48.30000,14.32000
C,C,48.33000,14.28000
D,D,48.36000,14.28000

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,LINZ,R,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R,S,T,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T,10:00:00,10:00:00,A,0
T,10:02:00,10:02:00,A2,1
T,10:10:00,10:10:00,B,2
T,10:20:00,10:20:00,C,3

# calendar_dates.txt
service_id,date,exception_type
S,20240304,1
)");
}

struct population_sampler_test : public ::testing::Test {
  void SetUp() override {
    tt_.date_range_ = {date::sys_days{2024_y / March / 4},
                       date::sys_days{2024_y / March / 5}};
    n::loader::register_special_stations(tt_);
    n::loader::gtfs::load_timetable({.default_tz_ = "Europe/Vienna"},
                                    n::source_idx_t{0}, tt_files(), tt_);
    n::loader::finalize(tt_);
  }

  n::location_idx_t loc(std::string_view const id) const {
    return tt_.locations_.location_id_to_idx_.at(
        {.id_ = id, .src_ = n::source_idx_t{0}});
  }

  geo::inspire_cell cell(std::string_view const id) const {
    return geo::inspire_cell_of(tt_.locations_.coordinates_[loc(id)],
                                kResolution);
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

  // index of the cell drawn for w and u
  static std::ptrdiff_t drawn(population_sampler const& s,
                              pop_weight const w,
                              double const u) {
    return &s.random_cell(w, u) - s.cell_stops_.data();
  }

  n::timetable tt_;
};

std::string grid_id(geo::inspire_cell const& c) {
  return fmt::format("CRS3035RES{}mN{}E{}", c.resolution_, c.northing_,
                     c.easting_);
}

void parse(population_sampler& s, std::vector<std::string> const& args) {
  auto desc = po::options_description{};
  s.add_options(desc);
  auto vm = po::variables_map{};
  po::store(po::command_line_parser(args).options(desc).run(), vm);
  po::notify(vm);
}

}  // namespace

TEST_F(population_sampler_test, match_stops) {
  ASSERT_EQ(cell("A"), cell("A2"));
  auto const no_stop = geo::inspire_cell_of({48.5, 14.5}, kResolution);

  auto s = population_sampler{};
  s.from_ = pop_weight::kHigh;
  s.grid_ = {{cell("A"), 10U}, {no_stop, 5U},    {cell("D"), 1000U},
             {cell("B"), 30U}, {cell("C"), 60U}, {cell("B"), 7U}};
  s.match_stops(tt_, all_locations());

  // dropped: the cell without a stop, D's cell (D has no route) and the
  // second listing of B's cell
  ASSERT_EQ(3U, s.grid_.size());
  EXPECT_EQ(cell("A"), s.grid_[0].id_);
  EXPECT_EQ(10U, s.grid_[0].data_);
  EXPECT_EQ(cell("B"), s.grid_[1].id_);
  EXPECT_EQ(30U, s.grid_[1].data_);
  EXPECT_EQ(cell("C"), s.grid_[2].id_);
  EXPECT_EQ(60U, s.grid_[2].data_);

  ASSERT_EQ(3U, s.cell_stops_.size());
  EXPECT_THAT(s.cell_stops_[0], UnorderedElementsAre(loc("A"), loc("A2")));
  EXPECT_THAT(s.cell_stops_[1], ElementsAre(loc("B")));
  EXPECT_THAT(s.cell_stops_[2], ElementsAre(loc("C")));

  // only the weighting in use is computed
  EXPECT_EQ((std::vector{10.0, 40.0, 100.0}), s.cumulative_weight_high_);
  EXPECT_TRUE(s.cumulative_weight_low_.empty());
}

TEST_F(population_sampler_test, match_stops_without_grid_is_noop) {
  auto s = population_sampler{};
  s.match_stops(tt_, all_locations());
  EXPECT_TRUE(s.grid_.empty());
  EXPECT_TRUE(s.cell_stops_.empty());
}

TEST_F(population_sampler_test, match_stops_rejects_mixed_resolutions) {
  auto s = population_sampler{};
  s.from_ = pop_weight::kHigh;
  s.grid_ = {
      {cell("A"), 10U},
      {geo::inspire_cell_of(tt_.locations_.coordinates_[loc("B")], 100), 30U}};
  EXPECT_THROW(s.match_stops(tt_, all_locations()), std::runtime_error);
}

TEST_F(population_sampler_test, match_stops_rejects_grid_without_stops) {
  auto s = population_sampler{};
  s.from_ = pop_weight::kHigh;
  s.grid_ = {{cell("D"), 1000U}};
  EXPECT_THROW(s.match_stops(tt_, all_locations()), std::runtime_error);
}

TEST_F(population_sampler_test, random_cell_high) {
  auto s = population_sampler{};
  s.from_ = pop_weight::kHigh;
  s.grid_ = {{cell("A"), 10U}, {cell("B"), 30U}, {cell("C"), 60U}};
  s.match_stops(tt_, all_locations());

  // running totals 10, 40, 100: u * 100 picks the first total above it
  EXPECT_EQ(0, drawn(s, pop_weight::kHigh, 0.0));
  EXPECT_EQ(0, drawn(s, pop_weight::kHigh, 0.05));
  EXPECT_EQ(1, drawn(s, pop_weight::kHigh, 0.15));
  EXPECT_EQ(1, drawn(s, pop_weight::kHigh, 0.35));
  EXPECT_EQ(2, drawn(s, pop_weight::kHigh, 0.45));
  EXPECT_EQ(2, drawn(s, pop_weight::kHigh, 0.999));
}

TEST_F(population_sampler_test, random_cell_low_mirrors_ranks) {
  auto s = population_sampler{};
  s.to_ = pop_weight::kLow;
  s.grid_ = {{cell("A"), 10U}, {cell("B"), 30U}, {cell("C"), 60U}};
  s.match_stops(tt_, all_locations());

  // the least populated cell gets the largest population as weight
  EXPECT_TRUE(s.cumulative_weight_high_.empty());
  EXPECT_EQ((std::vector{60.0, 90.0, 100.0}), s.cumulative_weight_low_);

  EXPECT_EQ(0, drawn(s, pop_weight::kLow, 0.3));
  EXPECT_EQ(1, drawn(s, pop_weight::kLow, 0.75));
  EXPECT_EQ(2, drawn(s, pop_weight::kLow, 0.95));
}

TEST_F(population_sampler_test, random_cell_low_ties_share_weight) {
  auto s = population_sampler{};
  s.from_ = pop_weight::kHigh;
  s.to_ = pop_weight::kLow;
  s.grid_ = {{cell("A"), 10U}, {cell("B"), 10U}, {cell("C"), 40U}};
  s.match_stops(tt_, all_locations());

  // A and B tie at rank 0/1 and share the mirrored weights 40 and 10
  EXPECT_EQ((std::vector{10.0, 20.0, 60.0}), s.cumulative_weight_high_);
  EXPECT_EQ((std::vector{25.0, 50.0, 60.0}), s.cumulative_weight_low_);
}

TEST(population_sampler, verify) {
  auto const with = [](std::optional<pop_weight> const from,
                       std::optional<pop_weight> const to, bool const grid) {
    auto s = population_sampler{};
    s.from_ = from;
    s.to_ = to;
    if (grid) {
      s.grid_ = {{geo::inspire_cell_of({48.3, 14.28}, kResolution), 1U}};
    }
    return s;
  };
  auto const high = std::optional{pop_weight::kHigh};
  auto const none = std::optional<pop_weight>{};

  EXPECT_NO_THROW(with(none, none, false).verify(false, false, false));
  EXPECT_NO_THROW(with(high, none, true).verify(false, true, true));
  EXPECT_NO_THROW(with(none, high, true).verify(true, false, false));

  EXPECT_THROW(with(high, none, false).verify(false, false, false),
               std::runtime_error);
  EXPECT_THROW(with(none, none, true).verify(false, false, false),
               std::runtime_error);
  EXPECT_THROW(with(high, none, true).verify(true, false, false),
               std::runtime_error);
  EXPECT_THROW(with(none, high, true).verify(false, true, false),
               std::runtime_error);
  EXPECT_THROW(with(none, high, true).verify(false, false, true),
               std::runtime_error);
}

TEST(population_sampler, options) {
  auto const populated = geo::inspire_cell_of({48.3, 14.28}, kResolution);
  auto const empty = geo::inspire_cell_of({48.5, 14.5}, kResolution);

  auto const path = fs::temp_directory_path() / "motis_population_test.csv";
  auto const remove_file = utl::make_finally([&]() { fs::remove(path); });
  auto const write_grid = [&](std::string const& rows) {
    std::ofstream{path} << "GRD_ID,T\n" << rows;
  };

  write_grid(fmt::format("{},10\n{},0\n", grid_id(populated), grid_id(empty)));
  auto s = population_sampler{};
  parse(s, {"--population_grid", path.string(), "--population_from", "high",
            "--population_to", "low"});
  EXPECT_EQ(pop_weight::kHigh, s.from_);
  EXPECT_EQ(pop_weight::kLow, s.to_);
  ASSERT_EQ(1U, s.grid_.size());  // the unpopulated cell is dropped
  EXPECT_EQ(populated, s.grid_[0].id_);
  EXPECT_EQ(10U, s.grid_[0].data_);

  auto invalid_weight = population_sampler{};
  EXPECT_THROW(parse(invalid_weight, {"--population_from", "mid"}),
               std::runtime_error);

  write_grid(fmt::format("{},0\n", grid_id(empty)));
  auto unpopulated = population_sampler{};
  EXPECT_THROW(parse(unpopulated, {"--population_grid", path.string()}),
               std::runtime_error);

  fs::remove(path);
  auto missing = population_sampler{};
  EXPECT_THROW(parse(missing, {"--population_grid", path.string()}),
               std::runtime_error);
}
