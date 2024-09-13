#include "gtest/gtest.h"

#include "motis/elevators/parse_fasta.h"

using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;
using namespace motis;
namespace n = nigiri;

constexpr auto const kFastaJson = R"__(
[
  {
    "description": "FFM HBF zu Gleis 101/102 (S-Bahn)",
    "equipmentnumber" : 10561326,
    "geocoordX" : 8.6628995,
    "geocoordY" : 50.1072933,
    "operatorname" : "DB InfraGO",
    "state" : "ACTIVE",
    "stateExplanation" : "available",
    "stationnumber" : 1866,
    "type" : "ELEVATOR",
    "outOfService": [["2024-07-18T12:00:00Z", "2024-07-19T12:00:00Z"]]
  },
  {
    "description": "FFM HBF zu Gleis 103/104 (S-Bahn)",
    "equipmentnumber": 10561327,
    "geocoordX": 8.6627516,
    "geocoordY": 50.1074549,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1866,
    "type": "ELEVATOR",
    "outOfService": [
      ["2024-07-18T11:00:00Z", "2024-07-18T14:00:00Z"],
      ["2024-07-19T11:00:00Z", "2024-07-19T14:00:00Z"]
    ]
  },
  {
    "description": "FFM HBF zu Gleis 103/104 (S-Bahn)",
    "equipmentnumber": 10561327,
    "geocoordX": 8.6627516,
    "geocoordY": 50.1074549,
    "operatorname": "DB InfraGO",
    "state": "INACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1866,
    "type": "ELEVATOR"
  }
]
)__"sv;

TEST(motis, parse_fasta) {
  auto const elevators = parse_fasta(kFastaJson);
  ASSERT_EQ(3, elevators.size());
  ASSERT_EQ(1, elevators[elevator_idx_t{0}].out_of_service_.size());
  ASSERT_EQ(2, elevators[elevator_idx_t{1}].out_of_service_.size());
  ASSERT_EQ(0, elevators[elevator_idx_t{2}].out_of_service_.size());
  EXPECT_EQ((n::interval<n::unixtime_t>{sys_days{2024_y / July / 18} + 12h,
                                        sys_days{2024_y / July / 19} + 12h}),
            elevators[elevator_idx_t{0}].out_of_service_[0]);
  EXPECT_EQ((n::interval<n::unixtime_t>{sys_days{2024_y / July / 18} + 11h,
                                        sys_days{2024_y / July / 18} + 14h}),
            elevators[elevator_idx_t{1}].out_of_service_[0]);
  EXPECT_EQ((n::interval<n::unixtime_t>{sys_days{2024_y / July / 19} + 11h,
                                        sys_days{2024_y / July / 19} + 14h}),
            elevators[elevator_idx_t{1}].out_of_service_[1]);
  EXPECT_EQ((std::vector<state_change<n::unixtime_t>>{
                {n::unixtime_t{n::unixtime_t::duration{0}}, true},
                {sys_days{2024_y / July / 18} + 12h, false},
                {sys_days{2024_y / July / 19} + 12h, true}}),
            elevators[elevator_idx_t{0}].state_changes_);
  EXPECT_EQ((std::vector<state_change<n::unixtime_t>>{
                {n::unixtime_t{n::unixtime_t::duration{0}}, true},
                {sys_days{2024_y / July / 18} + 11h, false},
                {sys_days{2024_y / July / 18} + 14h, true},
                {sys_days{2024_y / July / 19} + 11h, false},
                {sys_days{2024_y / July / 19} + 14h, true}}),
            elevators[elevator_idx_t{1}].state_changes_);

  auto const expected =
      std::array<std::pair<n::unixtime_t, std::vector<bool>>, 7>{
          std::pair<n::unixtime_t, std::vector<bool>>{
              n::unixtime_t{n::unixtime_t::duration{0}}, {true, true, false}},
          std::pair<n::unixtime_t, std::vector<bool>>{
              sys_days{2024_y / July / 18} + 11h, {true, false, false}},
          std::pair<n::unixtime_t, std::vector<bool>>{
              sys_days{2024_y / July / 18} + 12h, {false, false, false}},
          std::pair<n::unixtime_t, std::vector<bool>>{
              sys_days{2024_y / July / 18} + 14h, {false, true, false}},
          std::pair<n::unixtime_t, std::vector<bool>>{
              sys_days{2024_y / July / 19} + 11h, {false, false, false}},
          std::pair<n::unixtime_t, std::vector<bool>>{
              sys_days{2024_y / July / 19} + 12h, {true, false, false}},
          std::pair<n::unixtime_t, std::vector<bool>>{
              sys_days{2024_y / July / 19} + 14h, {true, true, false}}};
  auto const single_state_changes = utl::to_vec(
      elevators, [&](elevator const& e) { return e.get_state_changes(); });
  auto g = get_state_changes(single_state_changes);
  auto i = 0U;
  while (g) {
    auto const x = g();
    ASSERT_LT(i, expected.size());
    EXPECT_EQ(expected[i], x) << "i=" << i;
    ++i;
  }
  EXPECT_EQ(expected.size(), i);
}