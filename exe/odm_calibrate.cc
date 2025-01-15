#include <fstream>
#include <iostream>
#include <vector>

#include "fmt/printf.h"

#include "utl/zip.h"

#include "nigiri/routing/journey.h"

#include "motis/odm/calibration/json.h"
#include "motis/odm/equal_journeys.h"
#include "motis/odm/mixer.h"

using namespace std::string_view_literals;

namespace motis {

namespace n = nigiri;
using namespace motis::odm;

template <typename T>
struct search_space {
  T start_;
  T end_;
  T step_;
};

auto get_expected(auto const& reqs) {
  auto expected = std::vector<std::vector<n::routing::journey>>{};
  for (auto const& r : reqs) {
    expected.emplace_back(r.get_expected());
  }
  return expected;
}

bool operator==(std::vector<n::routing::journey> const& expected,
                std::vector<n::routing::journey> const& actual) {
  return expected.size() == actual.size() &&
         std::all_of(begin(expected), end(expected), [&](auto const& e) {
           return std::find_if(begin(actual), end(actual), [&](auto const& a) {
                    return a == e;
                  }) != end(actual);
         });
}

void search(std::vector<requirement> const& reqs,
            std::vector<std::vector<n::routing::journey>> const& expected,
            search_space<std::int32_t> const& walk_cost_0,
            search_space<std::int32_t> const& walk_cost_1,
            search_space<std::int32_t> const& taxi_cost_0,
            search_space<std::int32_t> const& taxi_cost_1,
            search_space<std::int32_t> const& transfer_cost,
            search_space<double> const& direct_taxi_factor,
            search_space<double> const& direct_taxi_constant,
            search_space<double> const& travel_time_weight,
            search_space<double> const& distance_weight,
            search_space<double> const& distance_exponent) {
  auto m = motis::odm::mixer{.walk_cost_ = {{0, 1}, {15, 11}},
                             .taxi_cost_ = {{0, 59}, {1, 13}},
                             .transfer_cost_ = {{0, 15}},
                             .direct_taxi_factor_ = 1.3,
                             .direct_taxi_constant_ = 27,
                             .travel_time_weight_ = 1.5,
                             .distance_weight_ = 0.07,
                             .distance_exponent_ = 1.5};
  auto odm_journeys = std::vector<n::routing::journey>{};
  for (m.walk_cost_[0].cost_ = walk_cost_0.start_;
       m.walk_cost_[0].cost_ < walk_cost_0.end_;
       m.walk_cost_[0].cost_ += walk_cost_0.step_) {
    for (m.walk_cost_[1].cost_ = walk_cost_1.start_;
         m.walk_cost_[1].cost_ < walk_cost_1.end_;
         m.walk_cost_[1].cost_ += walk_cost_1.step_) {
      for (m.taxi_cost_[0].cost_ = taxi_cost_0.start_;
           m.taxi_cost_[0].cost_ < taxi_cost_0.end_;
           m.taxi_cost_[0].cost_ += taxi_cost_0.step_) {
        for (m.taxi_cost_[1].cost_ = taxi_cost_1.start_;
             m.taxi_cost_[1].cost_ < taxi_cost_1.end_;
             m.taxi_cost_[1].cost_ += taxi_cost_1.step_) {
          for (m.transfer_cost_[0].cost_ = transfer_cost.start_;
               m.transfer_cost_[0].cost_ < transfer_cost.end_;
               m.transfer_cost_[0].cost_ += transfer_cost.step_) {
            for (m.direct_taxi_factor_ = direct_taxi_factor.start_;
                 m.direct_taxi_factor_ < direct_taxi_factor.end_;
                 m.direct_taxi_factor_ += direct_taxi_factor.step_) {
              for (m.direct_taxi_constant_ = direct_taxi_constant.start_;
                   m.direct_taxi_constant_ < direct_taxi_constant.end_;
                   m.direct_taxi_constant_ += direct_taxi_constant.step_) {
                for (m.travel_time_weight_ = travel_time_weight.start_;
                     m.travel_time_weight_ < travel_time_weight.end_;
                     m.travel_time_weight_ += travel_time_weight.step_) {
                  for (m.distance_weight_ = distance_weight.start_;
                       m.distance_weight_ < distance_weight.end_;
                       m.distance_weight_ += distance_weight.step_) {
                    for (m.distance_exponent_ = distance_exponent.start_;
                         m.distance_exponent_ < distance_exponent.end_;
                         m.distance_exponent_ += distance_exponent.step_) {
                      for (auto const [r, e] : utl::zip(reqs, expected)) {
                        odm_journeys = r.odm_;
                        m.mix(r.pt_, odm_journeys);
                        if (e == odm_journeys) {
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

int odm_calibrate(int ac, char** av) {
  if (ac > 1 && av[1] == "--help"sv) {
    fmt::println(
        "calibrate the odm journey domination model\n\n"
        "Usage:\n"
        "motis odm_calibrate requirements.json\n");
    return 0;
  }

  if (ac > 1) {
    auto file = std::ifstream{av[1]};
    auto const json_str = std::string{std::istreambuf_iterator<char>{file},
                                      std::istreambuf_iterator<char>{}};
    auto const reqs = motis::odm::read_requirements(json_str);
    auto const expected = get_expected(reqs);

    return 0;
  }

  return 1;
}

}  // namespace motis