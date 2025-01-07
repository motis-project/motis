#include <fstream>
#include <iostream>
#include <vector>

#include "fmt/printf.h"

#include "nigiri/routing/journey.h"

#include "motis/odm/calibration/json.h"

using namespace std::string_view_literals;

namespace motis {

namespace n = nigiri;

auto get_expected(auto const& reqs) {
  auto expected = std::vector<std::vector<n::routing::journey>>{};
  for (auto const& r : reqs) {
    expected.emplace_back(r.get_expected());
  }
  return expected;
}

bool fulfills(std::vector<n::routing::journey> const& expected,
              std::vector<n::routing::journey> const& actual) {}

int odm_calibrate(int ac, char** av) {
  if (ac > 1 && av[1] == "--help"sv) {
    fmt::println(
        "calibrate the odm journey domination model by providing connection "
        "sets\n\n"
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