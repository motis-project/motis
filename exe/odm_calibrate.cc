#include <fstream>
#include <iostream>

#include "fmt/printf.h"

#include "motis/odm/calibration/json.h"

using namespace std::string_view_literals;

namespace motis {

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

    return 0;
  }

  return 1;
}

}  // namespace motis