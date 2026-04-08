#include "motis/config.h"

#include "../test_case.h"

using motis::config;

template <>
test_case_params const import_test_case<test_case::FFM_osm_only>() {
  auto const c =
      config{.server_ = {{.web_folder_ = "ui/build", .n_threads_ = 1U}},
             .osm_ = {"test/resources/test_case.osm.pbf"},
             .street_routing_ = true};
  return import_test_case(std::move(c), "test/test_case/ffm_osm_only_data");
}
