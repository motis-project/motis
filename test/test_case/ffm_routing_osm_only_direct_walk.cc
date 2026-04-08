#include "motis/config.h"

#include "../test_case.h"

using motis::config;

template <>
test_case_params const
import_test_case<test_case::FFM_routing_osm_only_direct_walk>() {
  auto const c =
      config{.server_ = {{.web_folder_ = "ui/build", .n_threads_ = 1U}},
             .osm_ = {"test/resources/test_case.osm.pbf"},
             .street_routing_ = true};
  return import_test_case(
      std::move(c), "test/test_case/ffm_routing_osm_only_direct_walk_data");
}
