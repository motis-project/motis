#pragma once

#include <string_view>

#include "motis/config.h"
#include "motis/data.h"

using motis::config;
using motis::data;

enum class test_case {
  // TODO Rename based on test
  FFM_ojp_requests,  // 1
  FFM_matching_test,
  FFM_routing_osm_only_direct_walk,  // 1
  FFM_siri_fm_routing,  // 1
  FFM_stop_times,  // 1
  FFM_routing,  // 1
  FFM_map_routes,  // 1
  FFM_one_to_many,  // 1
  FFM_tag_lookup,  // 1
  FFM_get_way_candidates,  // 1
  CH_trip_notice_translations,  // 1
  generated_trip_siri_sx_alerts,  // 1
  generated_trip_stop_naming,  // 1
  generated_stop_group_geocoding,  // 1
};

using test_case_params = std::pair<std::string_view, config>;

template <test_case TestCase>
test_case_params const import_test_case();

template <test_case TestCase>
std::pair<data, config const&> get_test_case() {
  static auto const params{import_test_case<TestCase>()};
  return {data{std::get<0>(params), std::get<1>(params)}, std::get<1>(params)};
}

test_case_params const import_test_case(config const&&, std::string_view path);
