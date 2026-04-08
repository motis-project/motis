#pragma once

#include <string_view>

#include "motis/config.h"
#include "motis/data.h"

using motis::config;
using motis::data;

enum class test_case {
  CH_trip_notice_translations,
  FFM_get_way_candidates,
  FFM_map_routes,
  FFM_ojp_requests,
  FFM_one_to_many,
  FFM_routing,
  FFM_routing_osm_only_direct_walk,
  FFM_siri_fm_routing,
  FFM_stop_times,
  FFM_tag_lookup,
  generated_stop_group_geocoding,
  generated_trip_siri_sx_alerts,
  generated_trip_stop_naming,
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
