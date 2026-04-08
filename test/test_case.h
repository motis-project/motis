#pragma once

#include <string_view>

#include "motis/config.h"
#include "motis/data.h"

using motis::config;
using motis::data;

enum class test_case {
  // TODO Rename based on test
  FFM_default_with_geocoding,
  FFM_default_no_map,  // default_no_osm
  FFM_matching_test,
  FFM_osm_only,
  FFM_with_elevators_siri,  // ??
  FFM_with_gbfs,
  FFM_simple_transfers,  // ??
  FFM_for_first_last_mile,
  FFM_tag_lookup,
  CH_cable_car_netex,
  generated_minimal_bw,  // ??
  generated_hs,  // ??
  generated_stop_group_geocoding,
};

using test_case_params = std::pair<std::string_view, config>;

template <test_case TestCase>
test_case_params const import_test_case();

template <test_case TestCase>
data get_test_case() {
  static auto const params{import_test_case<TestCase>()};
  return data{std::get<0>(params), std::get<1>(params)};
}

template <test_case TestCase>
std::pair<data, config const&> get_test_case2() {
  static auto const params{import_test_case<TestCase>()};
  return {data{std::get<0>(params), std::get<1>(params)}, std::get<1>(params)};
}

test_case_params const import_test_case(config const&&, std::string_view path);
