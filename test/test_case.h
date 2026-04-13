#pragma once

/**
 * Framework for reusable tests
 *
 * Test cases need to be defined once and can then be used for multiple tests
 * See comments on how to add new test cases
 *
 * Reminder: Use with care! Try to reuse already existing tests first
 */

#include <string_view>

#include "motis/config.h"
#include "motis/data.h"

using motis::config;
using motis::data;

// Requires an element for each reusable test case
enum class test_case {
  FFM_one_to_many,
};

using test_case_params = std::pair<std::string_view, config>;

// Requires a specialisation for each test case
template <test_case TestCase>
test_case_params const import_test_case();

// Most tests will only use 'data', but some might require access to 'config'
template <test_case TestCase>
std::pair<data, config const&> get_test_case() {
  static auto const params{import_test_case<TestCase>()};
  return {data{std::get<0>(params), std::get<1>(params)}, std::get<1>(params)};
}

test_case_params const import_test_case(config const&&, std::string_view path);
