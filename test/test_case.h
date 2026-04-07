#pragma once

#include <string_view>

#include "motis/config.h"
#include "motis/data.h"

using motis::config;
using motis::data;

enum class test_case {
  FFM_simple_transfers,
  FFM_for_first_last_mile,
};

using test_case_params = std::pair<std::string_view, config>;

template <test_case TestCase>
test_case_params import_test_case();

template <test_case TestCase>
data get_test_case() {
  static auto params{import_test_case<TestCase>()};
  return data{std::get<0>(params), std::get<1>(params)};
}

test_case_params import_test_case(config&&, std::string_view path);
