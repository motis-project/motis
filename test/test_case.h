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

template <test_case TestCase>
data import_test_case();

template <test_case TestCase>
data& get_test_case() {
  static auto data{import_test_case<TestCase>()};
  return data;
}

data import_test_case(config const&, std::string_view path);
