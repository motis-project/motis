#pragma once

#include <iostream>
#include <regex>

namespace motis::footpaths {

// -- string matcher --

bool exact_str_match(std::string str_a, std::string_view strv_b) {
  // remove all special characters from str_a and strv_b
  str_a = std::regex_replace(str_a, std::regex("[^0-9a-zA-Z]+"), "");
  strv_b =
      std::regex_replace(std::string{strv_b}, std::regex("[^0-9a-zA-Z]+"), "");
  return str_a == strv_b;
}

bool exact_first_number_match(std::string str_a, std::string_view strv_b) {
  str_a = std::regex_replace(str_a, std::regex("[^0-9]+"), std::string("$1"));
  str_a = std::regex_replace(std::string{strv_b}, std::regex("[^0-9]+"),
                             std::string("$1"));

  if (strv_b.length() == 0) {
    return false;
  }

  return str_a == strv_b;
}

// -- location and platform matching

}  // namespace motis::footpaths
