#include "gtest/gtest.h"

#include <string>

#include "geo/latlng.h"

#include "motis/footpaths/matching.h"

TEST(footpaths, remove_special_characters) {
  using namespace motis::footpaths;

  std::map<std::string, std::string> testcases{};
  testcases.emplace(std::pair<std::string, std::string>{"a-A-1", "aA1"});
  testcases.emplace(std::pair<std::string, std::string>{
      "abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxyz"});
  testcases.emplace(std::pair<std::string, std::string>{
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"});
  testcases.emplace(
      std::pair<std::string, std::string>{"0123456789", "0123456789"});
  testcases.emplace((std::pair<std::string, std::string>){",.-<>#'+-*/~", ""});

  for (auto const& [input, result] : testcases) {
    ASSERT_EQ(result, remove_special_characters(input));
  }
}

TEST(footpaths, get_first_number_sequence) {
  using namespace motis::footpaths;

  std::map<std::string, std::string> testcases;
  testcases.emplace(std::pair<std::string, std::string>{"a-A-1", "1"});
  testcases.emplace(
      std::pair<std::string, std::string>{"abcdefghijklmnopqrstuvwxyz", ""});
  testcases.emplace(
      std::pair<std::string, std::string>{"ABCDEFGHIJKLMNOPQRSTUVWXYZ", ""});
  testcases.emplace(
      std::pair<std::string, std::string>{"0123456789", "0123456789"});
  testcases.emplace(
      std::pair<std::string, std::string>{",.aA-<>012#'+-*/~", "012"});
  testcases.emplace(std::pair<std::string, std::string>{"12-34", "12"});
  testcases.emplace(std::pair<std::string, std::string>{"-1-2-3-", "1"});
  testcases.emplace(std::pair<std::string, std::string>{"ab.ba-2", "2"});

  for (auto const& [input, result] : testcases) {
    ASSERT_EQ(result, get_first_number_sequence(input));
  }
}

TEST(footpaths, exact_str_match) {
  using namespace motis::footpaths;
  std::vector<std::pair<std::string, std::string>> testcases_false;
  testcases_false.emplace_back("", "");
  testcases_false.emplace_back(".", "a");
  testcases_false.emplace_back("a", ".");

  std::vector<std::pair<std::string, std::string>> testcases_true;
  testcases_true.emplace_back("aA", "a-A");
  testcases_true.emplace_back("aA", "a-a");
  testcases_true.emplace_back("!1-a.A", "1aA");
  testcases_true.emplace_back("!1-a.A", "1Aa");

  for (auto& [str_a, str_b] : testcases_false) {
    ASSERT_FALSE(exact_str_match(str_a, str_b));
  }
  for (auto& [str_a, str_b] : testcases_true) {
    ASSERT_TRUE(exact_str_match(str_a, str_b));
  }
}

TEST(footpaths, exact_first_number_match) {
  using namespace motis::footpaths;

  std::vector<std::pair<std::string, std::string>> testcases_false;
  testcases_false.emplace_back("", "");
  testcases_false.emplace_back(".", "1");
  testcases_false.emplace_back("1", ".");
  testcases_false.emplace_back("aA", "a-A");
  testcases_false.emplace_back("A12b-13", "13");

  std::vector<std::pair<std::string, std::string>> testcases_true;
  testcases_true.emplace_back("1", "1");
  testcases_true.emplace_back("A-1--4", "1-.4");

  for (auto& [str_a, str_b] : testcases_false) {
    ASSERT_FALSE(exact_first_number_match(str_a, str_b));
  }
  for (auto& [str_a, str_b] : testcases_true) {
    ASSERT_TRUE(exact_first_number_match(str_a, str_b));
  }
}
