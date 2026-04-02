#include "./test_case.h"

#include "motis/import.h"

using motis::import;

data import_test_case(config const& c, std::string_view path) {
  auto ec = std::error_code{};
  std::filesystem::remove_all(path, ec);

  return import(c, path, true);
}
