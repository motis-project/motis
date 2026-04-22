#include "./test_case.h"

#include "motis/import.h"

using motis::import;

test_case_params const import_test_case(config const&& c,
                                        std::string_view path) {
  auto ec = std::error_code{};
  std::filesystem::remove_all(path, ec);

  import(c, path);
  return {path, std::move(c)};
}
