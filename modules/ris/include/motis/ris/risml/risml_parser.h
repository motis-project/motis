#pragma once

#include <functional>
#include <string_view>
#include <vector>

#include "motis/module/message.h"
#include "motis/ris/ris_message.h"

namespace motis::ris::risml {

struct risml_parser {

  void to_ris_message(std::string_view,
                      std::function<void(ris_message&&)> const&);
  std::vector<ris_message> parse(std::string_view);

  risml_parser() = default;
  ~risml_parser() = default;

  risml_parser(risml_parser const&) = delete;
  risml_parser& operator=(risml_parser const&) = delete;

  risml_parser(risml_parser&&) = delete;
  risml_parser& operator=(risml_parser&&) = delete;
};

}  // namespace motis::ris::risml
