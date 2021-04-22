#pragma once

#include <functional>
#include <string_view>
#include <vector>

#include "motis/module/message.h"
#include "motis/ris/ris_message.h"

namespace motis::ris::ribasis {

struct ribasis_parser {
  static void to_ris_message(std::string_view,
                             std::function<void(ris_message&&)> const&);
  static std::vector<ris_message> parse(std::string_view);

  ribasis_parser() = default;
  ~ribasis_parser() = default;

  ribasis_parser(ribasis_parser const&) = delete;
  ribasis_parser& operator=(ribasis_parser const&) = delete;

  ribasis_parser(ribasis_parser&&) = delete;
  ribasis_parser& operator=(ribasis_parser&&) = delete;
};

}  // namespace motis::ris::ribasis
