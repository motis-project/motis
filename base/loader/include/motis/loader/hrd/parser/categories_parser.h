#pragma once

#include <cinttypes>
#include <map>
#include <string>

#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/loaded_file.h"
#include "motis/schedule-format/Category_generated.h"

namespace motis::loader::hrd {

struct category {
  category() = default;
  category(std::string name, uint8_t output_rule)
      : name_(std::move(name)), output_rule_(output_rule) {}

  std::string name_;
  uint8_t output_rule_{0};
};

std::map<uint32_t, category> parse_categories(loaded_file const&,
                                              config const&);

}  // namespace motis::loader::hrd
