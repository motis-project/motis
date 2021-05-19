#pragma once

#include <bitset>
#include <map>

#include "utl/parser/cstr.h"

#include "motis/schedule/bitfield.h"
#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/loaded_file.h"

namespace motis::loader::hrd {

constexpr int ALL_DAYS_KEY = 0;

bitfield hex_str_to_bitset(utl::cstr hex, char const* filename,
                           int line_number);

std::map<int, bitfield> parse_bitfields(loaded_file const&, config const&);

}  // namespace motis::loader::hrd
