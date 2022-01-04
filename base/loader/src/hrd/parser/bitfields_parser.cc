#include "motis/loader/hrd/parser/bitfields_parser.h"

#include <algorithm>
#include <string>

#include "flatbuffers/util.h"

#include "utl/parser/arg_parser.h"

#include "motis/core/common/logging.h"
#include "motis/loader/parser_error.h"
#include "motis/loader/util.h"

using namespace utl;
using namespace motis::logging;

namespace motis::loader::hrd {

std::string hex_to_string(char c) {
  char str[2] = {c, '\0'};
  auto i = flatbuffers64::StringToInt(str, 16);  // NOLINT
  std::bitset<4> bits(i);
  return bits.to_string();
}

template <typename T>
std::string hex_to_string(T const& char_collection) {
  std::string bit_str;
  for (auto const& c : char_collection) {
    bit_str.append(hex_to_string(c));
  }
  return bit_str;
}

bitfield hex_str_to_bitset(cstr hex, char const* filename, int line_number) {
  std::string bit_str = hex_to_string(hex);
  auto period_begin = bit_str.find("11");
  auto period_end = bit_str.rfind("11");
  if (period_begin == std::string::npos || period_end == std::string::npos ||
      period_begin == period_end || period_end - period_begin <= 2) {
    throw parser_error(filename, line_number);
  }
  std::string bitstring(std::next(begin(bit_str), period_begin + 2),
                        std::next(begin(bit_str), period_end));
  std::reverse(begin(bitstring), end(bitstring));
  if (bitstring.size() > MAX_DAYS) {
    LOG(error) << bitstring.size()
               << " bits in traffic days > MAX_DAYS=" << MAX_DAYS;
    throw parser_error{filename, line_number};
  }
  return bitfield{bitstring};
}

std::map<int, bitfield> parse_bitfields(loaded_file const& f, config const& c) {
  scoped_timer timer("parsing bitfields");

  std::map<int, bitfield> bitfields;
  for_each_line_numbered(f.content(), [&](cstr line, int line_number) {
    if (line.len == 0 || line.str[0] == '%') {
      return;
    } else if (line.len < 9) {
      throw parser_error(f.name(), line_number);
    }

    bitfields[parse<int>(line.substr(c.bf_.index_))] =
        hex_str_to_bitset(line.substr(c.bf_.value_), f.name(), line_number);
  });

  bitfields[ALL_DAYS_KEY] = create_uniform_bitfield('1');

  return bitfields;
}

}  // namespace motis::loader::hrd
