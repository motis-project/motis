#include "motis/loader/hrd/parser/directions_parser.h"

#include "motis/loader/parser_error.h"
#include "motis/loader/util.h"

namespace motis::loader::hrd {

std::map<uint64_t, std::string> parse_directions(loaded_file const& file,
                                                 config const& c) {
  std::map<uint64_t, std::string> directions;
  for_each_line_numbered(file.content(), [&](cstr line, int line_number) {
    if (line.length() < 9 && line[7] == ' ') {
      throw parser_error(file.name(), line_number);
    } else {
      auto const text = line.substr(c.dir_.text_);
      directions[raw_to_int<uint64_t>(line.substr(c.dir_.eva_))] =
          std::string(text.str, text.len);
    }
  });
  return directions;
}

}  // namespace motis::loader::hrd
