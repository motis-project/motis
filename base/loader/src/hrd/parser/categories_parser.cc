#include "motis/loader/hrd/parser/categories_parser.h"

#include "utl/parser/arg_parser.h"
#include "utl/parser/cstr.h"

#include "motis/core/common/logging.h"
#include "motis/loader/parser_error.h"
#include "motis/loader/util.h"

using namespace utl;
using namespace motis::logging;

namespace motis::loader::hrd {

std::map<uint32_t, category> parse_categories(loaded_file const& file,
                                              config const& c) {
  scoped_timer timer("parsing categories");
  bool ignore = false;
  std::map<uint32_t, category> categories;
  for_each_line_numbered(file.content(), [&](cstr line, int line_number) {
    if (ignore || line.len <= 1 || line.str[0] == '#' || line.str[0] == '%') {
      return;
    } else if (line.starts_with("<")) {
      ignore = true;
      return;
    } else if (line.len < 20) {
      throw parser_error(file.name(), line_number);
    }

    auto const code = raw_to_int<uint32_t>(line.substr(c.cat_.code_));
    auto const output_rule = parse<uint8_t>(line.substr(c.cat_.output_rule_));
    auto const name = line.substr(c.cat_.name_).trim();
    categories[code] = {std::string(name.c_str(), name.length()), output_rule};
  });
  return categories;
}

}  // namespace motis::loader::hrd
