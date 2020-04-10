#include "motis/ris/risml/parse_type.h"

#include <map>

using namespace flatbuffers;
using namespace utl;

namespace motis::ris::risml {

boost::optional<EventType> parse_type(
    cstr const& raw, boost::optional<EventType> const& default_value) {
  static const std::map<cstr, EventType> map({{"Start", EventType_DEP},
                                              {"Ab", EventType_DEP},
                                              {"An", EventType_ARR},
                                              {"Ziel", EventType_ARR}});
  auto it = map.find(raw);
  if (it == end(map)) {
    return default_value;
  }
  return it->second;
}

}  // namespace motis::ris::risml
