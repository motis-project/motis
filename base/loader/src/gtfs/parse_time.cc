#include "motis/loader/gtfs/parse_time.h"

#include "utl/parser/arg_parser.h"

using namespace utl;

namespace motis::loader::gtfs {

int hhmm_to_min(cstr s) {
  if (s.empty()) {
    return kInterpolate;
  } else {
    int hours = 0;
    parse_arg(s, hours, 0);
    if (s) {
      ++s;
    } else {
      return -1;
    }

    int minutes = 0;
    parse_arg(s, minutes, 0);

    return hours * 60 + minutes;
  }
}

}  // namespace motis::loader::gtfs
