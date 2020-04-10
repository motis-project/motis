#include "motis/loader/hrd/model/range.h"

#include <cassert>

#include "utl/parser/arg_parser.h"
#include "utl/verify.h"

#include "motis/core/common/date_time_util.h"
#include "motis/loader/util.h"

using namespace utl;

namespace motis::loader::hrd {

bool is_index(cstr s) { return s[0] == '#'; }

int parse_index(cstr s) { return parse<int>(s.substr(1)); }

int get_index(std::vector<hrd_service::stop> const& stops, cstr eva_or_idx,
              cstr hhmm_or_idx, bool is_departure_event) {
  assert(!eva_or_idx.empty() && !hhmm_or_idx.empty());
  if (is_index(eva_or_idx)) {
    // eva_or_idx is an index which is already definite
    return parse_index(eva_or_idx);
  } else if (is_index(hhmm_or_idx) || hhmm_or_idx.trim().len == 0) {
    // eva_or_idx is not an index -> eva_or_idx is an eva number
    // hhmm_or_idx is empty -> search for first occurrence
    // hhmm_or_idx is an index -> search for nth occurrence
    const auto eva_num = parse<int>(eva_or_idx);
    const auto n = is_index(hhmm_or_idx) ? parse_index(hhmm_or_idx) + 1 : 1;
    const auto it = find_nth(
        begin(stops), end(stops), n,
        [&](hrd_service::stop const& s) { return s.eva_num_ == eva_num; });
    utl::verify(it != end(stops), "{}th occurrence of eva number {} not found",
                n, eva_num);
    return static_cast<int>(std::distance(begin(stops), it));
  } else {
    // hhmm_or_idx must be a time
    // -> return stop where eva number and time matches
    const auto eva_num = parse<int>(eva_or_idx);
    const auto time = hhmm_to_min(parse<int>(hhmm_or_idx.substr(1)));
    const auto it =
        std::find_if(begin(stops), end(stops), [&](hrd_service::stop const& s) {
          return s.eva_num_ == eva_num &&
                 (is_departure_event ? s.dep_.time_ : s.arr_.time_) == time;
        });
    utl::verify(it != end(stops),
                "event with time {} at eva number {} not found", time, eva_num);
    return static_cast<int>(std::distance(begin(stops), it));
  }
}

range::range(std::vector<hrd_service::stop> const& stops, cstr from_eva_or_idx,
             cstr to_eva_or_idx, cstr from_hhmm_or_idx, cstr to_hhmm_or_idx) {
  if (from_eva_or_idx.trim().empty() && to_eva_or_idx.trim().empty() &&
      from_hhmm_or_idx.trim().empty() && to_hhmm_or_idx.trim().empty()) {
    from_idx_ = 0;
    to_idx_ = stops.size() - 1;
  } else {
    from_idx_ = get_index(stops, from_eva_or_idx, from_hhmm_or_idx, true);
    to_idx_ = get_index(stops, to_eva_or_idx, to_hhmm_or_idx, false);
  }
}

}  // namespace motis::loader::hrd
