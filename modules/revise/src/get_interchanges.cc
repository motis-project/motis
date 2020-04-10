#include "motis/revise/get_interchanges.h"

namespace motis::revise {

std::vector<extern_interchange> get_interchanges(journey const& j) {
  // Get first enter and last exit.
  auto const first_it =
      std::find_if(begin(j.stops_), end(j.stops_),
                   [](journey::stop const& s) { return s.enter_; });
  auto const last_it =
      std::find_if(j.stops_.rbegin(), j.stops_.rend(),
                   [](journey::stop const& s) { return s.exit_; });
  utl::verify(first_it != end(j.stops_),
              "get interchange(first): invalid journey");
  utl::verify(last_it != j.stops_.rend(),
              "get interchange(last): invalid journey");

  // Get interchanges in ]first enter, last exit[.
  std::vector<extern_interchange> interchanges;
  for (auto it = std::next(first_it); it != last_it.base() - 1; ++it) {
    if (it->exit_) {
      interchanges.emplace_back(
          *it, static_cast<int>(std::distance(begin(j.stops_), it)));
    }
    if (it->enter_) {
      utl::verify(!interchanges.empty(),
                  "get interchage(empty): invalid journey");
      interchanges.back().second_stop_ = *it;
      interchanges.back().second_stop_idx_ = std::distance(begin(j.stops_), it);
    }
  }
  utl::verify(std::all_of(begin(interchanges), end(interchanges),
                          [](extern_interchange const& ic) {
                            return ic.first_stop_idx_ != -1 &&
                                   ic.second_stop_idx_ != -1;
                          }),
              "get interchange(all): invalid journey");
  return interchanges;
}

}  // namespace motis::revise