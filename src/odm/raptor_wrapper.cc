#include "motis/odm/raptor_wrapper.h"

#include "nigiri/routing/raptor_search.h"

#include "motis/endpoints/routing.h"

namespace n = nigiri;

namespace motis::odm {

n::routing::routing_result<n::routing::raptor_stats> route(
    n::timetable const& tt,
    n::rt_timetable const* rtt,
    n::routing::query q,
    n::direction const search_dir,
    std::optional<std::chrono::seconds> const timeout) {
  if (ep::search_state.get() == nullptr) {
    ep::search_state.reset(new n::routing::search_state{});
  }
  if (ep::raptor_state.get() == nullptr) {
    ep::raptor_state.reset(new n::routing::raptor_state{});
  }

  return n::routing::raptor_search(tt, rtt, *ep::search_state,
                                   *ep::raptor_state, std::move(q), search_dir,
                                   timeout);
}

}  // namespace motis::odm