#include "motis/odm/raptor_wrapper.h"

#include "nigiri/routing/raptor_search.h"

#include "motis/endpoints/routing.h"

namespace n = nigiri;

namespace motis::odm {

raptor_result::raptor_result(
    nigiri::routing::routing_result<nigiri::routing::raptor_stats>&& r)
    : journeys_{*r.journeys_},
      interval_{r.interval_},
      search_stats_{r.search_stats_},
      algo_stats_{r.algo_stats_} {
  std::cout << "raptor_result ctor\n";
}

raptor_result raptor_wrapper(
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

  auto const start_time_str = [](auto const& t) {
    return std::visit(utl::overloaded{[](n::unixtime_t const& u) {
                                        auto ss = std::stringstream{};
                                        ss << u;
                                        return ss.str();
                                      },
                                      [](n::interval<n::unixtime_t> const& i) {
                                        auto ss = std::stringstream{};
                                        ss << i;
                                        return ss.str();
                                      }},
                      t);
  };
  std::cout << "raptor_wrapper start time: " << start_time_str(q.start_time_)
            << "\n";

  auto const result = raptor_result{
      n::routing::raptor_search(tt, rtt, *ep::search_state, *ep::raptor_state,
                                std::move(q), search_dir, timeout)};

  std::cout << "result journeys:\n";
  for (auto const& j : result.journeys_) {
    j.print(std::cout, tt);
  }

  return result;
}

}  // namespace motis::odm