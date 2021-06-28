#include "motis/paxmon/service_info.h"

#include <algorithm>

#include "utl/to_vec.h"

#include "motis/core/access/service_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/hash_map.h"

namespace motis::paxmon {

service_info get_service_info(schedule const& sched, connection const& fc,
                              connection_info const* ci) {
  return service_info{get_service_name(sched, ci),
                      sched.categories_.at(ci->family_)->name_.view(),
                      output_train_nr(ci->train_nr_, ci->original_train_nr_),
                      ci->line_identifier_.view(),
                      ci->provider_ != nullptr
                          ? ci->provider_->full_name_.view()
                          : std::string_view{},
                      fc.clasz_};
}

std::vector<std::pair<service_info, unsigned>> get_service_infos(
    schedule const& sched, trip const* trp) {
  mcd::hash_map<service_info, unsigned> si_counts;
  for (auto const& section : motis::access::sections(trp)) {
    auto const& fc = section.fcon();
    for (auto ci = fc.con_info_; ci != nullptr; ci = ci->merged_with_) {
      auto const si = get_service_info(sched, fc, ci);
      ++si_counts[si];
    }
  }
  auto sis = utl::to_vec(si_counts, [](auto const& e) {
    return std::make_pair(e.first, e.second);
  });
  std::sort(begin(sis), end(sis),
            [](auto const& a, auto const& b) { return a.second > b.second; });
  return sis;
}

}  // namespace motis::paxmon
