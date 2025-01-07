#include "motis/odm/equal_journeys.h"

#include "utl/overloaded.h"
#include "utl/zip.h"

namespace n = nigiri;

namespace motis::odm {

bool operator==(nigiri::routing::journey const& a,
                nigiri::routing::journey const& b) {
  if (std::tie(a.start_time_, a.dest_time_, a.dest_, a.transfers_) !=
          std::tie(b.start_time_, b.dest_time_, b.dest_, b.transfers_) ||
      a.legs_.size() != b.legs_.size()) {
    return false;
  }

  auto const zip_legs = utl::zip(a.legs_, b.legs_);
  return std::all_of(begin(zip_legs), end(zip_legs), [&](auto const& t) {
    auto const& l1 = std::get<0>(t);
    auto const& l2 = std::get<1>(t);

    return (std::tie(l1.from_, l1.to_, l1.dep_time_, l1.arr_time_) ==
            std::tie(l2.from_, l2.to_, l2.dep_time_, l2.arr_time_)) &&
           std::visit(
               utl::overloaded{
                   [&](n::routing::journey::run_enter_exit const& ree1) {
                     return std::holds_alternative<
                                n::routing::journey::run_enter_exit>(
                                l2.uses_) &&
                            ree1.stop_range_ ==
                                std::get<n::routing::journey::run_enter_exit>(
                                    l2.uses_)
                                    .stop_range_;
                   },
                   [&](n::footpath const& fp1) {
                     return std::holds_alternative<n::footpath>(l2.uses_) &&
                            fp1 == std::get<n::footpath>(l2.uses_);
                   },
                   [&](n::routing::offset const& o1) {
                     return std::holds_alternative<n::routing::offset>(
                                l2.uses_) &&
                            o1 == std::get<n::routing::offset>(l2.uses_);
                   }},
               l1.uses_);
  });
}

}  // namespace motis::odm