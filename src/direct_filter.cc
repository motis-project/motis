#include "motis/direct_filter.h"

#include "utl/erase_if.h"
#include "utl/visit.h"

#include "nigiri/types.h"

namespace motis {

using namespace std::chrono_literals;
namespace n = nigiri;

void direct_filter(std::vector<api::Itinerary> const& direct,
                   std::vector<n::routing::journey>& journeys) {
  auto const get_direct_duration = [&](auto const transport_mode_id) {
    auto const m = static_cast<api::ModeEnum>(transport_mode_id);
    auto const i = utl::find_if(
        direct, [&](auto const& d) { return d.legs_.front().mode_ == m; });
    return i != end(direct)
               ? n::duration_t{std::chrono::round<std::chrono::minutes>(
                     std::chrono::seconds{i->duration_})}
               : n::duration_t::max();
  };

  auto const not_better_than_direct = [&](n::routing::journey const& j) {
    auto const first_leg_offset = utl::visit(
        j.legs_.front().uses_,
        [&](n::routing::offset const& o) { return std::optional{o}; });

    auto const last_leg_offset = utl::visit(
        j.legs_.back().uses_,
        [&](n::routing::offset const& o) { return std::optional{o}; });

    auto const longer_than_direct = [&](n::routing::offset const& o) {
      return std::optional{o.duration_ >=
                           get_direct_duration(o.transport_mode_id_)};
    };

    return first_leg_offset.and_then(longer_than_direct).value_or(false) ||
           last_leg_offset.and_then(longer_than_direct).value_or(false) ||
           (first_leg_offset && last_leg_offset &&
            first_leg_offset->transport_mode_id_ ==
                last_leg_offset->transport_mode_id_ &&
            first_leg_offset->duration_ + last_leg_offset->duration_ >=
                get_direct_duration(first_leg_offset->transport_mode_id_));
  };

  utl::erase_if(journeys, not_better_than_direct);
}

}  // namespace motis