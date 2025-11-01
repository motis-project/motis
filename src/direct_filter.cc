#include "motis/direct_filter.h"

#include <variant>

#include "utl/erase_if.h"

#include "nigiri/types.h"

namespace motis {

using namespace std::chrono_literals;

void direct_filter(std::vector<api::Itinerary> const& direct,
                   std::vector<nigiri::routing::journey>& journeys) {
  auto const get_direct_duration = [&](auto const m) {
    auto const i = utl::find_if(
        direct, [&](auto const& d) { return d.legs_.front().mode_ == m; });
    return i != end(direct)
               ? nigiri::duration_t{std::chrono::round<std::chrono::minutes>(
                     std::chrono::seconds{i->duration_})}
               : nigiri::duration_t::max();
  };

  auto const not_better_than_direct = [&](nigiri::routing::journey const& j) {
    auto const first_leg_mode =
        std::holds_alternative<nigiri::routing::offset>(j.legs_.front().uses_)
            ? std::optional{std::get<nigiri::routing::offset>(
                                j.legs_.front().uses_)
                                .transport_mode_id_}
            : std::nullopt;

    auto const last_leg_mode =
        std::holds_alternative<nigiri::routing::offset>(j.legs_.back().uses_)
            ? std::optional{std::get<nigiri::routing::offset>(
                                j.legs_.back().uses_)
                                .transport_mode_id_}
            : std::nullopt;

    if ((first_leg_mode || last_leg_mode) &&
        (!(first_leg_mode && last_leg_mode) ||
         *first_leg_mode == *last_leg_mode)) {
      auto const pre_post_duration =
          (first_leg_mode
               ? std::get<nigiri::routing::offset>(j.legs_.front().uses_)
                     .duration_
               : nigiri::duration_t{0min}) +
          (last_leg_mode
               ? std::get<nigiri::routing::offset>(j.legs_.back().uses_)
                     .duration_
               : nigiri::duration_t{0min});
      auto const mode = static_cast<api::ModeEnum>(
          first_leg_mode ? *first_leg_mode : *last_leg_mode);
      auto const not_better = pre_post_duration >= get_direct_duration(mode);
      if (not_better) {
        fmt::println("[direct_filter] dep: {}, arr: {}, pre_post_duration: {}",
                     j.departure_time(), j.arrival_time(), pre_post_duration);
      }
      return not_better;
    }

    return false;
  };

  utl::erase_if(journeys, not_better_than_direct);
}

}  // namespace motis