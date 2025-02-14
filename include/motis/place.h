#pragma once

#include "osr/location.h"

#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"

namespace motis {

struct tt_location {
  explicit tt_location(nigiri::rt::run_stop const& stop);
  explicit tt_location(
      nigiri::location_idx_t l,
      nigiri::location_idx_t scheduled = nigiri::location_idx_t::invalid());

  friend std::ostream& operator<<(std::ostream& out, tt_location const& l) {
    return out << "{ l=" << l.l_ << ", scheduled=" << l.scheduled_ << " }";
  }

  nigiri::location_idx_t l_;
  nigiri::location_idx_t scheduled_;
};

using place_t = std::variant<osr::location, tt_location>;

inline std::ostream& operator<<(std::ostream& out, place_t const p) {
  return std::visit([&](auto const l) -> std::ostream& { return out << l; }, p);
}

osr::level_t get_lvl(osr::ways const*,
                     osr::platforms const*,
                     platform_matches_t const*,
                     nigiri::location_idx_t);

api::Place to_place(osr::location, std::string_view name);

api::Place to_place(nigiri::timetable const*,
                    tag_lookup const*,
                    osr::ways const* w,
                    osr::platforms const* pl,
                    platform_matches_t const* matches,
                    place_t l,
                    place_t start = osr::location{},
                    place_t dest = osr::location{},
                    std::string_view name = "");

osr::location get_location(api::Place const&);

osr::location get_location(nigiri::timetable const*,
                           osr::ways const*,
                           osr::platforms const*,
                           platform_matches_t const*,
                           place_t const loc,
                           place_t const start = {},
                           place_t const dest = {});

}  // namespace motis