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

api::Place to_place(osr::location,
                    std::string_view name,
                    std::optional<std::string> const& tz);

api::Place to_place(
    nigiri::timetable const*,
    tag_lookup const*,
    osr::ways const*,
    osr::platforms const*,
    platform_matches_t const*,
    adr_ext const*,
    tz_map_t const*,
    nigiri::lang_t const&,
    place_t,
    place_t start = osr::location{},
    place_t dest = osr::location{},
    std::string_view name = "",
    std::optional<std::string> const& fallback_tz = std::nullopt);

api::Place to_place(nigiri::timetable const*,
                    tag_lookup const*,
                    osr::ways const*,
                    osr::platforms const*,
                    platform_matches_t const*,
                    adr_ext const* ae,
                    tz_map_t const* tz,
                    nigiri::lang_t const&,
                    nigiri::rt::run_stop const&,
                    place_t start = osr::location{},
                    place_t dest = osr::location{});

osr::location get_location(api::Place const&);

osr::location get_location(nigiri::timetable const*,
                           osr::ways const*,
                           osr::platforms const*,
                           platform_matches_t const*,
                           place_t const loc,
                           place_t const start = {},
                           place_t const dest = {});

place_t get_place(nigiri::timetable const*,
                  tag_lookup const*,
                  std::string_view user_input);

}  // namespace motis
