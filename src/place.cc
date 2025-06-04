#include "motis/place.h"

#include "osr/platforms.h"

#include "nigiri/rt/frun.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

#include "motis/tag_lookup.h"

namespace n = nigiri;

namespace motis {

tt_location::tt_location(nigiri::rt::run_stop const& stop)
    : l_{stop.get_location_idx()},
      scheduled_{stop.get_scheduled_location_idx()} {}

tt_location::tt_location(nigiri::location_idx_t const l,
                         nigiri::location_idx_t const scheduled)
    : l_{l},
      scheduled_{scheduled == n::location_idx_t::invalid() ? l : scheduled} {}

api::Place to_place(osr::location const l, std::string_view name) {
  return {
      .name_ = std::string{name},
      .lat_ = l.pos_.lat_,
      .lon_ = l.pos_.lng_,
      .level_ = l.lvl_.to_float(),
      .vertexType_ = api::VertexTypeEnum::NORMAL,
  };
}

osr::level_t get_lvl(osr::ways const* w,
                     osr::platforms const* pl,
                     platform_matches_t const* matches,
                     n::location_idx_t const l) {
  return w && pl && matches ? pl->get_level(*w, (*matches)[l]) : osr::kNoLevel;
}

double get_level(osr::ways const* w,
                 osr::platforms const* pl,
                 platform_matches_t const* matches,
                 n::location_idx_t const l) {
  return get_lvl(w, pl, matches, l).to_float();
}

osr::location get_location(api::Place const& p) {
  return {{p.lat_, p.lon_}, osr::level_t{static_cast<float>(p.level_)}};
}

osr::location get_location(n::timetable const* tt,
                           osr::ways const* w,
                           osr::platforms const* pl,
                           platform_matches_t const* matches,
                           place_t const loc,
                           place_t const start,
                           place_t const dest) {
  return std::visit(
      utl::overloaded{
          [&](osr::location const& l) { return l; },
          [&](tt_location const l) {
            auto const l_idx = l.l_;
            switch (to_idx(l_idx)) {
              case static_cast<n::location_idx_t::value_t>(
                  n::special_station::kStart):
                assert(std::holds_alternative<osr::location>(start));
                return std::get<osr::location>(start);
              case static_cast<n::location_idx_t::value_t>(
                  n::special_station::kEnd):
                assert(std::holds_alternative<osr::location>(dest));
                return std::get<osr::location>(dest);
              default:
                utl::verify(tt != nullptr,
                            "resolving stop coordinates: timetable not set");
                return osr::location{tt->locations_.coordinates_[l_idx],
                                     get_lvl(w, pl, matches, l_idx)};
            }
          }},
      loc);
}

api::Place to_place(n::timetable const* tt,
                    tag_lookup const* tags,
                    osr::ways const* w,
                    osr::platforms const* pl,
                    platform_matches_t const* matches,
                    place_t const l,
                    place_t const start,
                    place_t const dest,
                    std::string_view name) {

  return std::visit(
      utl::overloaded{
          [&](osr::location const& l) { return to_place(l, name); },
          [&](tt_location const tt_l) -> api::Place {
            utl::verify(tt && tags, "resolving stops requires timetable");

            auto const l = tt_l.l_;
            if (l == n::get_special_station(n::special_station::kStart)) {
              return to_place(std::get<osr::location>(start), "START");
            } else if (l == n::get_special_station(n::special_station::kEnd)) {
              return to_place(std::get<osr::location>(dest), "END");
            } else {
              auto const is_track = [&](n::location_idx_t const x) {
                auto const type = tt->locations_.types_.at(x);
                return (type == n::location_type::kGeneratedTrack ||
                        type == n::location_type::kTrack);
              };

              auto const get_track = [&](n::location_idx_t const x) {
                return is_track(x) ? std::optional{std::string{
                                         tt->locations_.names_.at(x).view()}}
                                   : std::nullopt;
              };

              // check if description is available, if not, return nullopt
              auto const get_description = [&](n::location_idx_t const x) {
                return tt->locations_.descriptions_.at(x).empty()
                           ? std::nullopt
                           : std::optional{std::string{
                                 tt->locations_.descriptions_.at(x).view()}};
              };

              auto const pos = tt->locations_.coordinates_[l];
              auto const p =
                  is_track(tt_l.l_) ? tt->locations_.parents_.at(l) : l;
              return {.name_ = std::string{tt->locations_.names_[p].view()},
                      .stopId_ = tags->id(*tt, l),
                      .lat_ = pos.lat_,
                      .lon_ = pos.lng_,
                      .level_ = get_level(w, pl, matches, l),
                      .scheduledTrack_ = get_track(tt_l.scheduled_),
                      .track_ = get_track(tt_l.l_),
                      .description_ = get_description(tt_l.scheduled_),
                      .vertexType_ = api::VertexTypeEnum::TRANSIT};
            }
          }},
      l);
}

}  // namespace motis