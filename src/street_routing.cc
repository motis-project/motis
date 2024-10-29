#include "motis/street_routing.h"

#include "geo/polyline_format.h"

#include "motis/constants.h"
#include "motis/update_rtt_td_footpaths.h"

namespace motis {

std::optional<osr::path> get_path(osr::ways const& w,
                                  osr::lookup const& l,
                                  elevators const* e,
                                  osr::sharing_data const* sharing,
                                  osr::location const& from,
                                  osr::location const& to,
                                  transport_mode_t const transport_mode,
                                  osr::search_profile const profile,
                                  nigiri::unixtime_t const start_time,
                                  street_routing_cache_t& cache,
                                  osr::bitvec<osr::node_idx_t>& blocked_mem) {
  auto const s = e ? get_states_at(w, l, *e, start_time, from.pos_)
                   : std::optional{std::pair<nodes_t, states_t>{}};
  auto const& [e_nodes, e_states] = *s;
  auto const key =
      street_routing_cache_key_t{from, to, transport_mode, start_time};
  auto const it = cache.find(key);
  auto const path =
      it != end(cache)
          ? it->second
          : osr::route(
                w, l, profile, from, to, 3600, osr::direction::kForward,
                kMaxMatchingDistance,
                s ? &set_blocked(e_nodes, e_states, blocked_mem) : nullptr,
                sharing);
  if (it == end(cache)) {
    cache.emplace(std::pair{key, path});
  }
  if (!path.has_value()) {
    if (it == end(cache)) {
      std::cout << "no path found: " << from << " -> " << to
                << ", profile=" << to_str(profile) << std::endl;
    }
  }
  return path;
}

template <std::int64_t Precision>
api::EncodedPolyline to_polyline(geo::polyline const& polyline) {
  return {geo::encode_polyline<Precision>(polyline),
          static_cast<std::int64_t>(polyline.size())};
}

template <>
api::EncodedPolyline to_polyline<5>(geo::polyline const&);

template <>
api::EncodedPolyline to_polyline<7>(geo::polyline const&);

std::vector<api::StepInstruction> get_step_instructions(
    osr::ways const& w, std::span<osr::path::segment const> segments) {
  return utl::to_vec(
      segments, [&](osr::path::segment const& s) -> api::StepInstruction {
        auto const way_name = s.way_ == osr::way_idx_t::invalid()
                                  ? osr::string_idx_t::invalid()
                                  : w.way_names_[s.way_];
        return {
            .relativeDirection_ = api::RelativeDirectionEnum::CONTINUE,  // TODO
            .absoluteDirection_ = api::AbsoluteDirectionEnum::NORTH,  // TODO
            .distance_ = static_cast<double>(s.dist_),
            .fromLevel_ = to_float(s.from_level_),
            .toLevel_ = to_float(s.to_level_),
            .osmWay_ = s.way_ == osr::way_idx_t ::invalid()
                           ? std::nullopt
                           : std::optional{static_cast<std::int64_t>(
                                 to_idx(w.way_osm_idx_[s.way_]))},
            .polyline_ = {geo::encode_polyline<7>(polyline),
                          static_cast<std::int64_t>(polyline.size())},
            .streetName_ = way_name == osr::string_idx_t::invalid()
                               ? ""
                               : std::string{w.strings_[way_name].view()},
            .exit_ = {},  // TODO
            .stayOn_ = false,  // TODO
            .area_ = false  // TODO
        };
      });
}

}  // namespace motis