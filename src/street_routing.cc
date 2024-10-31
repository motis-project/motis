#include "motis/street_routing.h"

#include "geo/polyline_format.h"

#include "utl/concat.h"

#include "osr/routing/route.h"
#include "osr/routing/sharing_data.h"

#include "motis/constants.h"
#include "motis/mode_to_profile.h"
#include "motis/place.h"
#include "motis/polyline.h"
#include "motis/update_rtt_td_footpaths.h"

namespace n = nigiri;

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
                                  osr::cost_t const max,
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
                w, l, profile, from, to, max, osr::direction::kForward,
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

std::vector<api::StepInstruction> get_step_instructions(
    osr::ways const& w, std::span<osr::path::segment const> segments) {
  return utl::to_vec(segments, [&](osr::path::segment const& s) {
    auto const way_name = s.way_ == osr::way_idx_t::invalid()
                              ? osr::string_idx_t::invalid()
                              : w.way_names_[s.way_];
    return api::StepInstruction{
        .relativeDirection_ = api::RelativeDirectionEnum::CONTINUE,  // TODO
        .absoluteDirection_ = api::AbsoluteDirectionEnum::NORTH,  // TODO
        .distance_ = static_cast<double>(s.dist_),
        .fromLevel_ = to_float(s.from_level_),
        .toLevel_ = to_float(s.to_level_),
        .osmWay_ = s.way_ == osr::way_idx_t ::invalid()
                       ? std::nullopt
                       : std::optional{static_cast<std::int64_t>(
                             to_idx(w.way_osm_idx_[s.way_]))},
        .polyline_ = to_polyline<7>(s.polyline_),
        .streetName_ = way_name == osr::string_idx_t::invalid()
                           ? ""
                           : std::string{w.strings_[way_name].view()},
        .exit_ = {},  // TODO
        .stayOn_ = false,  // TODO
        .area_ = false  // TODO
    };
  });
}

struct sharing {
  sharing(osr::ways const& w,
          gbfs::gbfs_data const& gbfs,
          gbfs_provider_idx_t const provider_idx)
      : w_{w}, gbfs_{gbfs}, provider_{gbfs_.providers_.at(provider_idx)} {}

  api::Rental get_rental(osr::node_idx_t const n) const {
    auto ret = rental_;
    auto const& an = provider_.additional_nodes_.at(get_additional_node_idx(n));
    std::visit(utl::overloaded{
                   [&](gbfs::additional_node::station const& s) {
                     auto const& st = provider_.stations_.at(s.id_);
                     ret.stationName_ = st.info_.name_;
                     ret.rentalUriAndroid_ = st.info_.rental_uris_.android_;
                     ret.rentalUriIOS_ = st.info_.rental_uris_.ios_;
                     ret.rentalUriWeb_ = st.info_.rental_uris_.web_;
                   },
                   [&](gbfs::additional_node::vehicle const& v) {
                     auto const& vi = provider_.vehicle_status_.at(v.idx_);
                     ret.rentalUriAndroid_ = vi.rental_uris_.android_;
                     ret.rentalUriIOS_ = vi.rental_uris_.ios_;
                     ret.rentalUriWeb_ = vi.rental_uris_.web_;
                   }},
               an.data_);
    return ret;
  }

  geo::latlng get_node_pos(osr::node_idx_t const n) const {
    return std::visit(
        utl::overloaded{
            [&](gbfs::additional_node::station const& s) {
              return provider_.stations_.at(s.id_).info_.pos_;
            },
            [&](gbfs::additional_node::vehicle const& vehicle) {
              return provider_.vehicle_status_.at(vehicle.idx_).pos_;
            }},
        provider_.additional_nodes_.at(get_additional_node_idx(n)).data_);
  }

  std::size_t get_additional_node_idx(osr::node_idx_t const n) const {
    return to_idx(n) - sharing_data_.additional_node_offset_;
  }

  osr::ways const& w_;
  gbfs::gbfs_data const& gbfs_;
  gbfs::gbfs_provider const& provider_;
  osr::sharing_data sharing_data_{
      .start_allowed_ = provider_.start_allowed_,
      .end_allowed_ = provider_.end_allowed_,
      .through_allowed_ = provider_.through_allowed_,
      .additional_node_offset_ = w_.n_nodes(),
      .additional_edges_ = provider_.additional_edges_};
  api::Rental rental_{
      .systemId_ = provider_.sys_info_.id_,
      .systemName_ = provider_.sys_info_.name_,
      .url_ = provider_.sys_info_.url_,
  };
};

api::Itinerary route(osr::ways const& w,
                     osr::lookup const& l,
                     gbfs::gbfs_data const* gbfs,
                     elevators const* e,
                     api::Place const& from,
                     api::Place const& to,
                     api::ModeEnum const mode,
                     bool const wheelchair,
                     n::unixtime_t const start_time,
                     std::optional<n::unixtime_t> const end_time,
                     gbfs_provider_idx_t const provider_idx,
                     street_routing_cache_t& cache,
                     osr::bitvec<osr::node_idx_t>& blocked_mem,
                     std::chrono::seconds const max) {
  auto const profile = to_profile(mode, wheelchair);
  utl::verify(profile != osr::search_profile::kBikeSharing || gbfs != nullptr,
              "sharing mobility not configured");

  auto const is_additional_node = [&](osr::node_idx_t const n) {
    return n >= w.n_nodes();
  };

  auto const sharing_data = profile == osr::search_profile::kBikeSharing
                                ? std::optional{sharing(w, *gbfs, provider_idx)}
                                : std::nullopt;

  auto const get_node_pos = [&](osr::node_idx_t const n) -> geo::latlng {
    if (n == osr::node_idx_t::invalid()) {
      return {};
    } else if (!is_additional_node(n)) {
      return w.get_node_pos(n).as_latlng();
    } else {
      return sharing_data.value().get_node_pos(n);
    }
  };

  auto const path = [&]() {
    auto p =
        get_path(w, l, e, sharing_data ? &sharing_data->sharing_data_ : nullptr,
                 get_location(from), get_location(to),
                 static_cast<transport_mode_t>(
                     to_idx(provider_idx + kGbfsTransportModeIdOffset)),
                 to_profile(mode, wheelchair), start_time,
                 static_cast<osr::cost_t>(max.count()), cache, blocked_mem);

    if (p.has_value() && profile == osr::search_profile::kBikeSharing) {
      // Coordinates of additional nodes are not known to osr.
      // Therefore, segments to/from additional have empty polylines.
      for (auto& s : p->segments_) {
        if (s.polyline_.empty()) {
          s.polyline_ =
              geo::polyline{get_node_pos(s.from_), get_node_pos(s.to_)};
        }
      }
    }

    return p;
  }();

  if (!path.has_value()) {
    if (!end_time.has_value()) {
      return {};
    }

    auto itinerary = api::Itinerary{
        .duration_ = std::chrono::duration_cast<std::chrono::seconds>(
                         *end_time - start_time)
                         .count(),
        .startTime_ = start_time,
        .endTime_ = *end_time};
    auto& leg = itinerary.legs_.emplace_back(
        api::Leg{.mode_ = mode,
                 .from_ = from,
                 .to_ = to,
                 .duration_ = std::chrono::duration_cast<std::chrono::seconds>(
                                  *end_time - start_time)
                                  .count(),
                 .startTime_ = start_time,
                 .endTime_ = *end_time});
    leg.from_.departure_ = leg.startTime_;
    leg.to_.arrival_ = leg.endTime_;
    return itinerary;
  }

  auto itinerary = api::Itinerary{
      .duration_ = end_time ? std::chrono::duration_cast<std::chrono::seconds>(
                                  *end_time - start_time)
                                  .count()
                            : path->cost_,
      .startTime_ = start_time,
      .endTime_ = start_time + std::chrono::seconds{path->cost_},
      .transfers_ = 0};

  auto t = std::chrono::time_point_cast<std::chrono::seconds>(start_time);
  auto pred_place = from;
  auto pred_end_time = t;
  utl::equal_ranges_linear(
      path->segments_, [](auto&& a, auto&& b) { return a.mode_ == b.mode_; },
      [&](auto&& lb, auto&& ub) {
        auto const range = std::span{lb, ub};
        auto const is_last_leg = ub == end(path->segments_);
        auto const is_bike_leg = lb->mode_ == osr::mode::kBike;
        auto const is_rental =
            (profile == osr::search_profile::kBikeSharing && is_bike_leg &&
             is_additional_node(range.back().to_));

        auto const to_node = range.back().to_;
        auto const to_pos = get_node_pos(to_node);
        auto const next_place =
            is_last_leg ? to
            // All modes except sharing mobility have only one leg.
            // -> This is not the last leg = it has to be sharing mobility.
            : profile == osr::search_profile::kBikeSharing
                ? api::Place{.name_ =
                                 sharing_data.value().provider_.sys_info_.name_,
                             .lat_ = to_pos.lat_,
                             .lon_ = to_pos.lng_,
                             .vertexType_ = api::VertexTypeEnum::BIKESHARE}
                : api::Place{.lat_ = to_pos.lat_,
                             .lon_ = to_pos.lng_,
                             .vertexType_ = api::VertexTypeEnum::NORMAL};

        auto concat = geo::polyline{};
        auto dist = 0.0;
        for (auto const& p : range) {
          utl::concat(concat, p.polyline_);
          if (p.cost_ != osr::kInfeasible) {
            t += std::chrono::seconds{p.cost_};
            dist += p.dist_;
          }
        }

        auto& leg = itinerary.legs_.emplace_back(api::Leg{
            .mode_ = (lb->mode_ == osr::mode::kBike &&
                      profile == osr::search_profile::kBikeSharing)
                         ? api::ModeEnum::BIKE_RENTAL
                         : to_mode(lb->mode_),
            .from_ = pred_place,
            .to_ = next_place,
            .duration_ = std::chrono::duration_cast<std::chrono::seconds>(
                             t - pred_end_time)
                             .count(),
            .startTime_ = pred_end_time,
            .endTime_ = is_last_leg && end_time ? *end_time : t,
            .distance_ = dist,
            .legGeometry_ = to_polyline<7>(concat),
            .steps_ = get_step_instructions(w, range),
            .rental_ = is_rental ? std::optional{sharing_data->get_rental(
                                       range.back().to_)}
                                 : std::nullopt});

        leg.from_.departure_ = leg.startTime_;
        leg.to_.arrival_ = leg.endTime_;

        pred_place = next_place;
        pred_end_time = t;
      });

  return itinerary;
}

}  // namespace motis