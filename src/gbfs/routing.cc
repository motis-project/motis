#include "motis/gbfs/routing.h"

#include "utl/concat.h"
#include "utl/overloaded.h"

#include "geo/polyline_format.h"

#include "osr/routing/route.h"
#include "osr/routing/sharing_data.h"

#include "motis/constants.h"
#include "motis/gbfs/data.h"

namespace n = nigiri;

namespace motis::gbfs {

api::Itinerary route(osr::ways const& w,
                     osr::lookup const& l,
                     gbfs_data const& gbfs,
                     elevators const* e,
                     api::Place const& from,
                     api::Place const& to,
                     n::unixtime_t const start_time,
                     gbfs_provider_idx_t const provider_idx,
                     street_routing_cache_t& cache,
                     osr::bitvec<osr::node_idx_t>& blocked_mem) {
  auto const& provider = gbfs.providers_.at(provider_idx);
  auto const sharing =
      osr::sharing_data{.start_allowed_ = provider.start_allowed_,
                        .end_allowed_ = provider.end_allowed_,
                        .through_allowed_ = provider.through_allowed_,
                        .additional_node_offset_ = w.n_nodes(),
                        .additional_edges_ = provider.additional_edges_};

  auto const get_node_pos = [&](osr::node_idx_t const n) -> geo::latlng {
    if (n == osr::node_idx_t::invalid()) {
      return {};
    } else if (to_idx(n) < sharing.additional_node_offset_) {
      return w.get_node_pos(n).as_latlng();
    } else {
      return std::visit(
          utl::overloaded{
              [&](additional_node::station const& s) {
                return provider.stations_.at(s.id_).info_.pos_;
              },
              [&](additional_node::vehicle const& vehicle) {
                return provider.vehicle_status_.at(vehicle.idx_).pos_;
              }},
          provider.additional_nodes_
              .at(to_idx(n) - sharing.additional_node_offset_)
              .data_);
    }
  };

  auto const path = [&]() {
    auto p = get_path(w, l, e, &sharing, get_location(from), get_location(to),
                      static_cast<transport_mode_t>(
                          to_idx(provider_idx + kGbfsTransportModeIdOffset)),
                      osr::search_profile::kBikeSharing, start_time, cache,
                      blocked_mem);
    if (p.has_value()) {
      // Post-processing polylines: coordinates of additional nodes are not
      // known to osr. Therefore, polylines to/from additional nodes are empty.
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
    return {};
  }

  auto itinerary =
      api::Itinerary{.duration_ = path->cost_,
                     .startTime_ = start_time,
                     .endTime_ = start_time + std::chrono::seconds{path->cost_},
                     .transfers_ = 0};

  auto rental = api::Rental{
      .systemId_ = provider.sys_info_.id_,
      .systemName_ = provider.sys_info_.name_,
      .url_ = provider.sys_info_.url_,
  };

  auto t = std::chrono::time_point_cast<std::chrono::seconds>(start_time);
  auto pred_place = from;
  auto pred_end_time = t;
  utl::equal_ranges_linear(
      path->segments_, [](auto&& a, auto&& b) { return a.mode_ == b.mode_; },
      [&](auto&& lb, auto&& ub) {
        auto const range = std::span{lb, ub};
        auto const is_last_leg = ub == end(path->segments_);
        auto const is_bike_leg = lb->mode_ == osr::mode::kBike;

        auto next_place =
            is_last_leg
                ? to
                : api::Place{.name_ = provider.sys_info_.name_,
                             .lat_ = 0,
                             .lon_ = 0,
                             .vertexType_ = api::VertexTypeEnum::BIKESHARE};

        if (!is_last_leg) {
          auto const to_node = range.back().to_;
          auto const to_pos = get_node_pos(to_node);
          next_place.lat_ = to_pos.lat_;
          next_place.lon_ = to_pos.lng_;

          if (to_idx(to_node) >= sharing.additional_node_offset_) {
            auto const& an = provider.additional_nodes_.at(
                to_idx(to_node) - sharing.additional_node_offset_);
            std::visit(
                utl::overloaded{
                    [&](additional_node::station const& s) {
                      auto const& st = provider.stations_.at(s.id_);
                      next_place.name_ = st.info_.name_;
                      rental.stationName_ = st.info_.name_;
                      rental.rentalUriAndroid_ = st.info_.rental_uris_.android_;
                      rental.rentalUriIOS_ = st.info_.rental_uris_.ios_;
                      rental.rentalUriWeb_ = st.info_.rental_uris_.web_;
                    },
                    [&](additional_node::vehicle const& v) {
                      auto const& vi = provider.vehicle_status_.at(v.idx_);
                      rental.rentalUriAndroid_ = vi.rental_uris_.android_;
                      rental.rentalUriIOS_ = vi.rental_uris_.ios_;
                      rental.rentalUriWeb_ = vi.rental_uris_.web_;
                    }},
                an.data_);
          }
        }

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
            .mode_ = lb->mode_ == osr::mode::kBike ? api::ModeEnum::BIKE_RENTAL
                                                   : api::ModeEnum::WALK,
            .from_ = pred_place,
            .to_ = next_place,
            .duration_ =
                std::chrono::duration_cast<std::chrono::seconds>(t - start_time)
                    .count(),
            .startTime_ = pred_end_time,
            .endTime_ = t,
            .distance_ = dist,
            .legGeometry_ = to_polyline<7>(concat),
            .steps_ = get_step_instructions(w, range),
            .rental_ = is_bike_leg ? std::optional{rental} : std::nullopt});
        leg.from_.departure_ = leg.startTime_;
        leg.to_.arrival_ = leg.endTime_;

        pred_place = next_place;
        pred_end_time = t;
      });

  return itinerary;
}

}  // namespace motis::gbfs