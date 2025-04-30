#include "motis/street_routing.h"

#include "geo/polyline_format.h"

#include "utl/concat.h"

#include "osr/routing/route.h"
#include "osr/routing/sharing_data.h"

#include "motis/constants.h"
#include "motis/gbfs/mode.h"
#include "motis/gbfs/routing_data.h"
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
                                  osr::elevation_storage const* elevations,
                                  osr::location const& from,
                                  osr::location const& to,
                                  transport_mode_t const transport_mode,
                                  osr::search_profile const profile,
                                  nigiri::unixtime_t const start_time,
                                  double const max_matching_distance,
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
                max_matching_distance,
                s ? &set_blocked(e_nodes, e_states, blocked_mem) : nullptr,
                sharing, elevations);
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
    osr::ways const& w,
    osr::location const& from,
    osr::location const& to,
    std::span<osr::path::segment const> segments,
    unsigned const api_version) {
  auto steps = std::vector<api::StepInstruction>{};
  auto pred_lvl = from.lvl_.to_float();
  for (auto const& s : segments) {
    if (s.from_ != osr::node_idx_t::invalid() && s.from_ < w.n_nodes() &&
        w.r_->node_properties_[s.from_].is_elevator()) {
      steps.push_back(api::StepInstruction{
          .relativeDirection_ = api::DirectionEnum::ELEVATOR,
          .fromLevel_ = pred_lvl,
          .toLevel_ = s.from_level_.to_float()});
    }

    auto const way_name = s.way_ == osr::way_idx_t::invalid()
                              ? osr::string_idx_t::invalid()
                              : w.way_names_[s.way_];
    auto const props = s.way_ != osr::way_idx_t::invalid()
                           ? w.r_->way_properties_[s.way_]
                           : osr::way_properties{};
    steps.push_back(api::StepInstruction{
        .relativeDirection_ =
            s.way_ != osr::way_idx_t::invalid()
                ? (props.is_elevator() ? api::DirectionEnum::ELEVATOR
                   : props.is_steps()  ? api::DirectionEnum::STAIRS
                                       : api::DirectionEnum::CONTINUE)
                : api::DirectionEnum::CONTINUE,  // TODO entry/exit/u-turn
        .distance_ = static_cast<double>(s.dist_),
        .fromLevel_ = s.from_level_.to_float(),
        .toLevel_ = s.to_level_.to_float(),
        .osmWay_ = s.way_ == osr::way_idx_t ::invalid()
                       ? std::nullopt
                       : std::optional{static_cast<std::int64_t>(
                             to_idx(w.way_osm_idx_[s.way_]))},
        .polyline_ = api_version == 1 ? to_polyline<7>(s.polyline_)
                                      : to_polyline<6>(s.polyline_),
        .streetName_ = way_name == osr::string_idx_t::invalid()
                           ? ""
                           : std::string{w.strings_[way_name].view()},
        .exit_ = {},  // TODO
        .stayOn_ = false,  // TODO
        .area_ = false  // TODO
    });
  }

  if (!segments.empty()) {
    auto& last = segments.back();
    if (last.to_ != osr::node_idx_t::invalid() && last.to_ < w.n_nodes() &&
        w.r_->node_properties_[last.to_].is_elevator()) {
      steps.push_back(api::StepInstruction{
          .relativeDirection_ = api::DirectionEnum::ELEVATOR,
          .fromLevel_ = pred_lvl,
          .toLevel_ = to.lvl_.to_float()});
    }
  }

  return steps;
}

bool is_additional_node(osr::ways const& w, osr::node_idx_t const n) {
  return n != osr::node_idx_t::invalid() && n >= w.n_nodes();
}

struct sharing {
  sharing(osr::ways const& w,
          gbfs::gbfs_routing_data& gbfs_rd,
          gbfs::gbfs_products_ref const prod_ref)
      : w_{w},
        gbfs_rd_{gbfs_rd},
        provider_{*gbfs_rd_.data_->providers_.at(prod_ref.provider_)},
        products_{provider_.products_.at(prod_ref.products_)},
        prod_rd_{gbfs_rd_.get_products_routing_data(prod_ref)} {}

  api::Rental get_rental(osr::node_idx_t const from_node,
                         osr::node_idx_t const to_node) const {
    auto ret = rental_;
    if (is_additional_node(w_, from_node)) {
      auto const& an = prod_rd_->compressed_.additional_nodes_.at(
          get_additional_node_idx(from_node));
      std::visit(
          utl::overloaded{
              [&](gbfs::additional_node::station const& s) {
                auto const& st = provider_.stations_.at(s.id_);
                ret.fromStationName_ = st.info_.name_;
                ret.stationName_ = st.info_.name_;
                ret.rentalUriAndroid_ = st.info_.rental_uris_.android_;
                ret.rentalUriIOS_ = st.info_.rental_uris_.ios_;
                ret.rentalUriWeb_ = st.info_.rental_uris_.web_;
              },
              [&](gbfs::additional_node::vehicle const& v) {
                auto const& vs = provider_.vehicle_status_.at(v.idx_);
                if (auto const st = provider_.stations_.find(vs.station_id_);
                    st != end(provider_.stations_)) {
                  ret.fromStationName_ = st->second.info_.name_;
                }
                ret.rentalUriAndroid_ = vs.rental_uris_.android_;
                ret.rentalUriIOS_ = vs.rental_uris_.ios_;
                ret.rentalUriWeb_ = vs.rental_uris_.web_;
              }},
          an.data_);
    }
    if (is_additional_node(w_, to_node)) {
      auto const& an = prod_rd_->compressed_.additional_nodes_.at(
          get_additional_node_idx(to_node));
      std::visit(
          utl::overloaded{
              [&](gbfs::additional_node::station const& s) {
                auto const& st = provider_.stations_.at(s.id_);
                ret.toStationName_ = st.info_.name_;
                if (!ret.stationName_) {
                  ret.stationName_ = ret.toStationName_;
                }
              },
              [&](gbfs::additional_node::vehicle const& v) {
                auto const& vs = provider_.vehicle_status_.at(v.idx_);
                if (auto const st = provider_.stations_.find(vs.station_id_);
                    st != end(provider_.stations_)) {
                  ret.toStationName_ = st->second.info_.name_;
                }
              }},
          an.data_);
    }
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
        prod_rd_->compressed_.additional_nodes_.at(get_additional_node_idx(n))
            .data_);
  }

  std::string get_node_name(osr::node_idx_t const n) const {
    if (!is_additional_node(w_, n)) {
      return provider_.sys_info_.name_;
    }
    auto const& an =
        prod_rd_->compressed_.additional_nodes_.at(get_additional_node_idx(n));
    return std::visit(
        utl::overloaded{[&](gbfs::additional_node::station const& s) {
                          auto const& st = provider_.stations_.at(s.id_);
                          return st.info_.name_;
                        },
                        [&](gbfs::additional_node::vehicle const& v) {
                          auto const& vs = provider_.vehicle_status_.at(v.idx_);
                          if (auto const st =
                                  provider_.stations_.find(vs.station_id_);
                              st != end(provider_.stations_)) {
                            return st->second.info_.name_;
                          }
                          return provider_.sys_info_.name_;
                        }},
        an.data_);
  }

  std::size_t get_additional_node_idx(osr::node_idx_t const n) const {
    return to_idx(n) - sharing_data_.additional_node_offset_;
  }

  osr::ways const& w_;
  gbfs::gbfs_routing_data& gbfs_rd_;
  gbfs::gbfs_provider const& provider_;
  gbfs::provider_products const& products_;
  gbfs::products_routing_data const* prod_rd_;
  osr::sharing_data sharing_data_{
      .start_allowed_ = prod_rd_->start_allowed_,
      .end_allowed_ = prod_rd_->end_allowed_,
      .through_allowed_ = prod_rd_->through_allowed_,
      .additional_node_offset_ = w_.n_nodes(),
      .additional_edges_ = prod_rd_->compressed_.additional_edges_};
  api::Rental rental_{
      .systemId_ = provider_.sys_info_.id_,
      .systemName_ = provider_.sys_info_.name_,
      .url_ = provider_.sys_info_.url_,
      .formFactor_ = gbfs::to_api_form_factor(products_.form_factor_),
      .propulsionType_ =
          gbfs::to_api_propulsion_type(products_.propulsion_type_),
      .returnConstraint_ =
          gbfs::to_api_return_constraint(products_.return_constraint_)};
};

api::Itinerary dummy_itinerary(api::Place const& from,
                               api::Place const& to,
                               api::ModeEnum const mode,
                               n::unixtime_t const start_time,
                               n::unixtime_t const end_time) {
  auto itinerary = api::Itinerary{
      .duration_ = std::chrono::duration_cast<std::chrono::seconds>(end_time -
                                                                    start_time)
                       .count(),
      .startTime_ = start_time,
      .endTime_ = end_time};
  auto& leg = itinerary.legs_.emplace_back(api::Leg{
      .mode_ = mode,
      .from_ = from,
      .to_ = to,
      .duration_ = std::chrono::duration_cast<std::chrono::seconds>(end_time -
                                                                    start_time)
                       .count(),
      .startTime_ = start_time,
      .endTime_ = end_time,
      .scheduledStartTime_ = start_time,
      .scheduledEndTime_ = end_time});
  leg.from_.departure_ = leg.from_.scheduledDeparture_ = leg.startTime_;
  leg.to_.arrival_ = leg.to_.scheduledArrival_ = leg.endTime_;
  return itinerary;
}

api::Itinerary route(osr::ways const& w,
                     osr::lookup const& l,
                     gbfs::gbfs_routing_data& gbfs_rd,
                     elevators const* e,
                     osr::elevation_storage const* elevations,
                     api::Place const& from,
                     api::Place const& to,
                     api::ModeEnum const mode,
                     osr::search_profile const profile,
                     n::unixtime_t const start_time,
                     std::optional<n::unixtime_t> const end_time,
                     double const max_matching_distance,
                     gbfs::gbfs_products_ref const prod_ref,
                     street_routing_cache_t& cache,
                     osr::bitvec<osr::node_idx_t>& blocked_mem,
                     unsigned const api_version,
                     std::chrono::seconds const max,
                     bool const dummy) {
  if (dummy) {
    return dummy_itinerary(from, to, mode, start_time, *end_time);
  }

  auto const rental_profile = osr::is_rental_profile(profile);
  utl::verify(!rental_profile || gbfs_rd.has_data(),
              "sharing mobility not configured");

  auto const sharing_data = rental_profile
                                ? std::optional{sharing(w, gbfs_rd, prod_ref)}
                                : std::nullopt;

  auto const get_node_pos = [&](osr::node_idx_t const n) -> geo::latlng {
    if (n == osr::node_idx_t::invalid()) {
      return {};
    } else if (!is_additional_node(w, n)) {
      return w.get_node_pos(n).as_latlng();
    } else {
      return sharing_data.value().get_node_pos(n);
    }
  };

  auto const transport_mode =
      rental_profile
          ? static_cast<transport_mode_t>(gbfs_rd.get_transport_mode(prod_ref))
          : static_cast<transport_mode_t>(profile);

  auto const path = [&]() {
    auto p =
        get_path(w, l, e, sharing_data ? &sharing_data->sharing_data_ : nullptr,
                 elevations, get_location(from), get_location(to),
                 transport_mode, profile, start_time, max_matching_distance,
                 static_cast<osr::cost_t>(max.count()), cache, blocked_mem);

    if (p.has_value() && rental_profile) {
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
    std::cout << "ROUTING\n  FROM:  " << from << "     \n    TO:  " << to
              << "\n  -> CREATING DUMMY LEG (mode=" << mode
              << ", profile=" << osr::to_str(profile)
              << ", provider=" << prod_ref.provider_
              << ", products=" << prod_ref.products_ << ")\n";
    return dummy_itinerary(from, to, mode, start_time, *end_time);
  }

  auto itinerary = api::Itinerary{
      .duration_ = end_time ? std::chrono::duration_cast<std::chrono::seconds>(
                                  *end_time - start_time)
                                  .count()
                            : path->cost_,
      .startTime_ = start_time,
      .endTime_ =
          end_time ? *end_time : start_time + std::chrono::seconds{path->cost_},
      .transfers_ = 0};

  auto t = std::chrono::time_point_cast<std::chrono::seconds>(start_time);
  auto pred_place = from;
  auto pred_end_time = t;
  utl::equal_ranges_linear(
      path->segments_, [](auto&& a, auto&& b) { return a.mode_ == b.mode_; },
      [&](auto&& lb, auto&& ub) {
        auto const range = std::span{lb, ub};
        auto const is_last_leg = ub == end(path->segments_);
        auto const from_node = range.front().from_;
        auto const from_additional_node = is_additional_node(w, from_node);
        auto const to_node = range.back().to_;
        auto const to_additional_node = is_additional_node(w, to_node);
        auto const is_rental =
            (rental_profile &&
             (lb->mode_ == osr::mode::kBike || lb->mode_ == osr::mode::kCar) &&
             (from_additional_node || to_additional_node));

        auto const to_pos = get_node_pos(to_node);
        auto const next_place =
            is_last_leg ? to
            // All modes except sharing mobility have only one leg.
            // -> This is not the last leg = it has to be sharing mobility.
            : profile == osr::search_profile::kBikeSharing
                ? api::Place{.name_ = sharing_data->get_node_name(to_node),
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
            .mode_ = is_rental                    ? api::ModeEnum::RENTAL
                     : mode == api::ModeEnum::ODM ? mode
                                                  : to_mode(lb->mode_),
            .from_ = pred_place,
            .to_ = next_place,
            .duration_ = std::chrono::duration_cast<std::chrono::seconds>(
                             t - pred_end_time)
                             .count(),
            .startTime_ = pred_end_time,
            .endTime_ = is_last_leg && end_time ? *end_time : t,
            .distance_ = dist,
            .legGeometry_ = api_version == 1 ? to_polyline<7>(concat)
                                             : to_polyline<6>(concat),
            .steps_ = get_step_instructions(
                w, get_location(from), get_location(to), range, api_version),
            .rental_ = is_rental ? std::optional{sharing_data->get_rental(
                                       from_node, to_node)}
                                 : std::nullopt});

        leg.from_.departure_ = leg.from_.scheduledDeparture_ =
            leg.scheduledStartTime_ = leg.startTime_;
        leg.to_.arrival_ = leg.to_.scheduledArrival_ = leg.scheduledEndTime_ =
            leg.endTime_;

        pred_place = next_place;
        pred_end_time = t;
      });

  if (end_time && !itinerary.legs_.empty()) {
    itinerary.legs_.back().to_.arrival_ =
        itinerary.legs_.back().to_.scheduledArrival_ =
            itinerary.legs_.back().endTime_ =
                itinerary.legs_.back().scheduledEndTime_ = *end_time;
    for (auto& leg : itinerary.legs_) {
      leg.duration_ = (leg.endTime_.time_ - leg.startTime_.time_).count();
    }
  }

  return itinerary;
}

}  // namespace motis
