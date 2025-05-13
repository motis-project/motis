#include "motis/street_routing.h"

#include "geo/polyline_format.h"

#include "utl/concat.h"
#include "utl/get_or_create.h"

#include "osr/routing/route.h"
#include "osr/routing/sharing_data.h"

#include "motis/constants.h"
#include "motis/mode_to_profile.h"
#include "motis/place.h"
#include "motis/polyline.h"
#include "motis/update_rtt_td_footpaths.h"

namespace n = nigiri;

namespace motis {

default_output::~default_output() = default;

transport_mode_t default_output::get_cache_key(
    osr::search_profile const profile) const {
  return static_cast<transport_mode_t>(profile);
}

api::VertexTypeEnum default_output::get_vertex_type() const {
  return api::VertexTypeEnum::NORMAL;
}

std::string default_output::get_node_name(osr::node_idx_t) const { return ""; }

osr::sharing_data const* default_output::get_sharing_data() const {
  return nullptr;
}

geo::latlng default_output::get_node_pos(osr::node_idx_t) const {
  return geo::latlng{};
}

void default_output::annotate_leg(osr::node_idx_t,
                                  osr::node_idx_t,
                                  api::Leg&) const {}

default_output g_default_output = {};

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
  auto const cache_key =
      street_routing_cache_key_t{from, to, transport_mode, start_time};
  return utl::get_or_create(cache, cache_key, [&]() {
    auto const& [e_nodes, e_states] = *s;
    return osr::route(
        w, l, profile, from, to, max, osr::direction::kForward,
        max_matching_distance,
        s ? &set_blocked(e_nodes, e_states, blocked_mem) : nullptr, sharing,
        elevations);
  });
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

api::Itinerary street_routing(osr::ways const& w,
                              osr::lookup const& l,
                              output& out,
                              elevators const* e,
                              osr::elevation_storage const* elevations,
                              api::Place const& from,
                              api::Place const& to,
                              api::ModeEnum const mode,
                              osr::search_profile const profile,
                              n::unixtime_t const start_time,
                              std::optional<n::unixtime_t> const end_time,
                              double const max_matching_distance,
                              street_routing_cache_t& cache,
                              osr::bitvec<osr::node_idx_t>& blocked_mem,
                              unsigned const api_version,
                              std::chrono::seconds const max,
                              bool const dummy) {
  if (dummy) {
    return dummy_itinerary(from, to, mode, start_time, *end_time);
  }

  auto const rental_profile = osr::is_rental_profile(profile);
  auto const path =
      get_path(w, l, e, out.get_sharing_data(), elevations, get_location(from),
               get_location(to), out.get_cache_key(profile), profile,
               start_time, max_matching_distance,
               static_cast<osr::cost_t>(max.count()), cache, blocked_mem);

  if (!path.has_value()) {
    if (!end_time.has_value()) {
      return {};
    }
    std::cout << "ROUTING\n  FROM:  " << from << "     \n    TO:  " << to
              << "\n  -> CREATING DUMMY LEG (mode=" << mode
              << ", profile=" << osr::to_str(profile) << ")\n";
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
  auto& pred_place = from;
  auto pred_end_time = t;
  utl::equal_ranges_linear(
      path->segments_,
      [](osr::path::segment const& a, osr::path::segment const& b) {
        return a.mode_ == b.mode_;
      },
      [&](std::vector<osr::path::segment>::const_iterator const& lb,
          std::vector<osr::path::segment>::const_iterator const& ub) {
        auto const range = std::span{lb, ub};
        auto const is_last_leg = ub == end(path->segments_);
        auto const from_node = range.front().from_;
        auto const to_node = range.back().to_;

        auto const to_pos = get_node_pos(to_node);
        auto const next_place = out.annotate_place(
            api::Place{.lat_ = to_pos.lat_, .lon_ = to_pos.lng_});

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
            .mode_ = mode,
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
                w, get_location(from), get_location(to), range, api_version)});

        out.annotate_leg(from_node, to_node, leg);

        leg.from_.departure_ = leg.from_.scheduledDeparture_ =
            leg.scheduledStartTime_ = leg.startTime_;
        leg.to_.arrival_ = leg.to_.scheduledArrival_ = leg.scheduledEndTime_ =
            leg.endTime_;

        pred_place = leg.to_;
        pred_end_time = t;
      });

  if (end_time && !itinerary.legs_.empty()) {
    auto& last = itinerary.legs_.back();
    last.to_.arrival_ = last.to_.scheduledArrival_ = last.endTime_ =
        last.scheduledEndTime_ = *end_time;
    for (auto& leg : itinerary.legs_) {
      leg.duration_ = (leg.endTime_.time_ - leg.startTime_.time_).count();
    }
  }

  return itinerary;
}

}  // namespace motis
