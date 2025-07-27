#include "motis/street_routing.h"

#include "geo/polyline_format.h"

#include "utl/concat.h"
#include "utl/get_or_create.h"

#include "osr/routing/algorithms.h"
#include "osr/routing/route.h"
#include "osr/routing/sharing_data.h"

#include "motis/constants.h"
#include "motis/mode_to_profile.h"
#include "motis/place.h"
#include "motis/polyline.h"
#include "motis/transport_mode_ids.h"
#include "motis/update_rtt_td_footpaths.h"
#include "utl/verify.h"

namespace n = nigiri;

namespace motis {

default_output::default_output(osr::ways const& w,
                               osr::search_profile const profile)
    : w_{w},
      profile_{profile},
      id_{static_cast<std::underlying_type_t<osr::search_profile>>(profile)} {}

default_output::default_output(osr::ways const& w,
                               nigiri::transport_mode_id_t const id)
    : w_{w},
      profile_{id == kOdmTransportModeId
                   ? osr::search_profile::kCar
                   : osr::search_profile{static_cast<
                         std::underlying_type_t<osr::search_profile>>(id)}},
      id_{id} {
  utl::verify(id <= kOdmTransportModeId, "invalid mode id={}", id);
}

default_output::~default_output() = default;

api::ModeEnum default_output::get_mode() const {
  if (id_ == kOdmTransportModeId) {
    return api::ModeEnum::ODM;
  }

  switch (profile_) {
    case osr::search_profile::kFoot: [[fallthrough]];
    case osr::search_profile::kWheelchair: return api::ModeEnum::WALK;
    case osr::search_profile::kBike: [[fallthrough]];
    case osr::search_profile::kBikeElevationLow: [[fallthrough]];
    case osr::search_profile::kBikeElevationHigh: return api::ModeEnum::BIKE;
    case osr::search_profile::kCar: return api::ModeEnum::CAR;
    case osr::search_profile::kCarParking: [[fallthrough]];
    case osr::search_profile::kCarParkingWheelchair:
      return api::ModeEnum::CAR_PARKING;
    case osr::search_profile::kCarDropOff: [[fallthrough]];
    case osr::search_profile::kCarDropOffWheelchair:
      return api::ModeEnum::CAR_DROPOFF;
    case osr::search_profile::kBikeSharing: [[fallthrough]];
    case osr::search_profile::kCarSharing: return api::ModeEnum::RENTAL;
  }

  return api::ModeEnum::OTHER;
}

osr::search_profile default_output::get_profile() const { return profile_; }

api::Place default_output::get_place(osr::node_idx_t const n) const {
  auto const pos = w_.get_node_pos(n).as_latlng();
  return api::Place{.lat_ = pos.lat_,
                    .lon_ = pos.lng_,
                    .vertexType_ = api::VertexTypeEnum::NORMAL};
}

bool default_output::is_time_dependent() const {
  return profile_ == osr::search_profile::kWheelchair ||
         profile_ == osr::search_profile::kCarParkingWheelchair ||
         profile_ == osr::search_profile::kCarDropOffWheelchair;
}

transport_mode_t default_output::get_cache_key() const {
  return static_cast<transport_mode_t>(profile_);
}

osr::sharing_data const* default_output::get_sharing_data() const {
  return nullptr;
}

void default_output::annotate_leg(osr::node_idx_t,
                                  osr::node_idx_t,
                                  api::Leg&) const {}

std::vector<api::StepInstruction> get_step_instructions(
    osr::ways const& w,
    osr::elevation_storage const* elevations,
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
        .area_ = false,  // TODO
        .toll_ = props.has_toll(),
        .accessRestriction_ = w.get_access_restriction(s.way_).and_then(
            [](std::string_view s) { return std::optional{std::string{s}}; }),
        .elevationUp_ =
            elevations ? std::optional{to_idx(s.elevation_.up_)} : std::nullopt,
        .elevationDown_ = elevations ? std::optional{to_idx(s.elevation_.down_)}
                                     : std::nullopt});
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
  leg.from_.pickupType_ = std::nullopt;
  leg.from_.dropoffType_ = std::nullopt;
  leg.to_.pickupType_ = std::nullopt;
  leg.to_.dropoffType_ = std::nullopt;
  leg.from_.departure_ = leg.from_.scheduledDeparture_ = leg.startTime_;
  leg.to_.arrival_ = leg.to_.scheduledArrival_ = leg.endTime_;
  return itinerary;
}

api::Itinerary street_routing(osr::ways const& w,
                              osr::lookup const& l,
                              elevators const* e,
                              osr::elevation_storage const* elevations,
                              api::Place const& from_place,
                              api::Place const& to_place,
                              output const& out,
                              std::optional<n::unixtime_t> const start_time,
                              std::optional<n::unixtime_t> const end_time,
                              double const max_matching_distance,
                              street_routing_cache_t& cache,
                              osr::bitvec<osr::node_idx_t>& blocked_mem,
                              unsigned const api_version,
                              std::chrono::seconds const max) {
  utl::verify(start_time.has_value() || end_time.has_value(),
              "either start_time or end_time must be set");
  auto const bound_time =
      start_time.or_else([&]() { return end_time; }).value();
  auto const from = get_location(from_place);
  auto const to = get_location(to_place);
  auto const s = e ? get_states_at(w, l, *e, bound_time, from.pos_)
                   : std::optional{std::pair<nodes_t, states_t>{}};
  auto const cache_key = street_routing_cache_key_t{
      from, to, out.get_cache_key(),
      out.is_time_dependent() ? bound_time : n::unixtime_t{n::i32_minutes{0}}};
  auto const path = utl::get_or_create(cache, cache_key, [&]() {
    auto const& [e_nodes, e_states] = *s;
    return osr::route(
        w, l, out.get_profile(), from, to,
        static_cast<osr::cost_t>(max.count()), osr::direction::kForward,
        max_matching_distance,
        s ? &set_blocked(e_nodes, e_states, blocked_mem) : nullptr,
        out.get_sharing_data(), elevations, osr::routing_algorithm::kAStarBi);
  });

  if (!path.has_value()) {
    if (!start_time.has_value() || !end_time.has_value()) {
      return {};
    }
    std::cout << "ROUTING\n  FROM:  " << from << "     \n    TO:  " << to
              << "\n  -> CREATING DUMMY LEG (mode=" << out.get_mode()
              << ", profile=" << to_str(out.get_profile()) << ")\n";
    return dummy_itinerary(from_place, to_place, out.get_mode(), *start_time,
                           *end_time);
  }

  auto const deduced_start_time =
      start_time ? *start_time : *end_time - std::chrono::seconds{path->cost_};
  auto itinerary = api::Itinerary{
      .duration_ = start_time && end_time
                       ? std::chrono::duration_cast<std::chrono::seconds>(
                             *end_time - *start_time)
                             .count()
                       : path->cost_,
      .startTime_ = deduced_start_time,
      .endTime_ = end_time ? *end_time
                           : *start_time + std::chrono::seconds{path->cost_},
      .transfers_ = 0};

  auto t =
      std::chrono::time_point_cast<std::chrono::seconds>(deduced_start_time);
  auto pred_place = from_place;
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
            .mode_ = out.get_mode() == api::ModeEnum::ODM ? api::ModeEnum::ODM
                                                          : to_mode(lb->mode_),
            .from_ = pred_place,
            .to_ = is_last_leg ? to_place : out.get_place(to_node),
            .duration_ = std::chrono::duration_cast<std::chrono::seconds>(
                             t - pred_end_time)
                             .count(),
            .startTime_ = pred_end_time,
            .endTime_ = is_last_leg && end_time ? *end_time : t,
            .distance_ = dist,
            .legGeometry_ = api_version == 1 ? to_polyline<7>(concat)
                                             : to_polyline<6>(concat),
            .steps_ = get_step_instructions(w, elevations, from, to, range,
                                            api_version)});

        leg.from_.departure_ = leg.from_.scheduledDeparture_ =
            leg.scheduledStartTime_ = leg.startTime_;
        leg.to_.arrival_ = leg.to_.scheduledArrival_ = leg.scheduledEndTime_ =
            leg.endTime_;
        leg.from_.pickupType_ = std::nullopt;
        leg.from_.dropoffType_ = std::nullopt;
        leg.to_.pickupType_ = std::nullopt;
        leg.to_.dropoffType_ = std::nullopt;

        out.annotate_leg(from_node, to_node, leg);

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
