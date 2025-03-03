#include "motis/journey_to_response.h"

#include <cmath>
#include <iostream>
#include <span>

#include "utl/enumerate.h"
#include "utl/overloaded.h"

#include "geo/polyline_format.h"

#include "nigiri/common/split_duration.h"
#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/special_stations.h"
#include "nigiri/types.h"

#include "motis/constants.h"
#include "motis/gbfs/mode.h"
#include "motis/gbfs/routing_data.h"
#include "motis/odm/odm.h"
#include "motis/place.h"
#include "motis/street_routing.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/clasz_to_mode.h"
#include "motis/timetable/time_conv.h"
#include "motis/transport_mode_ids.h"

namespace n = nigiri;

namespace motis {

api::ModeEnum to_mode(osr::search_profile const m) {
  switch (m) {
    case osr::search_profile::kCarParkingWheelchair: [[fallthrough]];
    case osr::search_profile::kCarParking: return api::ModeEnum::CAR_PARKING;
    case osr::search_profile::kFoot: [[fallthrough]];
    case osr::search_profile::kWheelchair: return api::ModeEnum::WALK;
    case osr::search_profile::kCar: return api::ModeEnum::CAR;
    case osr::search_profile::kBikeElevationLow:
    case osr::search_profile::kBikeElevationHigh: [[fallthrough]];
    case osr::search_profile::kBike: return api::ModeEnum::BIKE;
    case osr::search_profile::kBikeSharing: return api::ModeEnum::RENTAL;
  }
  std::unreachable();
}

void cleanup_intermodal(api::Itinerary& i) {
  if (i.legs_.front().from_.name_ == "END") {
    i.legs_.front().from_.name_ = "START";
  }
  if (i.legs_.back().to_.name_ == "START") {
    i.legs_.back().to_.name_ = "END";
  }
}

struct fare_indices {
  std::int64_t transfer_idx_;
  std::int64_t effective_fare_leg_idx_;
};

std::optional<fare_indices> get_fare_indices(
    std::optional<std::vector<n::fare_transfer>> const& fares,
    n::routing::journey::leg const& l) {
  if (!fares.has_value()) {
    return std::nullopt;
  }

  for (auto const [transfer_idx, transfer] : utl::enumerate(*fares)) {
    for (auto const [eff_fare_leg_idx, eff_fare_leg] :
         utl::enumerate(transfer.legs_)) {
      for (auto const* x : eff_fare_leg.joined_leg_) {
        if (x == &l) {
          return fare_indices{static_cast<std::int64_t>(transfer_idx),
                              static_cast<std::int64_t>(eff_fare_leg_idx)};
        }
      }
    }
  }

  return std::nullopt;
}

api::Itinerary journey_to_response(osr::ways const* w,
                                   osr::lookup const* l,
                                   osr::platforms const* pl,
                                   n::timetable const& tt,
                                   tag_lookup const& tags,
                                   elevators const* e,
                                   n::rt_timetable const* rtt,
                                   platform_matches_t const* matches,
                                   n::shapes_storage const* shapes,
                                   gbfs::gbfs_routing_data& gbfs_rd,
                                   bool const wheelchair,
                                   n::routing::journey const& j,
                                   place_t const& start,
                                   place_t const& dest,
                                   street_routing_cache_t& cache,
                                   osr::bitvec<osr::node_idx_t>& blocked_mem,
                                   bool const detailed_transfers,
                                   bool const with_fares,
                                   double const timetable_max_matching_distance,
                                   double const max_matching_distance) {
  utl::verify(!j.legs_.empty(), "journey without legs");

  auto const fares =
      with_fares ? std::optional{n::get_fares(tt, j)} : std::nullopt;
  auto const to_fare_media_type =
      [](n::fares::fare_media::fare_media_type const t) {
        using fare_media_type = n::fares::fare_media::fare_media_type;
        switch (t) {
          case fare_media_type::kNone: return api::FareMediaTypeEnum::NONE;
          case fare_media_type::kPaper:
            return api::FareMediaTypeEnum::PAPER_TICKET;
          case fare_media_type::kCard:
            return api::FareMediaTypeEnum::TRANSIT_CARD;
          case fare_media_type::kContactless:
            return api::FareMediaTypeEnum::CONTACTLESS_EMV;
          case fare_media_type::kApp: return api::FareMediaTypeEnum::MOBILE_APP;
        }
        std::unreachable();
      };
  auto const to_media = [&](n::fares::fare_media const& m) -> api::FareMedia {
    return {.fareMediaName_ =
                m.name_ == n::string_idx_t::invalid()
                    ? std::nullopt
                    : std::optional{std::string{tt.strings_.get(m.name_)}},
            .fareMediaType_ = to_fare_media_type(m.type_)};
  };
  auto const to_rider_category =
      [&](n::fares::rider_category const& r) -> api::RiderCategory {
    return {
        .riderCategoryName_ = std::string{tt.strings_.get(r.eligibility_url_)},
        .isDefaultFareCategory_ = r.is_default_fare_category_,
        .eligibilityUrl_ = tt.strings_.try_get(r.eligibility_url_)
                               .and_then([](std::string_view s) {
                                 return std::optional{std::string{s}};
                               })};
  };
  auto const to_product =
      [&](n::fares const& f,
          n::fare_product_idx_t const x) -> api::FareProduct {
    auto const& p = f.fare_products_[x];
    return {.name_ = std::string{tt.strings_.get(p.name_)},
            .amount_ = p.amount_,
            .currency_ = std::string{tt.strings_.get(p.currency_code_)},
            .riderCategory_ =
                p.rider_category_ == n::rider_category_idx_t::invalid()
                    ? std::nullopt
                    : std::optional{to_rider_category(
                          f.rider_categories_[p.rider_category_])},
            .media_ = p.media_ == n::fare_media_idx_t::invalid()
                          ? std::nullopt
                          : std::optional{to_media(f.fare_media_[p.media_])}};
  };
  auto const to_rule = [](n::fares::fare_transfer_rule const& x) {
    switch (x.fare_transfer_type_) {
      case nigiri::fares::fare_transfer_rule::fare_transfer_type::kAB:
        return api::FareTransferRuleEnum::AB;
      case nigiri::fares::fare_transfer_rule::fare_transfer_type::kAPlusAB:
        return api::FareTransferRuleEnum::A_AB;
      case nigiri::fares::fare_transfer_rule::fare_transfer_type::kAPlusABPlusB:
        return api::FareTransferRuleEnum::A_AB_B;
    }
    std::unreachable();
  };

  auto itinerary = api::Itinerary{
      .duration_ = to_seconds(j.arrival_time() - j.departure_time()),
      .startTime_ = j.legs_.front().dep_time_,
      .endTime_ = j.legs_.back().arr_time_,
      .transfers_ = std::max(
          static_cast<std::iterator_traits<
              decltype(j.legs_)::iterator>::difference_type>(0),
          utl::count_if(
              j.legs_,
              [](auto&& leg) {
                return holds_alternative<n::routing::journey::run_enter_exit>(
                           leg.uses_) ||
                       odm::is_odm_leg(leg);
              }) -
              1),
      .fareTransfers_ =
          fares.and_then([&](std::vector<n::fare_transfer> const& transfers) {
            return std::optional{utl::to_vec(
                transfers, [&](n::fare_transfer const& t) -> api::FareTransfer {
                  return {.rule_ = t.rule_.and_then([&](auto&& r) {
                            return std::optional{to_rule(r)};
                          }),
                          .transferProduct_ = t.rule_.and_then([&](auto&& r) {
                            return t.legs_.empty()
                                       ? std::nullopt
                                       : std::optional{to_product(
                                             tt.fares_[t.legs_.front().src_],
                                             r.fare_product_)};
                          }),
                          .effectiveFareLegProducts_ =
                              utl::to_vec(t.legs_, [&](auto&& l) {
                                return utl::to_vec(l.rule_, [&](auto&& r) {
                                  return to_product(tt.fares_[l.src_],
                                                    r.fare_product_);
                                });
                              })};
                })};
          })};

  auto const append = [&](api::Itinerary&& x) {
    itinerary.legs_.insert(end(itinerary.legs_),
                           std::move_iterator{begin(x.legs_)},
                           std::move_iterator{end(x.legs_)});
  };

  for (auto const [_, j_leg] : utl::enumerate(j.legs_)) {
    auto const pred =
        itinerary.legs_.empty() ? nullptr : &itinerary.legs_.back();
    auto const from = pred == nullptr
                          ? to_place(&tt, &tags, w, pl, matches,
                                     tt_location{j_leg.from_}, start, dest)
                          : pred->to_;
    auto const to = to_place(&tt, &tags, w, pl, matches, tt_location{j_leg.to_},
                             start, dest);

    auto const to_place = [&](auto&& l) {
      return ::motis::to_place(&tt, &tags, w, pl, matches, l, start, dest);
    };

    std::visit(
        utl::overloaded{
            [&](n::routing::journey::run_enter_exit const& t) {
              // TODO interlining
              auto const fr = n::rt::frun{tt, rtt, t.r_};
              auto const enter_stop = fr[t.stop_range_.from_];
              auto const exit_stop = fr[t.stop_range_.to_ - 1U];
              auto const color = enter_stop.get_route_color();
              auto const agency = enter_stop.get_provider();
              auto const fare_indices = get_fare_indices(fares, j_leg);

              auto& leg = itinerary.legs_.emplace_back(api::Leg{
                  .mode_ = to_mode(enter_stop.get_clasz()),
                  .from_ = to_place(tt_location{enter_stop}),
                  .to_ = to_place(tt_location{exit_stop}),
                  .duration_ = std::chrono::duration_cast<std::chrono::seconds>(
                                   j_leg.arr_time_ - j_leg.dep_time_)
                                   .count(),
                  .startTime_ = j_leg.dep_time_,
                  .endTime_ = j_leg.arr_time_,
                  .scheduledStartTime_ =
                      enter_stop.scheduled_time(n::event_type::kDep),
                  .scheduledEndTime_ =
                      exit_stop.scheduled_time(n::event_type::kArr),
                  .realTime_ = fr.is_rt(),
                  .headsign_ = std::string{enter_stop.direction()},
                  .routeColor_ = to_str(color.color_),
                  .routeTextColor_ = to_str(color.text_color_),
                  .agencyName_ = {std::string{agency.long_name_}},
                  .agencyUrl_ = {std::string{agency.url_}},
                  .agencyId_ = {std::string{agency.short_name_}},
                  .tripId_ = tags.id(tt, enter_stop, n::event_type::kDep),
                  .routeShortName_ = {std::string{
                      enter_stop.trip_display_name()}},
                  .source_ = fmt::to_string(fr.dbg()),
                  .fareTransferIndex_ = fare_indices.and_then(
                      [](auto&& x) { return std::optional{x.transfer_idx_}; }),
                  .effectiveFareLegIndex_ = fare_indices.and_then([](auto&& x) {
                    return std::optional{x.effective_fare_leg_idx_};
                  })});
              leg.from_.vertexType_ = api::VertexTypeEnum::TRANSIT;
              leg.from_.departure_ = leg.startTime_;
              leg.from_.scheduledDeparture_ = leg.scheduledStartTime_;
              leg.to_.vertexType_ = api::VertexTypeEnum::TRANSIT;
              leg.to_.arrival_ = leg.endTime_;
              leg.to_.scheduledArrival_ = leg.scheduledEndTime_;

              auto polyline = geo::polyline{};
              fr.for_each_shape_point(
                  shapes, t.stop_range_,
                  [&](geo::latlng const& pos) { polyline.emplace_back(pos); });
              leg.legGeometry_.points_ = geo::encode_polyline<7>(polyline);
              leg.legGeometry_.length_ =
                  static_cast<std::int64_t>(polyline.size());

              auto const first =
                  static_cast<n::stop_idx_t>(t.stop_range_.from_ + 1U);
              auto const last =
                  static_cast<n::stop_idx_t>(t.stop_range_.to_ - 1U);
              leg.intermediateStops_ = std::vector<api::Place>{};
              for (auto i = first; i < last; ++i) {
                auto const stop = fr[i];
                if (stop.is_canceled()) {
                  continue;
                }
                auto& p = leg.intermediateStops_->emplace_back(
                    to_place(tt_location{stop}));
                p.departure_ = stop.time(n::event_type::kDep);
                p.scheduledDeparture_ =
                    stop.scheduled_time(n::event_type::kDep);
                p.arrival_ = stop.time(n::event_type::kArr);
                p.scheduledArrival_ = stop.scheduled_time(n::event_type::kArr);
              }
            },
            [&](n::footpath) {
              append(
                  w && l
                      ? route(*w, *l, gbfs_rd, e, from, to, api::ModeEnum::WALK,
                              wheelchair, j_leg.dep_time_, j_leg.arr_time_,
                              timetable_max_matching_distance, {}, cache,
                              blocked_mem,
                              std::chrono::duration_cast<std::chrono::seconds>(
                                  j_leg.arr_time_ - j_leg.dep_time_) +
                                  std::chrono::minutes{10},
                              !detailed_transfers)
                      : dummy_itinerary(from, to, api::ModeEnum::WALK,
                                        j_leg.dep_time_, j_leg.arr_time_));
            },
            [&](n::routing::offset const x) {
              append(route(
                  *w, *l, gbfs_rd, e, from, to,
                  x.transport_mode_id_ >= kGbfsTransportModeIdOffset
                      ? api::ModeEnum::RENTAL
                  : x.transport_mode_id_ == kOdmTransportModeId
                      ? api::ModeEnum::ODM
                      : to_mode(osr::search_profile{
                            static_cast<std::uint8_t>(x.transport_mode_id_)}),
                  wheelchair, j_leg.dep_time_, j_leg.arr_time_,
                  max_matching_distance,
                  x.transport_mode_id_ >= kGbfsTransportModeIdOffset
                      ? gbfs_rd.get_products_ref(x.transport_mode_id_)
                      : gbfs::gbfs_products_ref{},
                  cache, blocked_mem,
                  std::chrono::duration_cast<std::chrono::seconds>(
                      j_leg.arr_time_ - j_leg.dep_time_) +
                      std::chrono::minutes{5}));
            }},
        j_leg.uses_);
  }

  cleanup_intermodal(itinerary);

  return itinerary;
}

}  // namespace motis
