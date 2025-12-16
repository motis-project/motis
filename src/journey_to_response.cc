#include "motis/journey_to_response.h"

#include <cmath>
#include <iostream>
#include <span>
#include <variant>

#include "utl/enumerate.h"
#include "utl/overloaded.h"
#include "utl/visit.h"

#include "geo/polyline_format.h"

#include "nigiri/common/split_duration.h"
#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/service_alert.h"
#include "nigiri/special_stations.h"
#include "nigiri/types.h"

#include "adr/typeahead.h"
#include "motis-api/motis-api.h"
#include "motis/constants.h"
#include "motis/flex/flex_output.h"
#include "motis/gbfs/gbfs_output.h"
#include "motis/gbfs/routing_data.h"
#include "motis/odm/odm.h"
#include "motis/osr/mode_to_profile.h"
#include "motis/osr/street_routing.h"
#include "motis/place.h"
#include "motis/polyline.h"
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
    case osr::search_profile::kCarDropOffWheelchair: [[fallthrough]];
    case osr::search_profile::kCarDropOff: return api::ModeEnum::CAR_DROPOFF;
    case osr::search_profile::kFoot: [[fallthrough]];
    case osr::search_profile::kWheelchair: return api::ModeEnum::WALK;
    case osr::search_profile::kCar: return api::ModeEnum::CAR;
    case osr::search_profile::kBikeElevationLow: [[fallthrough]];
    case osr::search_profile::kBikeElevationHigh: [[fallthrough]];
    case osr::search_profile::kBikeFast: [[fallthrough]];
    case osr::search_profile::kBike: return api::ModeEnum::BIKE;
    case osr::search_profile::kBikeSharing: [[fallthrough]];
    case osr::search_profile::kCarSharing: return api::ModeEnum::RENTAL;
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

std::optional<std::vector<api::Alert>> get_alerts(
    n::rt::frun const& fr,
    std::optional<std::pair<n::rt::run_stop, n::event_type>> const& s,
    bool const fuzzy_stop,
    std::optional<std::vector<std::string>> const& language) {
  if (fr.rtt_ == nullptr || !fr.is_scheduled()) {  // TODO added
    return std::nullopt;
  }

  auto const& tt = *fr.tt_;
  auto const* rtt = fr.rtt_;

  auto const to_time_range =
      [&](n::interval<n::unixtime_t> const x) -> api::TimeRange {
    return {x.from_, x.to_};
  };
  auto const to_cause = [](n::alert_cause const x) {
    return api::AlertCauseEnum{static_cast<int>(x)};
  };
  auto const to_effect = [](n::alert_effect const x) {
    return api::AlertEffectEnum{static_cast<int>(x)};
  };
  auto const convert_to_str = [](std::string_view s) {
    return std::optional{std::string{s}};
  };
  auto const to_alert = [&](n::alert_idx_t const x) -> api::Alert {
    auto const& a = rtt->alerts_;
    auto const get_translation =
        [&](auto const& translations) -> std::optional<std::string> {
      if (translations.empty()) {
        return std::nullopt;
      } else if (!language.has_value()) {
        return a.strings_.try_get(translations.front().text_)
            .and_then(convert_to_str);
      } else {
        for (auto const& req_lang : *language) {
          auto const it = utl::find_if(
              translations, [&](n::alert_translation const translation) {
                auto const translation_lang =
                    a.strings_.try_get(translation.language_);
                return translation_lang.has_value() &&
                       translation_lang->starts_with(req_lang);
              });
          if (it == end(translations)) {
            continue;
          }
          return a.strings_.try_get(it->text_).and_then(convert_to_str);
        }
        return a.strings_.try_get(translations.front().text_)
            .and_then(convert_to_str);
      }
    };
    return {
        .communicationPeriod_ =
            a.communication_period_[x].empty()
                ? std::nullopt
                : std::optional{utl::to_vec(a.communication_period_[x],
                                            to_time_range)},
        .impactPeriod_ = a.impact_period_[x].empty()
                             ? std::nullopt
                             : std::optional{utl::to_vec(a.impact_period_[x],
                                                         to_time_range)},
        .cause_ = to_cause(a.cause_[x]),
        .causeDetail_ = get_translation(a.cause_detail_[x]),
        .effect_ = to_effect(a.effect_[x]),
        .effectDetail_ = get_translation(a.effect_detail_[x]),
        .url_ = get_translation(a.url_[x]),
        .headerText_ = get_translation(a.header_text_[x]).value_or(""),
        .descriptionText_ =
            get_translation(a.description_text_[x]).value_or(""),
        .ttsHeaderText_ = get_translation(a.tts_header_text_[x]),
        .ttsDescriptionText_ = get_translation(a.tts_description_text_[x]),
        .imageUrl_ = a.image_[x].empty()
                         ? std::nullopt
                         : a.strings_.try_get(a.image_[x].front().url_)
                               .and_then(convert_to_str),
        .imageMediaType_ =
            a.image_[x].empty()
                ? std::nullopt
                : a.strings_.try_get(a.image_[x].front().media_type_)
                      .and_then(convert_to_str),
        .imageAlternativeText_ = get_translation(a.image_alternative_text_[x])};
  };

  auto const x =
      s.and_then([](std::pair<n::rt::run_stop, n::event_type> const& rs_ev) {
         auto const& [rs, ev_type] = rs_ev;
         return std::optional{rs.get_trip_idx(ev_type)};
       }).value_or(fr.trip_idx());
  auto const l =
      s.and_then([](std::pair<n::rt::run_stop, n::event_type> const& rs) {
         return std::optional{rs.first.get_location_idx()};
       }).value_or(n::location_idx_t::invalid());
  auto alerts = std::vector<api::Alert>{};
  for (auto const& t : tt.trip_ids_[x]) {
    auto const src = tt.trip_id_src_[t];
    for (auto const& a :
         rtt->alerts_.get_alerts(tt, src, x, fr.rt_, l, fuzzy_stop)) {
      alerts.emplace_back(to_alert(a));
    }
  }

  return alerts.empty() ? std::nullopt : std::optional{std::move(alerts)};
}

struct parent_name_hash {
  bool operator()(n::location_idx_t const l) const {
    return cista::hash(tt_->get_default_name(tt_->locations_.get_root_idx(l)));
  }
  n::timetable const* tt_{nullptr};
};

struct parent_name_eq {
  bool operator()(n::location_idx_t const a, n::location_idx_t const b) const {
    return tt_->get_default_name(tt_->locations_.get_root_idx(a)) ==
           tt_->get_default_name(tt_->locations_.get_root_idx(b));
  }
  n::timetable const* tt_{nullptr};
};

using unique_stop_map_t =
    hash_map<n::location_idx_t, bool, parent_name_hash, parent_name_eq>;

void get_is_unique_stop_name(n::rt::frun const& fr,
                             n::interval<n::stop_idx_t> const& stops,
                             unique_stop_map_t& is_unique) {
  is_unique.clear();
  for (auto const i : stops) {
    auto const [it, is_new] = is_unique.emplace(fr[i].get_location_idx(), true);
    if (!is_new) {
      it->second = false;
    }
  }
}

api::Itinerary journey_to_response(
    osr::ways const* w,
    osr::lookup const* l,
    osr::platforms const* pl,
    n::timetable const& tt,
    tag_lookup const& tags,
    flex::flex_areas const* fl,
    elevators const* e,
    n::rt_timetable const* rtt,
    platform_matches_t const* matches,
    osr::elevation_storage const* elevations,
    n::shapes_storage const* shapes,
    gbfs::gbfs_routing_data& gbfs_rd,
    adr_ext const* ae,
    tz_map_t const* tz_map,
    n::routing::journey const& j,
    place_t const& start,
    place_t const& dest,
    street_routing_cache_t& cache,
    osr::bitvec<osr::node_idx_t>* blocked_mem,
    bool const car_transfers,
    osr_parameters const& osr_params,
    api::PedestrianProfileEnum const pedestrian_profile,
    api::ElevationCostsEnum const elevation_costs,
    bool const join_interlined_legs,
    bool const detailed_transfers,
    bool const with_fares,
    bool const with_scheduled_skipped_stops,
    double const timetable_max_matching_distance,
    double const max_matching_distance,
    unsigned const api_version,
    bool const ignore_start_rental_return_constraints,
    bool const ignore_dest_rental_return_constraints,
    n::lang_t const& lang) {
  utl::verify(!j.legs_.empty(), "journey without legs");

  auto const fares =
      with_fares ? std::optional{n::get_fares(tt, rtt, j)} : std::nullopt;
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
    return {.riderCategoryName_ = std::string{tt.strings_.get(r.name_)},
            .isDefaultFareCategory_ = r.is_default_fare_category_,
            .eligibilityUrl_ = tt.strings_.try_get(r.eligibility_url_)
                                   .and_then([](std::string_view s) {
                                     return std::optional{std::string{s}};
                                   })};
  };
  auto const to_products =
      [&](n::fares const& f,
          n::fare_product_idx_t const x) -> std::vector<api::FareProduct> {
    if (x == n::fare_product_idx_t::invalid()) {
      return {};
    }
    return utl::to_vec(
        f.fare_products_[x],
        [&](n::fares::fare_product const& p) -> api::FareProduct {
          return {
              .name_ = std::string{tt.strings_.get(p.name_)},
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
        });
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
              [](n::routing::journey::leg const& leg) {
                return holds_alternative<n::routing::journey::run_enter_exit>(
                           leg.uses_) ||
                       odm::is_odm_leg(leg, kOdmTransportModeId) ||
                       odm::is_odm_leg(leg, kRideSharingTransportModeId);
              }) -
              1),
      .fareTransfers_ =
          fares.and_then([&](std::vector<n::fare_transfer> const& transfers) {
            return std::optional{utl::to_vec(
                transfers, [&](n::fare_transfer const& t) -> api::FareTransfer {
                  return {.rule_ = t.rule_.and_then([&](auto&& r) {
                            return std::optional{to_rule(r)};
                          }),
                          .transferProducts_ = t.rule_.and_then([&](auto&& r) {
                            return t.legs_.empty()
                                       ? std::nullopt
                                       : std::optional{to_products(
                                             tt.fares_[t.legs_.front().src_],
                                             r.fare_product_)};
                          }),
                          .effectiveFareLegProducts_ =
                              utl::to_vec(t.legs_, [&](auto&& l) {
                                return utl::to_vec(l.rule_, [&](auto&& r) {
                                  return to_products(tt.fares_[l.src_],
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

  auto const get_first_run_tz = [&]() -> std::optional<std::string> {
    if (j.legs_.size() < 2) {
      return std::nullopt;
    }
    auto const osm_tz = get_tz(tt, ae, tz_map, j.legs_[1].from_);
    if (osm_tz != nullptr) {
      return std::optional{osm_tz->name()};
    }
    return utl::visit(
        j.legs_[1].uses_, [&](n::routing::journey::run_enter_exit const& x) {
          return n::rt::frun{tt, rtt, x.r_}[0].get_tz_name(n::event_type::kDep);
        });
  };

  for (auto const [_, j_leg] : utl::enumerate(j.legs_)) {
    auto const pred =
        itinerary.legs_.empty() ? nullptr : &itinerary.legs_.back();
    auto const fallback_tz =
        pred == nullptr ? get_first_run_tz() : pred->to_.tz_;
    auto const from =
        pred == nullptr
            ? to_place(&tt, &tags, w, pl, matches, ae, tz_map, lang,
                       tt_location{j_leg.from_}, start, dest, "", fallback_tz)
            : pred->to_;
    auto const to =
        to_place(&tt, &tags, w, pl, matches, ae, tz_map, lang,
                 tt_location{j_leg.to_}, start, dest, "", fallback_tz);

    auto is_unique =
        unique_stop_map_t{0U, parent_name_hash{&tt}, parent_name_eq{&tt}};
    auto const to_place = [&](n::rt::run_stop const& s,
                              n::event_type const ev_type) {
      auto p = ::motis::to_place(&tt, &tags, w, pl, matches, ae, tz_map, lang,
                                 s, start, dest);
      p.alerts_ = get_alerts(*s.fr_, std::pair{s, ev_type}, false, lang);
      if (auto const it = is_unique.find(s.get_location_idx());
          it != end(is_unique) && !it->second) {
        p.name_ =
            tt.translate(lang, tt.locations_.names_[s.get_location_idx()]);
      }
      return p;
    };

    std::visit(
        utl::overloaded{
            [&](n::routing::journey::run_enter_exit const& t) {
              auto const fr = n::rt::frun{tt, rtt, t.r_};
              auto is_first_part = true;
              auto const write_run_leg = [&](auto,
                                             n::interval<n::stop_idx_t> const
                                                 subrange) {
                auto const common_stops = subrange.intersect(t.stop_range_);
                if (common_stops.size() <= 1) {
                  return;
                }

                get_is_unique_stop_name(fr, common_stops, is_unique);

                auto const enter_stop = fr[common_stops.from_];
                auto const exit_stop = fr[common_stops.to_ - 1U];
                auto const color =
                    enter_stop.get_route_color(n::event_type::kDep);
                auto const& agency =
                    enter_stop.get_provider(n::event_type::kDep);
                auto const fare_indices = get_fare_indices(fares, j_leg);

                auto const src = [&]() {
                  if (!fr.is_scheduled()) {
                    return n::source_idx_t::invalid();
                  }
                  auto const trip =
                      enter_stop.get_trip_idx(n::event_type::kDep);
                  auto const id_idx = tt.trip_ids_[trip].front();
                  return tt.trip_id_src_[id_idx];
                }();
                auto const [service_day, _] =
                    enter_stop.get_trip_start(n::event_type::kDep);

                auto& leg = itinerary.legs_.emplace_back(api::Leg{
                    .mode_ = to_mode(enter_stop.get_clasz(n::event_type::kDep),
                                     api_version),
                    .from_ = to_place(enter_stop, n::event_type::kDep),
                    .to_ = to_place(exit_stop, n::event_type::kArr),
                    .duration_ =
                        std::chrono::duration_cast<std::chrono::seconds>(
                            exit_stop.time(n::event_type::kArr) -
                            enter_stop.time(n::event_type::kDep))
                            .count(),
                    .startTime_ = enter_stop.time(n::event_type::kDep),
                    .endTime_ = exit_stop.time(n::event_type::kArr),
                    .scheduledStartTime_ =
                        enter_stop.scheduled_time(n::event_type::kDep),
                    .scheduledEndTime_ =
                        exit_stop.scheduled_time(n::event_type::kArr),
                    .realTime_ = fr.is_rt(),
                    .scheduled_ = fr.is_scheduled(),
                    .interlineWithPreviousLeg_ = !is_first_part,
                    .headsign_ = std::string{enter_stop.direction(
                        lang, n::event_type::kDep)},
                    .tripTo_ =
                        [&]() {
                          auto const last = enter_stop.get_last_trip_stop(
                              n::event_type::kDep);
                          auto p = to_place(last, n::event_type::kArr);
                          p.arrival_ = last.time(n::event_type::kArr);
                          p.scheduledArrival_ =
                              last.scheduled_time(n::event_type::kArr);
                          return p;
                        }(),
                    .routeId_ = std::string{enter_stop.get_route_id(
                        n::event_type::kDep)},
                    .directionId_ =
                        enter_stop.get_direction_id(n::event_type::kDep) == 0
                            ? "0"
                            : "1",
                    .routeColor_ = to_str(color.color_),
                    .routeTextColor_ = to_str(color.text_color_),
                    .routeType_ = enter_stop.route_type(n::event_type::kDep)
                                      .and_then([](n::route_type_t const x) {
                                        return std::optional{to_idx(x)};
                                      }),
                    .agencyName_ =
                        std::string{tt.translate(lang, agency.name_)},
                    .agencyUrl_ = std::string{tt.translate(lang, agency.url_)},
                    .agencyId_ =
                        std::string{
                            tt.strings_.try_get(agency.id_).value_or("?")},
                    .tripId_ = tags.id(tt, enter_stop, n::event_type::kDep),
                    .routeShortName_ = {std::string{
                        api_version > 3 ? enter_stop.route_short_name(
                                              n::event_type::kDep, lang)
                                        : enter_stop.display_name(
                                              n::event_type::kDep, lang)}},
                    .routeLongName_ = {std::string{
                        enter_stop.route_long_name(n::event_type::kDep, lang)}},
                    .tripShortName_ = {std::string{
                        enter_stop.trip_short_name(n::event_type::kDep, lang)}},
                    .displayName_ = {std::string{
                        enter_stop.display_name(n::event_type::kDep, lang)}},
                    .cancelled_ = fr.is_cancelled(),
                    .source_ = fmt::to_string(fr.dbg()),
                    .fareTransferIndex_ = fare_indices.and_then([](auto&& x) {
                      return std::optional{x.transfer_idx_};
                    }),
                    .effectiveFareLegIndex_ =
                        fare_indices.and_then([](auto&& x) {
                          return std::optional{x.effective_fare_leg_idx_};
                        }),
                    .alerts_ = get_alerts(fr, std::nullopt, false, lang),
                    .loopedCalendarSince_ =
                        (fr.is_scheduled() &&
                         src != n::source_idx_t::invalid() &&
                         tt.src_end_date_[src] < service_day)
                            ? std::optional{tt.src_end_date_[src]}
                            : std::nullopt,
                });

                auto const attributes =
                    tt.attribute_combinations_[enter_stop
                                                   .get_attribute_combination(
                                                       n::event_type::kDep)];
                if (!leg.alerts_ && !attributes.empty()) {
                  leg.alerts_ = std::vector<api::Alert>{};
                }
                for (auto const& a : attributes) {
                  leg.alerts_->push_back(api::Alert{
                      .code_ = std::string{tt.attributes_[a].code_.view()},
                      .headerText_ = std::string{
                          tt.translate(lang, tt.attributes_[a].text_)}});
                }

                leg.from_.vertexType_ = api::VertexTypeEnum::TRANSIT;
                leg.from_.departure_ = leg.startTime_;
                leg.from_.scheduledDeparture_ = leg.scheduledStartTime_;
                leg.to_.vertexType_ = api::VertexTypeEnum::TRANSIT;
                leg.to_.arrival_ = leg.endTime_;
                leg.to_.scheduledArrival_ = leg.scheduledEndTime_;
                auto polyline = geo::polyline{};
                fr.for_each_shape_point(shapes, common_stops,
                                        [&](geo::latlng const& pos) {
                                          polyline.emplace_back(pos);
                                        });
                leg.legGeometry_ = api_version == 1 ? to_polyline<7>(polyline)
                                                    : to_polyline<6>(polyline);

                auto const first =
                    static_cast<n::stop_idx_t>(common_stops.from_ + 1U);
                auto const last =
                    static_cast<n::stop_idx_t>(common_stops.to_ - 1U);
                leg.intermediateStops_ = std::vector<api::Place>{};
                for (auto i = first; i < last; ++i) {
                  auto const stop = fr[i];
                  if (!with_scheduled_skipped_stops &&
                      !stop.get_scheduled_stop().in_allowed() &&
                      !stop.get_scheduled_stop().out_allowed() &&
                      !stop.in_allowed() && !stop.out_allowed()) {
                    continue;
                  }
                  auto& p = leg.intermediateStops_->emplace_back(
                      to_place(stop, n::event_type::kDep));
                  p.departure_ = stop.time(n::event_type::kDep);
                  p.scheduledDeparture_ =
                      stop.scheduled_time(n::event_type::kDep);
                  p.arrival_ = stop.time(n::event_type::kArr);
                  p.scheduledArrival_ =
                      stop.scheduled_time(n::event_type::kArr);
                }
                is_first_part = false;
              };

              if (join_interlined_legs) {
                write_run_leg(n::trip_idx_t{}, t.stop_range_);
              } else {
                fr.for_each_trip(write_run_leg);
              }
            },
            [&](n::footpath) {
              append(w && l && detailed_transfers
                         ? street_routing(
                               *w, *l, e, elevations, lang, from, to,
                               default_output{
                                   *w, car_transfers
                                           ? osr::search_profile::kCar
                                           : to_profile(api::ModeEnum::WALK,
                                                        pedestrian_profile,
                                                        elevation_costs)},
                               j_leg.dep_time_, j_leg.arr_time_,
                               car_transfers ? 250.0
                                             : timetable_max_matching_distance,
                               osr_params, cache, *blocked_mem, api_version,
                               std::chrono::duration_cast<std::chrono::seconds>(
                                   j_leg.arr_time_ - j_leg.dep_time_) +
                                   std::chrono::minutes{10})
                         : dummy_itinerary(from, to, api::ModeEnum::WALK,
                                           j_leg.dep_time_, j_leg.arr_time_));
            },
            [&](n::routing::offset const x) {
              auto out = std::unique_ptr<output>{};
              if (flex::mode_id::is_flex(x.transport_mode_id_)) {
                out = std::make_unique<flex::flex_output>(
                    *w, *l, pl, matches, ae, tz_map, tags, tt, *fl,
                    flex::mode_id{x.transport_mode_id_});
              } else if (x.transport_mode_id_ >= kGbfsTransportModeIdOffset) {
                auto const is_pre_transit = pred == nullptr;
                out = std::make_unique<gbfs::gbfs_output>(
                    *w, gbfs_rd, gbfs_rd.get_products_ref(x.transport_mode_id_),
                    is_pre_transit ? ignore_start_rental_return_constraints
                                   : ignore_dest_rental_return_constraints);
              } else {
                out =
                    std::make_unique<default_output>(*w, x.transport_mode_id_);
              }

              append(street_routing(
                  *w, *l, e, elevations, lang, from, to, *out, j_leg.dep_time_,
                  j_leg.arr_time_, max_matching_distance, osr_params, cache,
                  *blocked_mem, api_version,
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
