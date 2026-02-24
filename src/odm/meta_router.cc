#if defined(_MSC_VER)
// needs to be the first to include WinSock.h
#include "boost/asio.hpp"
#endif

#include "motis/odm/meta_router.h"

#include <vector>

#include "boost/asio/io_context.hpp"
#include "boost/thread/tss.hpp"

#include "prometheus/histogram.h"

#include "utl/erase_duplicates.h"

#include "ctx/ctx.h"

#include "nigiri/logging.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/types.h"

#include "osr/routing/route.h"

#include "motis-api/motis-api.h"
#include "motis/constants.h"
#include "motis/ctx_data.h"
#include "motis/elevators/elevators.h"
#include "motis/endpoints/routing.h"
#include "motis/gbfs/routing_data.h"
#include "motis/journey_to_response.h"
#include "motis/metrics_registry.h"
#include "motis/odm/bounds.h"
#include "motis/odm/journeys.h"
#include "motis/odm/mixer.h"
#include "motis/odm/odm.h"
#include "motis/odm/prima.h"
#include "motis/odm/shorten.h"
#include "motis/odm/td_offsets.h"
#include "motis/osr/parameters.h"
#include "motis/osr/street_routing.h"
#include "motis/place.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/modes_to_clasz_mask.h"
#include "motis/timetable/time_conv.h"
#include "motis/transport_mode_ids.h"

namespace n = nigiri;
using namespace std::chrono_literals;

using td_offsets_t =
    n::hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>;

namespace motis::odm {

constexpr auto kODMLookAhead = nigiri::duration_t{24h};
constexpr auto kSearchIntervalSize = nigiri::duration_t{10h};
constexpr auto kContextPadding = nigiri::duration_t{2h};
static auto const kMixer = get_default_mixer();

void print_time(auto const& start,
                std::string_view name,
                prometheus::Histogram& metric) {
  auto const millis = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
  n::log(n::log_lvl::debug, "motis.prima", "{} {}", name, millis);
  metric.Observe(static_cast<double>(millis.count()) / 1000.0);
}

meta_router::meta_router(ep::routing const& r,
                         api::plan_params const& query,
                         std::vector<api::ModeEnum> const& pre_transit_modes,
                         std::vector<api::ModeEnum> const& post_transit_modes,
                         std::vector<api::ModeEnum> const& direct_modes,
                         std::variant<osr::location, tt_location> const& from,
                         std::variant<osr::location, tt_location> const& to,
                         api::Place const& from_p,
                         api::Place const& to_p,
                         nigiri::routing::query const& start_time,
                         std::vector<api::Itinerary>& direct,
                         nigiri::duration_t const fastest_direct,
                         bool const odm_pre_transit,
                         bool const odm_post_transit,
                         bool const odm_direct,
                         bool const ride_sharing_pre_transit,
                         bool const ride_sharing_post_transit,
                         bool const ride_sharing_direct,
                         unsigned const api_version)
    : r_{r},
      query_{query},
      pre_transit_modes_{pre_transit_modes},
      post_transit_modes_{post_transit_modes},
      direct_modes_{direct_modes},
      from_{from},
      to_{to},
      from_place_{from_p},
      to_place_{to_p},
      start_time_{start_time},
      direct_{direct},
      fastest_direct_{fastest_direct},
      odm_pre_transit_{odm_pre_transit},
      odm_post_transit_{odm_post_transit},
      odm_direct_{odm_direct},
      ride_sharing_pre_transit_{ride_sharing_pre_transit},
      ride_sharing_post_transit_{ride_sharing_post_transit},
      ride_sharing_direct_{ride_sharing_direct},
      api_version_{api_version},
      tt_{r_.tt_},
      rt_{r.rt_},
      rtt_{rt_->rtt_.get()},
      e_{rt_->e_.get()},
      gbfs_rd_{r.w_, r.l_, r.gbfs_},
      start_{query_.arriveBy_ ? to_ : from_},
      dest_{query_.arriveBy_ ? from_ : to_},
      start_modes_{query_.arriveBy_ ? post_transit_modes_ : pre_transit_modes_},
      dest_modes_{query_.arriveBy_ ? pre_transit_modes_ : post_transit_modes_},
      start_form_factors_{query_.arriveBy_
                              ? query_.postTransitRentalFormFactors_
                              : query_.preTransitRentalFormFactors_},
      dest_form_factors_{query_.arriveBy_
                             ? query_.preTransitRentalFormFactors_
                             : query_.postTransitRentalFormFactors_},
      start_propulsion_types_{query_.arriveBy_
                                  ? query_.postTransitRentalPropulsionTypes_
                                  : query_.preTransitRentalPropulsionTypes_},
      dest_propulsion_types_{query_.arriveBy_
                                 ? query_.preTransitRentalPropulsionTypes_
                                 : query_.postTransitRentalPropulsionTypes_},
      start_rental_providers_{query_.arriveBy_
                                  ? query_.postTransitRentalProviders_
                                  : query_.preTransitRentalProviders_},
      dest_rental_providers_{query_.arriveBy_
                                 ? query_.preTransitRentalProviders_
                                 : query_.postTransitRentalProviders_},
      start_rental_provider_groups_{
          query_.arriveBy_ ? query_.postTransitRentalProviderGroups_
                           : query_.preTransitRentalProviderGroups_},
      dest_rental_provider_groups_{
          query_.arriveBy_ ? query_.preTransitRentalProviderGroups_
                           : query_.postTransitRentalProviderGroups_},
      start_ignore_rental_return_constraints_{
          query.arriveBy_ ? query_.ignorePreTransitRentalReturnConstraints_
                          : query_.ignorePostTransitRentalReturnConstraints_},
      dest_ignore_rental_return_constraints_{
          query.arriveBy_ ? query_.ignorePostTransitRentalReturnConstraints_
                          : query_.ignorePreTransitRentalReturnConstraints_} {}

meta_router::~meta_router() = default;

n::routing::query meta_router::get_base_query(
    n::interval<n::unixtime_t> const& intvl) const {
  return {
      .start_time_ = intvl,
      .start_match_mode_ = motis::ep::get_match_mode(r_, start_),
      .dest_match_mode_ = motis::ep::get_match_mode(r_, dest_),
      .use_start_footpaths_ = !motis::ep::is_intermodal(r_, start_),
      .max_transfers_ = static_cast<std::uint8_t>(
          query_.maxTransfers_.has_value() ? *query_.maxTransfers_
                                           : n::routing::kMaxTransfers),
      .max_travel_time_ = query_.maxTravelTime_
                              .and_then([](std::int64_t const dur) {
                                return std::optional{n::duration_t{dur}};
                              })
                              .value_or(ep::kInfinityDuration),
      .min_connection_count_ = 0U,
      .extend_interval_earlier_ = false,
      .extend_interval_later_ = false,
      .max_interval_ = std::nullopt,
      .prf_idx_ = static_cast<n::profile_idx_t>(
          query_.useRoutedTransfers_
              ? (query_.pedestrianProfile_ ==
                         api::PedestrianProfileEnum::WHEELCHAIR
                     ? 2U
                     : 1U)
              : 0U),
      .allowed_claszes_ = to_clasz_mask(query_.transitModes_),
      .require_bike_transport_ = query_.requireBikeTransport_,
      .require_car_transport_ = query_.requireCarTransport_,
      .transfer_time_settings_ =
          n::routing::transfer_time_settings{
              .default_ = (query_.minTransferTime_ == 0 &&
                           query_.additionalTransferTime_ == 0 &&
                           query_.transferTimeFactor_ == 1.0),
              .min_transfer_time_ = n::duration_t{query_.minTransferTime_},
              .additional_time_ = n::duration_t{query_.additionalTransferTime_},
              .factor_ = static_cast<float>(query_.transferTimeFactor_)},
      .via_stops_ =
          motis::ep::get_via_stops(*tt_, *r_.tags_, query_.via_,
                                   query_.viaMinimumStay_, query_.arriveBy_),
      .fastest_direct_ = fastest_direct_ == ep::kInfinityDuration
                             ? std::nullopt
                             : std::optional{fastest_direct_}};
}

std::vector<meta_router::routing_result> meta_router::search_interval(
    std::vector<n::routing::query> const& sub_queries) const {
  auto const tasks = utl::to_vec(sub_queries, [&](n::routing::query const& q) {
    auto fn = [&, q = std::move(q)]() mutable {
      auto const timeout = std::chrono::seconds{query_.timeout_.value_or(
          r_.config_.get_limits().routing_max_timeout_seconds_)};
      auto search_state = n::routing::search_state{};
      auto raptor_state = n::routing::raptor_state{};
      return routing_result{raptor_search(
          *tt_, rtt_, search_state, raptor_state, std::move(q),
          query_.arriveBy_ ? n::direction::kBackward : n::direction::kForward,
          timeout)};
    };
    return ctx_call(ctx_data{}, std::move(fn));
  });
  return utl::to_vec(
      tasks,
      [](ctx::future_ptr<ctx_data, meta_router::routing_result> const& t) {
        return t->val();
      });
}

std::vector<n::routing::journey> collect_odm_journeys(
    std::vector<meta_router::routing_result> const& results,
    nigiri::transport_mode_id_t const mode) {
  auto taxi_journeys = std::vector<n::routing::journey>{};
  for (auto const& r : results | std::views::drop(1)) {
    for (auto const& j : r.journeys_) {
      if (uses_odm(j, mode)) {
        taxi_journeys.push_back(j);
        taxi_journeys.back().transfers_ +=
            (j.legs_.empty() || !is_odm_leg(j.legs_.front(), mode) ? 0U : 1U) +
            (j.legs_.size() < 2U || !is_odm_leg(j.legs_.back(), mode) ? 0U
                                                                      : 1U);
      }
    }
  }
  n::log(n::log_lvl::debug, "motis.prima",
         "[routing] collected {} mixed ODM-PT journeys for mode {}",
         taxi_journeys.size(), mode);
  return taxi_journeys;
}

api::plan_response meta_router::run() {
  auto const init_start = std::chrono::steady_clock::now();
  utl::verify(r_.tt_ != nullptr && r_.tags_ != nullptr,
              "mode=TRANSIT requires timetable to be loaded");
  auto prepare_stats = motis::ep::stats_map_t{};
  auto const start_intvl = std::visit(
      utl::overloaded{[](n::interval<n::unixtime_t> const i) { return i; },
                      [](n::unixtime_t const t) {
                        return n::interval<n::unixtime_t>{t, t};
                      }},
      start_time_.start_time_);
  auto search_intvl =
      n::interval<n::unixtime_t>{start_time_.extend_interval_earlier_
                                     ? start_intvl.to_ - kSearchIntervalSize
                                     : start_intvl.from_,
                                 start_time_.extend_interval_later_
                                     ? start_intvl.from_ + kSearchIntervalSize
                                     : start_intvl.to_};
  search_intvl.from_ = r_.tt_->external_interval().clamp(search_intvl.from_);
  search_intvl.to_ = r_.tt_->external_interval().clamp(search_intvl.to_);
  auto const context_intvl = n::interval<n::unixtime_t>{
      search_intvl.from_ - kContextPadding, search_intvl.to_ + kContextPadding};
  auto const taxi_intvl =
      query_.arriveBy_
          ? n::interval<n::unixtime_t>{context_intvl.from_ - kODMLookAhead,
                                       context_intvl.to_}
          : n::interval<n::unixtime_t>{context_intvl.from_,
                                       context_intvl.to_ + kODMLookAhead};
  auto const to_osr_loc = [&](auto const& place) {
    return std::visit(
        utl::overloaded{
            [](osr::location const& l) { return l; },
            [&](tt_location const& l) {
              return osr::location{
                  .pos_ = tt_->locations_.coordinates_[l.l_],
                  .lvl_ = osr::level_t{std::uint8_t{osr::level_t::kNoLevel}}};
            }},
        place);
  };
  auto p = prima{r_.config_.prima_->url_, to_osr_loc(from_), to_osr_loc(to_),
                 query_};
  p.init(search_intvl, taxi_intvl, odm_pre_transit_, odm_post_transit_,
         odm_direct_, ride_sharing_pre_transit_, ride_sharing_post_transit_,
         ride_sharing_direct_, *tt_, rtt_, r_, e_, gbfs_rd_, from_place_,
         to_place_, query_, start_time_, api_version_);

  std::erase(start_modes_, api::ModeEnum::ODM);
  std::erase(start_modes_, api::ModeEnum::RIDE_SHARING);
  std::erase(dest_modes_, api::ModeEnum::ODM);
  std::erase(dest_modes_, api::ModeEnum::RIDE_SHARING);

  print_time(
      init_start,
      fmt::format("[init] (#first_mile_offsets: {}, #last_mile_offsets: {}, "
                  "#direct_rides: {})",
                  p.first_mile_taxi_.size(), p.last_mile_taxi_.size(),
                  p.direct_taxi_.size()),
      r_.metrics_->routing_execution_duration_seconds_init_);

  auto const blacklist_start = std::chrono::steady_clock::now();
  auto const blacklisted_taxis = p.blacklist_taxi(*tt_, taxi_intvl);
  print_time(blacklist_start,
             fmt::format("[blacklist taxi] (#first_mile_offsets: {}, "
                         "#last_mile_offsets: {}, #direct_rides: {})",
                         p.first_mile_taxi_.size(), p.last_mile_taxi_.size(),
                         p.direct_taxi_.size()),
             r_.metrics_->routing_execution_duration_seconds_blacklisting_);

  auto const whitelist_ride_sharing_start = std::chrono::steady_clock::now();
  auto const whitelisted_ride_sharing = p.whitelist_ride_sharing(*tt_);
  n::log(n::log_lvl::debug, "motis.prima",
         "[whitelist ride-sharing] ride-sharing events after whitelisting: {}",
         p.n_ride_sharing_events());
  print_time(
      whitelist_ride_sharing_start,
      fmt::format("[whitelist ride-sharing] (#first_mile_ride_sharing: {}, "
                  "#last_mile_ride_sharing: {}, #direct_ride_sharing: {})",
                  p.first_mile_ride_sharing_.size(),
                  p.last_mile_ride_sharing_.size(),
                  p.direct_ride_sharing_.size()),
      r_.metrics_->routing_execution_duration_seconds_blacklisting_);

  auto const prep_queries_start = std::chrono::steady_clock::now();
  auto const [first_mile_taxi_short, first_mile_taxi_long] =
      get_td_offsets_split(p.first_mile_taxi_, p.first_mile_taxi_times_,
                           kOdmTransportModeId);
  auto const [last_mile_taxi_short, last_mile_taxi_long] = get_td_offsets_split(
      p.last_mile_taxi_, p.last_mile_taxi_times_, kOdmTransportModeId);
  auto const params = get_osr_parameters(query_);
  auto const pre_transit_time = std::min(
      std::chrono::seconds{query_.maxPreTransitTime_},
      std::chrono::seconds{
          r_.config_.get_limits().street_routing_max_prepost_transit_seconds_});
  auto const post_transit_time = std::min(
      std::chrono::seconds{query_.maxPostTransitTime_},
      std::chrono::seconds{
          r_.config_.get_limits().street_routing_max_prepost_transit_seconds_});
  auto const qf = query_factory{
      .base_query_ = get_base_query(context_intvl),
      .start_walk_ = r_.get_offsets(
          rtt_, start_,
          query_.arriveBy_ ? osr::direction::kBackward
                           : osr::direction::kForward,
          start_modes_, start_form_factors_, start_propulsion_types_,
          start_rental_providers_, start_rental_provider_groups_,
          start_ignore_rental_return_constraints_, params,
          query_.pedestrianProfile_, query_.elevationCosts_,
          query_.arriveBy_ ? post_transit_time : pre_transit_time,
          query_.maxMatchingDistance_, gbfs_rd_, prepare_stats),
      .dest_walk_ = r_.get_offsets(
          rtt_, dest_,
          query_.arriveBy_ ? osr::direction::kForward
                           : osr::direction::kBackward,
          dest_modes_, dest_form_factors_, dest_propulsion_types_,
          dest_rental_providers_, dest_rental_provider_groups_,
          dest_ignore_rental_return_constraints_, params,
          query_.pedestrianProfile_, query_.elevationCosts_,
          query_.arriveBy_ ? pre_transit_time : post_transit_time,
          query_.maxMatchingDistance_, gbfs_rd_, prepare_stats),
      .td_start_walk_ = r_.get_td_offsets(
          rtt_, e_, start_,
          query_.arriveBy_ ? osr::direction::kBackward
                           : osr::direction::kForward,
          start_modes_, params, query_.pedestrianProfile_,
          query_.elevationCosts_, query_.maxMatchingDistance_,
          query_.arriveBy_ ? post_transit_time : pre_transit_time,
          context_intvl, prepare_stats),
      .td_dest_walk_ = r_.get_td_offsets(
          rtt_, e_, dest_,
          query_.arriveBy_ ? osr::direction::kForward
                           : osr::direction::kBackward,
          dest_modes_, params, query_.pedestrianProfile_,
          query_.elevationCosts_, query_.maxMatchingDistance_,
          query_.arriveBy_ ? pre_transit_time : post_transit_time,
          context_intvl, prepare_stats),
      .start_taxi_short_ =
          query_.arriveBy_ ? last_mile_taxi_short : first_mile_taxi_short,
      .start_taxi_long_ =
          query_.arriveBy_ ? last_mile_taxi_long : first_mile_taxi_long,
      .dest_taxi_short_ =
          query_.arriveBy_ ? first_mile_taxi_short : last_mile_taxi_short,
      .dest_taxi_long_ =
          query_.arriveBy_ ? first_mile_taxi_long : last_mile_taxi_long,
      .start_ride_sharing_ = query_.arriveBy_
                                 ? get_td_offsets(p.last_mile_ride_sharing_,
                                                  kRideSharingTransportModeId)
                                 : get_td_offsets(p.first_mile_ride_sharing_,
                                                  kRideSharingTransportModeId),
      .dest_ride_sharing_ = query_.arriveBy_
                                ? get_td_offsets(p.first_mile_ride_sharing_,
                                                 kRideSharingTransportModeId)
                                : get_td_offsets(p.last_mile_ride_sharing_,
                                                 kRideSharingTransportModeId)};
  print_time(prep_queries_start, "[prepare queries]",
             r_.metrics_->routing_execution_duration_seconds_preparing_);

  auto const routing_start = std::chrono::steady_clock::now();
  auto sub_queries =
      qf.make_queries(blacklisted_taxis, whitelisted_ride_sharing);
  n::log(n::log_lvl::debug, "motis.prima",
         "[prepare queries] {} queries prepared", sub_queries.size());
  auto const results = search_interval(sub_queries);
  utl::verify(!results.empty(), "prima: public transport result expected");
  auto const& pt_result = results.front();
  auto taxi_journeys = collect_odm_journeys(results, kOdmTransportModeId);
  shorten(taxi_journeys, p.first_mile_taxi_, p.first_mile_taxi_times_,
          p.last_mile_taxi_, p.last_mile_taxi_times_, *tt_, rtt_, query_);
  auto ride_share_journeys =
      collect_odm_journeys(results, kRideSharingTransportModeId);
  fix_first_mile_duration(ride_share_journeys, p.first_mile_ride_sharing_,
                          p.first_mile_ride_sharing_,
                          kRideSharingTransportModeId);
  fix_last_mile_duration(ride_share_journeys, p.last_mile_ride_sharing_,
                         p.last_mile_ride_sharing_,
                         kRideSharingTransportModeId);
  utl::erase_duplicates(
      taxi_journeys, std::less<n::routing::journey>{},
      [](auto const& a, auto const& b) {
        return a == b &&
               odm_time(a.legs_.front()) == odm_time(b.legs_.front()) &&
               odm_time(a.legs_.back()) == odm_time(b.legs_.back());
      });
  n::log(n::log_lvl::debug, "motis.prima", "[routing] interval searched: {}",
         pt_result.interval_);
  print_time(routing_start, "[routing]",
             r_.metrics_->routing_execution_duration_seconds_routing_);

  auto const whitelist_start = std::chrono::steady_clock::now();
  auto const was_whitelist_response_valid =
      p.whitelist_taxi(taxi_journeys, *tt_);
  if (was_whitelist_response_valid) {
    add_direct_odm(p.direct_taxi_, taxi_journeys, from_, to_, query_.arriveBy_,
                   kOdmTransportModeId);
  }
  if (whitelisted_ride_sharing) {
    add_direct_odm(p.direct_ride_sharing_, ride_share_journeys, from_, to_,
                   query_.arriveBy_, kRideSharingTransportModeId);
  }
  print_time(whitelist_start,
             fmt::format("[whitelisting] (#first_mile_taxi: {}, "
                         "#last_mile_taxi: {}, #direct_taxi: {})",
                         p.first_mile_taxi_.size(), p.last_mile_taxi_.size(),
                         p.direct_taxi_.size()),
             r_.metrics_->routing_execution_duration_seconds_whitelisting_);
  r_.metrics_->routing_odm_journeys_found_whitelist_.Observe(
      static_cast<double>(taxi_journeys.size()));

  auto const mixing_start = std::chrono::steady_clock::now();
  n::log(n::log_lvl::debug, "motis.prima",
         "[mixing] {} PT journeys and {} ODM journeys",
         pt_result.journeys_.size(), taxi_journeys.size());
  kMixer.mix(pt_result.journeys_, taxi_journeys, ride_share_journeys,
             r_.metrics_, std::nullopt);
  r_.metrics_->routing_odm_journeys_found_non_dominated_.Observe(
      static_cast<double>(taxi_journeys.size() - pt_result.journeys_.size()));
  print_time(mixing_start, "[mixing]",
             r_.metrics_->routing_execution_duration_seconds_mixing_);
  // remove journeys added for mixing context
  std::erase_if(taxi_journeys, [&](auto const& j) {
    return query_.arriveBy_ ? !search_intvl.contains(j.arrival_time())
                            : !search_intvl.contains(j.departure_time());
  });

  r_.metrics_->routing_journeys_found_.Increment(
      static_cast<double>(taxi_journeys.size()));
  r_.metrics_->routing_execution_duration_seconds_total_.Observe(
      static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - init_start)
                              .count()) /
      1000.0);

  if (!taxi_journeys.empty()) {
    r_.metrics_->routing_journey_duration_seconds_.Observe(static_cast<double>(
        to_seconds(taxi_journeys.begin()->arrival_time() -
                   taxi_journeys.begin()->departure_time())));
  }
  return {
      .from_ = from_place_,
      .to_ = to_place_,
      .direct_ = std::move(direct_),
      .itineraries_ = utl::to_vec(
          taxi_journeys,
          [&, cache = street_routing_cache_t{}](auto&& j) mutable {
            if (ep::blocked.get() == nullptr && r_.w_ != nullptr) {
              ep::blocked.reset(
                  new osr::bitvec<osr::node_idx_t>{r_.w_->n_nodes()});
            }
            auto response = journey_to_response(
                r_.w_, r_.l_, r_.pl_, *tt_, *r_.tags_, r_.fa_, e_, rtt_,
                r_.matches_, r_.elevations_, r_.shapes_, gbfs_rd_, r_.ae_,
                r_.tz_, j, start_, dest_, cache, ep::blocked.get(),
                query_.requireCarTransport_ && query_.useRoutedTransfers_,
                params, query_.pedestrianProfile_, query_.elevationCosts_,
                query_.joinInterlinedLegs_, query_.detailedTransfers_,
                query_.withFares_, query_.withScheduledSkippedStops_,
                r_.config_.timetable_.value().max_matching_distance_,
                query_.maxMatchingDistance_, api_version_,
                query_.ignorePreTransitRentalReturnConstraints_,
                query_.ignorePostTransitRentalReturnConstraints_,
                query_.language_);

            if (response.legs_.front().mode_ == api::ModeEnum::RIDE_SHARING &&
                response.legs_.size() == 1) {
              for (auto const [i, a] : utl::enumerate(p.direct_ride_sharing_)) {
                if (a.dep_ == response.legs_.front().startTime_ &&
                    a.arr_ == response.legs_.front().endTime_) {
                  response.legs_.front().tripId_ = std::optional{
                      p.direct_ride_sharing_tour_ids_.at(i).view()};
                  break;
                }
              }
              return response;
            }
            if (response.legs_.front().mode_ == api::ModeEnum::RIDE_SHARING) {
              for (auto const [i, a] :
                   utl::enumerate(p.first_mile_ride_sharing_)) {
                if (a.time_at_start_ ==
                        response.legs_.front()
                            .startTime_ &&  // not looking at time_at_stop_
                                            // because we would again need to
                                            // take into account the 5 min
                                            // shift...
                    r_.tags_->id(*tt_, a.stop_) ==
                        response.legs_.front().to_.stopId_) {
                  response.legs_.front().tripId_ = std::optional{
                      p.first_mile_ride_sharing_tour_ids_.at(i).view()};
                  break;
                }
              }
            }
            if (response.legs_.back().mode_ == api::ModeEnum::RIDE_SHARING) {
              for (auto const [i, a] :
                   utl::enumerate(p.last_mile_ride_sharing_)) {
                if (a.time_at_start_ ==
                        response.legs_.back()
                            .endTime_ &&  // not looking at time_at_stop_
                                          // because we would again need to take
                                          // into account the 5 min shift...
                    r_.tags_->id(*tt_, a.stop_) ==
                        response.legs_.back().from_.stopId_) {
                  response.legs_.back().tripId_ = std::optional{
                      p.last_mile_ride_sharing_tour_ids_.at(i).view()};
                  break;
                }
              }
            }

            auto const match_times = [&](motis::api::Leg const& leg,
                                         boost::json::array const& entries)
                -> std::optional<std::string> {
              auto const it = std::find_if(
                  std::begin(entries), std::end(entries),
                  [&](boost::json::value const& json_entry) {
                    if (json_entry.is_null()) {
                      return false;
                    }
                    auto const& object_entry = json_entry.as_object();
                    return to_unix(object_entry.at("pickupTime").as_int64()) ==
                               leg.startTime_ &&
                           to_unix(object_entry.at("dropoffTime").as_int64()) ==
                               leg.endTime_;
                  });

              if (it != std::end(entries)) {
                return boost::json::serialize(it->as_object());
              }

              return std::nullopt;
            };

            auto const match_location =
                [&](motis::api::Leg const& leg, boost::json::array const& outer,
                    std::vector<nigiri::location_idx_t> const& locations,
                    bool const check_to) -> std::optional<std::string> {
              auto const& stop_id =
                  check_to ? leg.to_.stopId_ : leg.from_.stopId_;
              for (auto const [loc, outer_value] : utl::zip(locations, outer)) {
                if (stop_id != r_.tags_->id(*tt_, loc)) {
                  continue;
                }
                auto const& inner = outer_value.as_array();
                if (auto result = match_times(leg, inner)) {
                  return result;
                }
              }
              return std::nullopt;
            };

            if (!was_whitelist_response_valid) {
              return response;
            }
            if (response.legs_.size() == 1 &&
                response.legs_.front().mode_ == api::ModeEnum::ODM) {
              if (auto const id = match_times(
                      response.legs_.front(),
                      p.whitelist_response_.at("direct").as_array());
                  id.has_value()) {
                response.legs_.front().tripId_ = std::optional{*id};
              }
              return response;
            }
            if (!response.legs_.empty() &&
                response.legs_.front().mode_ == api::ModeEnum::ODM) {
              if (auto const id = match_location(
                      response.legs_.front(),
                      p.whitelist_response_.at("start").as_array(),
                      p.whitelist_first_mile_locations_, true);
                  id.has_value()) {
                response.legs_.front().tripId_ = std::optional{*id};
              }
            }
            if (!response.legs_.empty() &&
                response.legs_.back().mode_ == api::ModeEnum::ODM) {
              if (auto const id = match_location(
                      response.legs_.back(),
                      p.whitelist_response_.at("target").as_array(),
                      p.whitelist_last_mile_locations_, false);
                  id.has_value()) {
                response.legs_.back().tripId_ = std::optional{*id};
              }
            }
            return response;
          }),
      .previousPageCursor_ =
          fmt::format("EARLIER|{}", to_seconds(search_intvl.from_)),
      .nextPageCursor_ = fmt::format("LATER|{}", to_seconds(search_intvl.to_))};
}

}  // namespace motis::odm
