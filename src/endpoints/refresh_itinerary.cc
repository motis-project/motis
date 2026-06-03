#include "motis/endpoints/refresh_itinerary.h"

#include <atomic>
#include <exception>
#include <memory>
#include <string>

#include "google/protobuf/util/json_util.h"

#include "utl/verify.h"

#include "net/base64.h"

#include "nigiri/types.h"

#include "motis/endpoints/routing.h"
#include "motis/endpoints/stop_times.h"
#include "motis/itinerary_id.h"
#include "motis/timetable/modes_to_clasz_mask.h"

#include "itinerary_id.pb.h"

namespace motis::ep {

// Mirrors the `plan` endpoint's profile selection.
nigiri::profile_idx_t leg_alternatives_prf_idx(
    bool const use_routed_transfers,
    bool const require_car_transport,
    api::PedestrianProfileEnum const pedestrian_profile) {
  if (!use_routed_transfers) {
    return nigiri::profile_idx_t{0U};
  }
  if (require_car_transport) {
    return nigiri::kCarProfile;
  }
  return pedestrian_profile == api::PedestrianProfileEnum::WHEELCHAIR
             ? nigiri::kWheelchairProfile
             : nigiri::kFootProfile;
}

template <typename Endpoint>
ep::routing make_routing(Endpoint const& ep) {
  return ep::routing{.config_ = ep.config_,
                     .w_ = ep.w_,
                     .l_ = ep.l_,
                     .pl_ = ep.pl_,
                     .elevations_ = ep.elevations_,
                     .tt_ = &ep.tt_,
                     .tbd_ = ep.tbd_,
                     .tags_ = &ep.tags_,
                     .loc_tree_ = &ep.loc_tree_,
                     .fa_ = ep.fa_,
                     .matches_ = ep.matches_,
                     .way_matches_ = ep.way_matches_,
                     .rt_ = ep.rt_,
                     .shapes_ = ep.shapes_,
                     .gbfs_ = ep.gbfs_,
                     .ae_ = ep.ae_,
                     .tz_ = ep.tz_,
                     .odm_bounds_ = ep.odm_bounds_,
                     .ride_sharing_bounds_ = ep.ride_sharing_bounds_,
                     .metrics_ = ep.metrics_};
}

template <typename Endpoint>
ep::stop_times make_scheduled_stop_times(Endpoint const& ep) {
  static auto const static_rt = std::make_shared<rt>();
  return ep::stop_times{ep.config_, ep.w_,    ep.pl_,   ep.matches_,
                        ep.t_,      ep.ae_,   ep.tz_,   ep.loc_tree_,
                        ep.tt_,     ep.tags_, static_rt};
}

api::Itinerary refresh_itinerary::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::refreshItinerary_params{url.params()};
  auto const stop_times_ep = make_scheduled_stop_times(*this);
  auto const rt = std::atomic_load(&rt_);
  return reconstruct_itinerary(
      make_routing(*this), stop_times_ep, *rt, query.itineraryId_,
      query.requireDisplayNameMatch_, query.joinInterlinedLegs_,
      query.detailedTransfers_.value_or(query.detailedLegs_),
      query.detailedLegs_, query.withScheduledSkippedStops_, query.language_,
      static_cast<std::size_t>(query.numLegAlternatives_),
      to_clasz_mask(query.transitModes_), query.requireBikeTransport_,
      query.requireCarTransport_,
      leg_alternatives_prf_idx(query.useRoutedTransfers_,
                               query.requireCarTransport_,
                               query.pedestrianProfile_),
      make_first_last_mile_options(query));
}

api::Itinerary refresh_itinerary_post::operator()(
    boost::urls::url_view const& url,
    api::RefreshItineraryPostBody const& body) const {
  // Routing params come from the query string; the body carries only the id.
  auto const query =
      api::refreshItinerary_params{url.params(), /*allow_missing*/ true};

  auto parsed = ::motis::ItineraryId{};
  auto const status = google::protobuf::util::JsonStringToMessage(
      boost::json::serialize(boost::json::value_from(body.id_)), &parsed);
  utl::verify(status.ok(), "Failed to decode itinerary-id JSON: {}",
              status.message());
  auto data = std::string{};
  utl::verify(parsed.SerializeToString(&data),
              "failed to serialize itinerary id");

  auto const rt = std::atomic_load(&rt_);
  return reconstruct_itinerary(
      make_routing(*this), make_scheduled_stop_times(*this), *rt,
      net::encode_base64(data), query.requireDisplayNameMatch_,
      query.joinInterlinedLegs_,
      query.detailedTransfers_.value_or(query.detailedLegs_),
      query.detailedLegs_, query.withScheduledSkippedStops_, query.language_,
      static_cast<std::size_t>(query.numLegAlternatives_),
      to_clasz_mask(query.transitModes_), query.requireBikeTransport_,
      query.requireCarTransport_,
      leg_alternatives_prf_idx(query.useRoutedTransfers_,
                               query.requireCarTransport_,
                               query.pedestrianProfile_),
      make_first_last_mile_options(query));
}

}  // namespace motis::ep
