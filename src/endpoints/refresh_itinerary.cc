#include "motis/endpoints/refresh_itinerary.h"

#include <atomic>
#include <exception>
#include <memory>
#include <string>

#include "google/protobuf/util/json_util.h"

#include "utl/verify.h"

#include "net/base64.h"

#include "motis/endpoints/stop_times.h"
#include "motis/itinerary_id.h"
#include "motis/server.h"

#include "itinerary_id.pb.h"

namespace motis::ep {

template <typename Endpoint>
ep::stop_times make_scheduled_stop_times(Endpoint const& ep) {
  static auto const static_rt = std::make_shared<rt>();
  return ep::stop_times{ep.config_, ep.w_,    ep.pl_,        ep.matches_,
                        ep.ae_,     ep.tz_,   ep.loc_rtree_, ep.tt_,
                        ep.tags_,   static_rt};
}

api::Itinerary refresh_itinerary::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::refreshItinerary_params{url.params()};
  auto const stop_times_ep = make_scheduled_stop_times(*this);
  auto const api_version = get_api_version(url);

  auto const rt = std::atomic_load(&rt_);

  return reconstruct_itinerary(
      stop_times_ep, shapes_, *rt, query.itineraryId_,
      query.requireDisplayNameMatch_, query.joinInterlinedLegs_,
      query.detailedTransfers_, query.detailedLegs_, query.withFares_,
      query.withScheduledSkippedStops_, query.language_, api_version);
}

api::Itinerary refresh_itinerary_post::operator()(
    api::RefreshItineraryPostBody const& body) const {
  auto parsed = ::motis::ItineraryId{};
  auto const status = google::protobuf::util::JsonStringToMessage(
      boost::json::serialize(boost::json::value_from(body.id_)), &parsed);
  utl::verify(status.ok(), "Failed to decode itinerary-id JSON: {}",
              status.message());
  auto data = std::string{};
  utl::verify(parsed.SerializeToString(&data),
              "failed to serialize itinerary id");

  auto const stop_times_ep = make_scheduled_stop_times(*this);

  auto const rt = std::atomic_load(&rt_);
  return reconstruct_itinerary(
      stop_times_ep, shapes_, *rt, net::encode_base64(data),
      body.requireDisplayNameMatch_, body.joinInterlinedLegs_,
      body.detailedTransfers_, body.detailedLegs_, body.withFares_,
      body.withScheduledSkippedStops_, body.language_, 5U);
}

}  // namespace motis::ep
