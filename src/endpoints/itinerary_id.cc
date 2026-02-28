#include "motis/endpoints/itinerary_id.h"

#include <string>

#include "boost/json.hpp"

#include "net/bad_request_exception.h"

#include "motis/endpoints/stop_times.h"
#include "motis/itinerary_id.h"

namespace motis::ep {

api::Itinerary refresh_itinerary::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::refreshItinerary_params{url.params()};
  auto const stop_times_ep = ep::stop_times{
      config_, w_, pl_, matches_, ae_, tz_, loc_rtree_, tt_, tags_, nullptr};
  return reconstruct_itinerary(stop_times_ep, query.itineraryId_);
}

}  // namespace motis::ep
