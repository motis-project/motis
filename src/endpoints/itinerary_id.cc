#include "motis/endpoints/itinerary_id.h"

#include <atomic>
#include <exception>
#include <memory>
#include <string>

#include "motis/endpoints/stop_times.h"
#include "motis/itinerary_id.h"

namespace motis::ep {

api::Itinerary refresh_itinerary::operator()(
    boost::urls::url_view const& url) const {
  static auto const kStaticOnlyRt = std::make_shared<rt>();
  auto const rt = std::atomic_load(&rt_);
  auto const stop_times_ep =
      ep::stop_times{config_, w_,         pl_, matches_, ae_,
                     tz_,     loc_rtree_, tt_, tags_,    kStaticOnlyRt};
  auto const query = api::refreshItinerary_params{url.params()};
  return reconstruct_itinerary(stop_times_ep, query.itineraryId_, rt.get());
}

}  // namespace motis::ep
