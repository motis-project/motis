#include "motis/endpoints/map/route_details.h"

#include "motis-api/motis-api.h"
#include "motis/data.h"
#include "motis/fwd.h"
#include "motis/railviz.h"
#include "motis/server.h"

namespace motis::ep {

api::routeDetails_response route_details::operator()(
    boost::urls::url_view const& url) const {
  auto const api_version = get_api_version(url);
  auto const rt = rt_;
  return get_route_details(tags_, tt_, rt->rtt_.get(), shapes_, w_, pl_,
                           matches_, ae_, tz_, *static_.impl_,
                           *rt->railviz_rt_->impl_,
                           api::routeDetails_params{url.params()}, api_version);
}

}  // namespace motis::ep
