#include "motis/endpoints/map/routes.h"

#include "motis-api/motis-api.h"
#include "motis/data.h"
#include "motis/fwd.h"
#include "motis/railviz.h"
#include "motis/server.h"

namespace motis::ep {

api::routes_response routes::operator()(
    boost::urls::url_view const& url) const {
  auto const api_version = get_api_version(url);
  auto const rt = rt_;
  return get_routes(tags_, tt_, rt->rtt_.get(), shapes_, w_, pl_, matches_, ae_,
                    tz_, *static_.impl_, *rt->railviz_rt_->impl_,
                    api::routes_params{url.params()}, api_version);
}

}  // namespace motis::ep
