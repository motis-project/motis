#include "motis/endpoints/one_to_many.h"

namespace motis::ep {

api::oneToMany_response one_to_many::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::oneToMany_params{url.params()};
  return one_to_many_handle_request(query, w_, l_, elevations_);
}

}  // namespace motis::ep
