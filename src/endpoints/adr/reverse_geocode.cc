#include "motis/endpoints/adr/reverse_geocode.h"

#include "adr/guess_context.h"
#include "adr/reverse.h"

#include "motis/endpoints/adr/filter_conv.h"
#include "motis/endpoints/adr/suggestions_to_response.h"
#include "motis/parse_location.h"
#include "motis/server.h"

namespace a = adr;

namespace motis::ep {

api::reverseGeocode_response reverse_geocode::operator()(
    boost::urls::url_view const& url) const {
  auto const params = api::reverseGeocode_params{url.params()};
  auto const api_version = get_api_version(url);
  return suggestions_to_response(
      t_, f_, ae_, tt_, tags_, w_, pl_, matches_, {}, {},
      r_.lookup(t_, parse_location((params.place_))->pos_, 5U,
                to_filter_type(params.type_)),
      api_version);
}

}  // namespace motis::ep