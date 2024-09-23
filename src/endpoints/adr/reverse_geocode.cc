#include "motis/endpoints/adr/reverse_geocode.h"

#include "adr/guess_context.h"
#include "adr/reverse.h"

#include "motis/endpoints/adr/suggestions_to_response.h"
#include "motis/parse_location.h"

namespace motis::ep {

api::reverseGeocode_response reverse_geocode::operator()(
    boost::urls::url_view const& url) const {
  auto const params = api::reverseGeocode_params{url.params()};
  return suggestions_to_response(
      t_, tt_, {}, {},
      r_.lookup(t_, parse_location((params.place_))->pos_, 5U));
}

}  // namespace motis::ep