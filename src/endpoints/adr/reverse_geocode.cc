#include "motis/endpoints/adr/reverse_geocode.h"

#include "adr/guess_context.h"
#include "adr/reverse.h"

#include "motis/endpoints/adr/suggestions_to_response.h"
#include "motis/parse_location.h"

namespace a = adr;

namespace motis::ep {

api::reverseGeocode_response reverse_geocode::operator()(
    boost::urls::url_view const& url) const {
  auto const params = api::reverseGeocode_params{url.params()};
  auto filter = a::filter_type::kNone;
  if (params.filterType_.has_value()) {
        switch (*params.filterType_) {
          case api::LocationTypeEnum::ADDRESS:
                filter = a::filter_type::kAddress;
                break;
          case api::LocationTypeEnum::PLACE:
                filter = a::filter_type::kPlace;
                break;
          case api::LocationTypeEnum::STOP:
                filter = a::filter_type::kExtra;
                break;
        }
  }
  return suggestions_to_response(
      t_, tt_, tags_, w_, pl_, matches_, {}, {},
      r_.lookup(t_, parse_location((params.place_))->pos_, 5U, filter), {});
}

}  // namespace motis::ep