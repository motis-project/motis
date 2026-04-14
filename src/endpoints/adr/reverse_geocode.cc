#include "motis/endpoints/adr/reverse_geocode.h"

#include "net/bad_request_exception.h"

#include "adr/guess_context.h"
#include "adr/reverse.h"

#include "motis/config.h"
#include "motis/endpoints/adr/filter_conv.h"
#include "motis/endpoints/adr/suggestions_to_response.h"
#include "motis/parse_location.h"

namespace a = adr;

namespace motis::ep {

api::reverseGeocode_response reverse_geocode::operator()(
    boost::urls::url_view const& url) const {
  auto const params = api::reverseGeocode_params{url.params()};
  auto const config_limit = config_.get_limits().reverse_geocode_max_results_;
  auto const requested_limit = params.limit_.value_or(config_limit);
  utl::verify<net::bad_request_exception>(requested_limit >= 1,
                                          "limit must be >= 1");
  utl::verify<net::bad_request_exception>(
      requested_limit <= config_limit,
      "limit must be <= reverse_geocode_max_results ({})", config_limit);
  auto const result_limit = requested_limit;
  return suggestions_to_response(
      t_, f_, ae_, tt_, tags_, w_, pl_, matches_, {}, {},
      r_.lookup(t_, parse_location((params.place_))->pos_, result_limit,
                to_filter_type(params.type_)));
}

}  // namespace motis::ep