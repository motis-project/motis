#include "motis/endpoints/adr/reverse_geocode.h"

#include "net/bad_request_exception.h"

#include "adr/guess_context.h"
#include "adr/reverse.h"
#include "adr/typeahead.h"

#include "motis/config.h"
#include "motis/endpoints/adr/filter_conv.h"
#include "motis/endpoints/adr/suggestions_to_response.h"
#include "motis/feed_labels.h"
#include "motis/parse_location.h"

namespace a = adr;

namespace motis::ep {

constexpr auto const kDefaultResults = 5U;

api::reverseGeocode_response reverse_geocode::operator()(
    boost::urls::url_view const& url) const {
  auto const params = api::reverseGeocode_params{url.params()};
  auto const config_limit = static_cast<std::int64_t>(
      config_.get_limits().reverse_geocode_max_results_);
  auto const requested_limit =
      std::min(params.numResults_.value_or(kDefaultResults), config_limit);
  utl::verify<net::bad_request_exception>(requested_limit >= 1,
                                          "limit must be >= 1");
  auto suggestions = r_.lookup(t_, parse_location((params.place_))->pos_,
                               static_cast<std::size_t>(requested_limit),
                               to_filter_type(params.type_));

  auto const active =
      labels_.active(params.includeLabels_, params.excludeLabels_);
  if ((~active).any()) {
    std::erase_if(suggestions, [&](a::suggestion const& s) {
      auto const* place = std::get_if<a::place_idx_t>(&s.location_);
      if (place == nullptr ||
          t_.place_type_[*place] != a::amenity_category::kExtra) {
        return false;
      }
      return !labels_.is_active(
          active,
          adr_extra_place_idx_t{static_cast<adr_extra_place_idx_t::value_t>(
              *place - t_.ext_start_)});
    });
  }

  return suggestions_to_response(t_, f_, ae_, tt_, tags_, w_, pl_, matches_, {},
                                 {}, suggestions);
}

}  // namespace motis::ep