#include "motis/endpoints/one_to_many_post.h"

#include <string_view>

#include "utl/to_vec.h"

#include "motis/endpoints/one_to_many.h"
#include "motis/place.h"

namespace motis::ep {

api::oneToManyPost_response one_to_many_post::operator()(
    api::OneToManyParams const& query) const {
  return one_to_many_handle_request(config_, query, w_, l_, elevations_,
                                    metrics_);
}

api::OneToManyIntermodalResponse one_to_many_intermodal_post::operator()(
    api::OneToManyIntermodalParams const& query) const {
  auto places = std::vector<std::string_view>{query.one_};
  places.insert(end(places), begin(query.many_), end(query.many_));
  verify_locations_exist(&tt_, &tags_, places, {});

  auto const one = get_place(&tt_, &tags_, query.one_);
  auto const many =
      utl::to_vec(query.many_, [&](std::string_view place) -> place_t {
        return get_place(&tt_, &tags_, place);
      });
  return run_one_to_many_intermodal(*this, query, one, many);
}

}  // namespace motis::ep
