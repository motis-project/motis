#include "motis/endpoints/one_to_many_post.h"

#include <string_view>

#include "utl/to_vec.h"

#include "motis/endpoints/one_to_many.h"
#include "motis/place.h"

namespace motis::ep {

api::oneToManyPost_response one_to_many_post::operator()(
    api::OneToManyParams const& query) const {
  return one_to_many_handle_request(query, w_, l_, elevations_);
}

api::oneToManyIntermodal_response one_to_many_intermodal_post::operator()(
    api::OneToManyIntermodalParams const& query) const {
  auto const one = get_place(&tt_, &tags_, query.one_);
  auto const many =
      utl::to_vec(query.many_, [&](std::string_view place) -> place_t {
        return get_place(&tt_, &tags_, place);
      });
  return run_one_to_many_intermodal(*this, query, one, many);
}

}  // namespace motis::ep
