#include "motis/endpoints/one_to_many_post.h"

#include "motis/endpoints/one_to_many.h"

namespace motis::ep {

api::oneToMany_response one_to_many_post::operator()(
    api::OneToManyParams const& query) const {
  return one_to_many_handle_request(query, w_, l_, elevations_);
}

}  // namespace motis::ep
