#include "motis/endpoints/one_to_many_post.h"

#include "motis/endpoints/one_to_many.h"

namespace motis::ep {

api::oneToMany_response one_to_many_post::operator()(
    api::OneToManyParams const& query) const {
  return one_to_many_handle_request(query, w_, l_, elevations_);
}

api::oneToManyIntermodal_response one_to_many_intermodal_post::operator()(
    api::OneToManyIntermodalParams const& query) const {
  return run_one_to_many_intermodal(*this, query);
}

}  // namespace motis::ep
