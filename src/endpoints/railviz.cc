#include "motis/endpoints/railviz.h"

#include "motis-api/motis-api.h"
#include "motis/data.h"
#include "motis/fwd.h"
#include "motis/railviz.h"

namespace motis::ep {

api::railviz_response railviz::operator()(
    boost::urls::url_view const& url) const {
  std::cout << "URL: " << url << "\n";
  auto const rt = rt_;
  return get_trains(tags_, tt_, rt->rtt_.get(), shapes_, *static_.impl_,
                    *rt->railviz_rt_->impl_, api::railviz_params{url.params()});
}

}  // namespace motis::ep