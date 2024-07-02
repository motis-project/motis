#include "icc/endpoints/routing.h"

#include "icc/endpoints/graph.h"

#include "osr/routing/profiles/foot.h"
#include "osr/routing/route.h"

namespace json = boost::json;

namespace icc::ep {

api::plan_response routing::operator()(boost::urls::url_view const& url) const {
  auto const query = api::plan_params{url};
  return api::plan_response{};
}

}  // namespace icc::ep