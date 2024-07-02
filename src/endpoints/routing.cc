#include "icc/endpoints/routing.h"

#include "osr/routing/profiles/foot.h"
#include "osr/routing/route.h"

#include "icc/parse_location.h"

namespace json = boost::json;
namespace n = nigiri;

namespace icc::ep {

using place_t = std::variant<osr::location, n::location_idx_t>;

place_t to_place(n::timetable const& tt, std::string_view s) {
  auto const location = parse_location(s);
  if (location.has_value()) {
    return *location;
  }
  try {
    return tt.locations_.get(n::location_id{s, n::source_idx_t{}}).l_;
  } catch (...) {
    throw utl::fail("could not find place {}", s);
  }
}

api::plan_response routing::operator()(boost::urls::url_view const& url) const {
  auto const params = url.params();
  auto const query = api::plan_params{params};
  auto const from = to_place(tt_, query.fromPlace_);
  auto const to = to_place(tt_, query.toPlace_);
  return api::plan_response{};
}

}  // namespace icc::ep