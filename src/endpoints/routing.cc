#include "icc/endpoints/routing.h"

#include "osr/routing/profiles/foot.h"
#include "osr/routing/route.h"

#include "nigiri/routing/query.h"

#include "icc/parse_location.h"

namespace json = boost::json;
namespace n = nigiri;

namespace icc::ep {

using place_t = std::variant<osr::location, n::location_idx_t>;

place_t to_place(n::timetable const& tt, std::string_view s) {
  if (auto const location = parse_location(s); location.has_value()) {
    return *location;
  }
  try {
    return tt.locations_.get(n::location_id{s, n::source_idx_t{}}).l_;
  } catch (...) {
    throw utl::fail("could not find place {}", s);
  }
}

api::plan_response routing::operator()(boost::urls::url_view const& url) const {
  auto const query = api::plan_params{url.params()};

  auto const from = to_place(tt_, query.fromPlace_);
  auto const to = to_place(tt_, query.toPlace_);

  auto const t = get_date_time(query.date_, query.time_);
  auto const window = std::chrono::duration_cast<n::duration_t>(
      std::chrono::seconds{query.searchWindow_ * (query.arriveBy_ ? -1 : 1)});
  auto const start_time = query.timetableView_
                              ? n::routing::start_time_t{n::interval{
                                    query.arriveBy_ ? t - window : t,
                                    query.arriveBy_ ? t : t + window}}
                              : n::routing::start_time_t{t};

  return api::plan_response{};
}

}  // namespace icc::ep