#include "motis/endpoints/stop_routes.h"

#include "utl/erase_duplicates.h"
#include "utl/verify.h"

#include "net/bad_request_exception.h"
#include "net/not_found_exception.h"

#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "motis/parse_location.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/clasz_to_mode.h"

namespace n = nigiri;

namespace motis::ep {

api::stopRoutes_response stop_routes::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::stopRoutes_params{url.params()};
  auto const& lang = query.language_;

  utl::verify<net::bad_request_exception>(
      query.stopId_.has_value() || query.center_.has_value(),
      "either stopId or center must be provided");

  auto locations = std::vector<n::location_idx_t>{};

  auto const add_with_children = [&](n::location_idx_t const l) {
    auto const root = tt_.locations_.get_root_idx(l);
    locations.emplace_back(root);
    for (auto const child : tt_.locations_.children_[root]) {
      locations.emplace_back(child);
      for (auto const grandchild : tt_.locations_.children_[child]) {
        locations.emplace_back(grandchild);
      }
    }
    auto const root_name =
        tt_.get_default_translation(tt_.locations_.names_[root]);
    for (auto const eq : tt_.locations_.equivalences_[root]) {
      if (tt_.get_default_translation(tt_.locations_.names_[eq]) ==
          root_name) {
        locations.emplace_back(eq);
        for (auto const child : tt_.locations_.children_[eq]) {
          locations.emplace_back(child);
        }
      }
    }
  };

  if (query.stopId_.has_value()) {
    auto const loc = tags_.find_location(tt_, *query.stopId_);
    utl::verify<net::not_found_exception>(loc.has_value(),
                                          "stop not found: {}", *query.stopId_);
    add_with_children(*loc);
  } else {
    auto const center = parse_location(*query.center_);
    utl::verify<net::bad_request_exception>(
        center.has_value(), "invalid center coordinate: {}", *query.center_);
    auto const radius =
        static_cast<double>(query.radius_.value_or(500LL));
    loc_rtree_.in_radius(center->pos_, radius,
                         [&](n::location_idx_t const l) { locations.push_back(l); });
  }

  utl::erase_duplicates(locations);

  auto result = api::stopRoutes_response{};
  auto seen = n::hash_set<std::string>{};
  auto const first_day = tt_.day_idx(tt_.date_range_.from_);

  for (auto const loc_idx : locations) {
    for (auto const r : tt_.location_routes_[loc_idx]) {
      auto const& range = tt_.route_transport_ranges_[r];
      if (range.empty()) {
        continue;
      }

      auto const t_idx = range.from_;
      auto const fr = n::rt::frun{
          tt_, nullptr,
          n::rt::run{.t_ = n::transport{t_idx, first_day},
                     .stop_range_ = {n::stop_idx_t{0}, n::stop_idx_t{1}}}};
      auto const rs = fr[0];

      auto const route_id = tags_.route_id(rs, n::event_type::kDep);
      if (!seen.emplace(route_id).second) {
        continue;
      }

      auto const& agency = rs.get_provider(n::event_type::kDep);
      auto const color = rs.get_route_color(n::event_type::kDep);

      result.push_back(
          {.routeId_ = route_id,
           .routeShortName_ =
               std::string{rs.route_short_name(n::event_type::kDep, lang)},
           .routeLongName_ =
               std::string{rs.route_long_name(n::event_type::kDep, lang)},
           .mode_ = to_mode(rs.get_clasz(n::event_type::kDep), 5U),
           .agencyId_ =
               std::string{tt_.strings_.try_get(agency.id_).value_or("?")},
           .agencyName_ = std::string{tt_.translate(lang, agency.name_)},
           .agencyUrl_ = std::string{tt_.translate(lang, agency.url_)},
           .routeColor_ = n::to_str(color.color_),
           .routeTextColor_ = n::to_str(color.text_color_),
           .routeType_ =
               rs.route_type(n::event_type::kDep)
                   .and_then([](n::route_type_t const x) {
                     return std::optional{to_idx(x)};
                   })});
    }
  }

  return result;
}

}  // namespace motis::ep
