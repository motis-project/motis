#include "motis/endpoints/stop.h"

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "net/bad_request_exception.h"

#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "motis/collect_locations.h"
#include "motis/data.h"
#include "motis/parse_location.h"
#include "motis/place.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/clasz_to_mode.h"

namespace n = nigiri;

namespace motis::ep {

api::stopInfo_response stop::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::stopInfo_params{url.params()};
  auto const& lang = query.language_;

  auto const query_stop = query.stopId_.and_then(
      [&](std::string const& x) { return tags_.find_location(tt_, x); });
  auto const query_center = query.center_.and_then(
      [&](std::string const& x) { return parse_location(x); });

  utl::verify<net::bad_request_exception>(
      query_stop.has_value() ||
          (query_center.has_value() && query.radius_.has_value()),
      "either stopId or center with radius must be provided");

  auto locations = std::vector<n::location_idx_t>{};
  auto const add = [&](n::location_idx_t const l) {
    add_location(tt_, t_, ae_, locations, l, query.exactRadius_);
  };

  if (query_stop.has_value()) {
    locations.emplace_back(tt_.locations_.get_root_idx(*query_stop));
    if (query.radius_) {
      loc_rtree_.in_radius(tt_.locations_.coordinates_[*query_stop],
                           static_cast<double>(*query.radius_), add);
    } else {
      add(*query_stop);
    }
  } else {
    loc_rtree_.in_radius(query_center->pos_,
                         static_cast<double>(*query.radius_), add);
  }

  utl::erase_duplicates(locations);

  auto result = api::stopInfo_response{};
  auto& routes = result.routes_;
  auto seen = n::hash_set<std::string>{};
  auto const first_day = tt_.day_idx(tt_.date_range_.from_);

  for (auto const loc_idx : locations) {
    for (auto const r : tt_.location_routes_[loc_idx]) {
      auto const& range = tt_.route_transport_ranges_[r];
      if (range.empty()) {
        continue;
      }

      // For block-merged transports (same block_id in GTFS), the route can
      // change mid-transport, so the route_id at stop 0 may not match the
      // route_id at the stop where loc_idx actually is. Iterate the route's
      // stop sequence to find the correct stop position(s) for this location,
      // then read route_id at that position. Also handles the case where a
      // single route_idx_t contains transports with different route_id values
      // (same stop sequence, different GTFS route).
      //
      // section_idx(kDep)=stop_idx and section_idx(kArr)=stop_idx-1, so kDep
      // is out of bounds at the last stop — use kArr there instead.
      auto const& stop_seq = tt_.route_location_seq_[r];
      auto const n_stops = static_cast<n::stop_idx_t>(stop_seq.size());
      for (auto stop_pos = n::stop_idx_t{0}; stop_pos != n_stops; ++stop_pos) {
        if (n::stop{stop_seq[stop_pos]}.location_idx() != loc_idx) {
          continue;
        }

        auto const ev = (stop_pos + 1U == n_stops) ? n::event_type::kArr
                                                   : n::event_type::kDep;
        auto seen_rid = n::hash_set<n::route_id_idx_t>{};
        for (auto t_idx = range.from_; t_idx != range.to_; ++t_idx) {
          auto const fr = n::rt::frun{
              tt_, nullptr,
              n::rt::run{.t_ = n::transport{t_idx, first_day},
                         .stop_range_ = {stop_pos, static_cast<n::stop_idx_t>(
                                                       stop_pos + 1U)}}};
          auto const rs = fr[0];
          auto const [route_ids, rid_idx] = rs.get_route(ev);
          if (route_ids == nullptr || !seen_rid.emplace(rid_idx).second) {
            continue;
          }

          auto const route_id = tags_.route_id(rs, ev);
          if (!seen.emplace(route_id).second) {
            continue;
          }

          auto const& agency = rs.get_provider(ev);
          auto const color = rs.get_route_color(ev);

          routes.push_back(
              {.routeId_ = route_id,
               .routeShortName_ = std::string{rs.route_short_name(ev, lang)},
               .routeLongName_ = std::string{rs.route_long_name(ev, lang)},
               .mode_ = to_mode(rs.get_clasz(ev), 5U),
               .agencyId_ =
                   std::string{tt_.strings_.try_get(agency.id_).value_or("?")},
               .agencyName_ = std::string{tt_.translate(lang, agency.name_)},
               .agencyUrl_ = std::string{tt_.translate(lang, agency.url_)},
               .routeColor_ = n::to_str(color.color_),
               .routeTextColor_ = n::to_str(color.text_color_),
               .routeType_ =
                   rs.route_type(ev).and_then([](n::route_type_t const x) {
                     return std::optional{to_idx(x)};
                   })});
        }
      }
    }
  }

  utl::sort(routes, [](api::Route const& a, api::Route const& b) {
    return a.routeShortName_ < b.routeShortName_ ||
           (a.routeShortName_ == b.routeShortName_ && a.routeId_ < b.routeId_);
  });

  if (query_stop.has_value()) {
    result.place_ = to_place(&tt_, &tags_, w_, pl_, matches_, ae_, tz_, lang,
                             tt_location{*query_stop});
  } else {
    result.place_ = to_place(*query_center, "center", std::nullopt);
  }

  return result;
}

}  // namespace motis::ep
