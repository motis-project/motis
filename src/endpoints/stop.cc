#include "motis/endpoints/stop.h"

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "net/bad_request_exception.h"
#include "net/not_found_exception.h"

#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
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

  utl::verify<net::bad_request_exception>(
      query.stopId_.has_value() || query.center_.has_value(),
      "either stopId or center must be provided");

  auto locations = std::vector<n::location_idx_t>{};
  auto query_loc = std::optional<n::location_idx_t>{};

  auto const add = [&](n::location_idx_t const l) {
    add_location(tt_, t_, ae_, locations, l);
  };

  if (query.stopId_.has_value()) {
    auto const loc = tags_.find_location(tt_, *query.stopId_);
    utl::verify<net::not_found_exception>(loc.has_value(), "stop not found: {}",
                                          *query.stopId_);
    query_loc = *loc;
    locations.emplace_back(tt_.locations_.get_root_idx(*loc));
    add(*loc);
  } else {
    auto const center = parse_location(*query.center_);
    utl::verify<net::bad_request_exception>(
        center.has_value(), "invalid center coordinate: {}", *query.center_);
    auto const radius = static_cast<double>(query.radius_.value_or(500LL));
    loc_rtree_.in_radius(center->pos_, radius, add);
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

  if (query_loc.has_value()) {
    result.place_ = to_place(&tt_, &tags_, w_, pl_, matches_, ae_, tz_, lang,
                             tt_location{*query_loc});
  } else {
    auto const center = parse_location(*query.center_);
    result.place_ = to_place(*center, "center", std::nullopt);
  }

  auto const* rtt = rt_->rtt_.get();
  if (query.withAlerts_ && rtt != nullptr && query_loc.has_value()) {
    auto const& al = rtt->alerts_;
    auto seen_alerts = n::hash_set<n::alert_idx_t>{};
    auto alerts = std::vector<api::Alert>{};

    auto const convert_str = [](std::string_view s) {
      return std::optional{std::string{s}};
    };
    auto const get_translation =
        [&](auto const& translations) -> std::optional<std::string> {
      if (translations.empty()) {
        return std::nullopt;
      }
      if (!lang.has_value()) {
        return al.strings_.try_get(translations.front().text_)
            .and_then(convert_str);
      }
      for (auto const& req_lang : *lang) {
        auto const it =
            utl::find_if(translations, [&](n::alert_translation const t) {
              auto const tl = al.strings_.try_get(t.language_);
              return tl.has_value() && tl->starts_with(req_lang);
            });
        if (it != end(translations)) {
          return al.strings_.try_get(it->text_).and_then(convert_str);
        }
      }
      return al.strings_.try_get(translations.front().text_)
          .and_then(convert_str);
    };
    auto const to_time_range = [](n::interval<n::unixtime_t> const x) {
      return api::TimeRange{x.from_, x.to_};
    };
    auto const add_alert = [&](n::alert_idx_t const a) {
      if (!seen_alerts.emplace(a).second) {
        return;
      }
      alerts.push_back(
          {.communicationPeriod_ =
               al.communication_period_[a].empty()
                   ? std::nullopt
                   : std::optional{utl::to_vec(al.communication_period_[a],
                                               to_time_range)},
           .impactPeriod_ = al.impact_period_[a].empty()
                                ? std::nullopt
                                : std::optional{utl::to_vec(
                                      al.impact_period_[a], to_time_range)},
           .cause_ = api::AlertCauseEnum{static_cast<int>(al.cause_[a])},
           .causeDetail_ = get_translation(al.cause_detail_[a]),
           .effect_ = api::AlertEffectEnum{static_cast<int>(al.effect_[a])},
           .effectDetail_ = get_translation(al.effect_detail_[a]),
           .url_ = get_translation(al.url_[a]),
           .headerText_ = get_translation(al.header_text_[a]).value_or(""),
           .descriptionText_ =
               get_translation(al.description_text_[a]).value_or("")});
    };

    for (auto const a : al.location_[*query_loc]) {
      add_alert(a);
    }
    auto const parent = tt_.locations_.parents_[*query_loc];
    if (parent != n::location_idx_t::invalid()) {
      for (auto const a : al.location_[parent]) {
        add_alert(a);
      }
    }

    if (!alerts.empty()) {
      result.place_.alerts_ = std::move(alerts);
    }
  }

  return result;
}

}  // namespace motis::ep
