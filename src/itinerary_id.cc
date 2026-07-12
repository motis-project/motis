#include "motis/itinerary_id.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <ranges>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "boost/json.hpp"
#include "boost/url/url_view.hpp"

#include "fmt/chrono.h"
#include "fmt/format.h"

#include "utl/concat.h"
#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"
#include "utl/verify.h"
#include "utl/visit.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/direct.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/leg_alternatives.h"
#include "nigiri/rt/frun.h"
#include "nigiri/special_stations.h"
#include "nigiri/td_footpath.h"

#include "osr/location.h"

#include "net/bad_request_exception.h"
#include "net/base64.h"

#include "motis/constants.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"
#include "motis/endpoints/stop_times.h"
#include "motis/flex/mode_id.h"
#include "motis/gbfs/routing_data.h"
#include "motis/journey_to_response.h"
#include "motis/osr/mode_to_profile.h"
#include "motis/osr/parameters.h"
#include "motis/parse_location.h"
#include "motis/place.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/clasz_to_mode.h"
#include "motis/timetable/time_conv.h"
#include "motis/transport_mode_ids.h"

#include "itinerary_id.pb.h"

namespace n = nigiri;
namespace json = boost::json;

namespace motis {

using proto_id_t = ItineraryId;
using proto_leg_t = LegId;

constexpr auto kItineraryIdApiVersion = 5U;

constexpr auto kTimeMul = 1.0 / 60.0 * 7;
constexpr auto kSearchRadiusMeters = 100;
constexpr auto kLookbackSeconds = std::int64_t{60 * 60};
constexpr auto kNonSchedAllowedDeviationSeconds = std::int64_t{30 * 60};

constexpr auto kExactTripIdMatchAddScore = 50.0;

// Backward compatibility
void join_interlined(proto_id_t& id) {
  auto const is_transit = [](proto_leg_t const& l) {
    return !l.trip_id().empty();
  };

  auto out = proto_id_t{};
  for (auto i = 0; i < id.legs_size(); ++i) {
    auto const& cur = id.legs(i);
    if (out.legs_size() > 0 && is_transit(out.legs(out.legs_size() - 1)) &&
        is_transit(cur)) {
      // Interlined continuation: extend the previous transit leg to cur's exit.
      auto* const prev = out.mutable_legs(out.legs_size() - 1);
      prev->set_to_id(cur.to_id());
      prev->set_to_lat(cur.to_lat());
      prev->set_to_lon(cur.to_lon());
      cur.has_to_level() ? prev->set_to_level(cur.to_level())
                         : prev->clear_to_level();
      prev->set_sched_end(cur.sched_end());
      prev->set_scheduled(prev->scheduled() && cur.scheduled());
    } else {
      *out.add_legs() = cur;
    }
  }
  id = std::move(out);
}

proto_id_t decode_itinerary_id(std::string const& id) {
  auto parsed = proto_id_t{};
  auto const data = net::decode_base64(id);
  utl::verify<net::bad_request_exception>(parsed.ParseFromString(data),
                                          "Failed to decode itinerary-id");
  join_interlined(parsed);
  return parsed;
}

osr::level_t proto_to_level(proto_leg_t const& leg, bool const from) {
  if (from) {
    return leg.has_from_level()
               ? osr::level_t{static_cast<float>(leg.from_level())}
               : osr::kNoLevel;
  }
  return leg.has_to_level() ? osr::level_t{static_cast<float>(leg.to_level())}
                            : osr::kNoLevel;
}

std::optional<double> level_to_api(osr::level_t const lvl) {
  return lvl.has_level() ? std::optional<double>{lvl.to_float()} : std::nullopt;
}

struct leg_hint {
  explicit leg_hint(proto_leg_t const& l)
      : display_name_{l.display_name()},
        trip_id_{l.trip_id()},
        from_stop_id_{l.from_id()},
        from_loc_{{l.from_lat(), l.from_lon()}, proto_to_level(l, true)},
        to_stop_id_{l.to_id()},
        to_loc_{{l.to_lat(), l.to_lon()}, proto_to_level(l, false)},
        sched_start_{l.sched_start()},
        sched_end_{l.sched_end()},
        mode_{json::value_to<api::ModeEnum>(json::value{l.mode()})},
        scheduled_{l.scheduled()} {}

  bool is_public_transport() const { return !trip_id_.empty(); }

  std::string display_name_;
  std::string trip_id_;
  std::string from_stop_id_;
  osr::location from_loc_;
  std::string to_stop_id_;
  osr::location to_loc_;
  std::int64_t sched_start_;
  std::int64_t sched_end_;
  api::ModeEnum mode_;
  bool scheduled_;
};

api::Leg make_dummy_leg(leg_hint const& lh, std::string error) {
  auto const to_dt = [](std::int64_t const s) {
    return openapi::date_time_t{
        std::chrono::sys_seconds{std::chrono::seconds{s}}};
  };
  auto const start = to_dt(lh.sched_start_);
  auto const end = to_dt(lh.sched_end_);

  return api::Leg{
      .mode_ = lh.mode_,
      .from_ = {.stopId_ = lh.from_stop_id_,
                .lat_ = lh.from_loc_.pos_.lat_,
                .lon_ = lh.from_loc_.pos_.lng_,
                .level_ = level_to_api(lh.from_loc_.lvl_),
                .departure_ = start,
                .scheduledDeparture_ = start,
                .cancelled_ = true},
      .to_ = {.stopId_ = lh.to_stop_id_,
              .lat_ = lh.to_loc_.pos_.lat_,
              .lon_ = lh.to_loc_.pos_.lng_,
              .level_ = level_to_api(lh.to_loc_.lvl_),
              .arrival_ = end,
              .scheduledArrival_ = end,
              .cancelled_ = true},
      .duration_ = std::max(std::int64_t{0}, lh.sched_end_ - lh.sched_start_),
      .startTime_ = start,
      .endTime_ = end,
      .scheduledStartTime_ = start,
      .scheduledEndTime_ = end,
      .realTime_ = false,
      .scheduled_ = lh.scheduled_,
      .displayName_ = lh.display_name_,
      .cancelled_ = true,
      .alerts_ = {{api::Alert{.headerText_ = std::move(error)}}},
  };
}

proto_leg_t make_leg_id(std::string const& display_name,
                        std::optional<std::string> const& trip_id,
                        api::Place const& from,
                        api::Place const& to,
                        openapi::date_time_t const sched_start,
                        openapi::date_time_t const sched_end,
                        api::ModeEnum const mode,
                        bool const scheduled) {
  if (trip_id.has_value()) {
    utl::verify(from.stopId_.has_value(),
                "itinerary id: PT leg missing 'from' stopId");
    utl::verify(to.stopId_.has_value(),
                "itinerary id: PT leg missing 'to' stopId");
  }
  auto id = proto_leg_t{};
  id.set_display_name(display_name);
  id.set_trip_id(trip_id.value_or(""));
  id.set_from_id(from.stopId_.value_or(""));
  id.set_to_id(to.stopId_.value_or(""));
  id.set_from_lat(from.lat_);
  id.set_from_lon(from.lon_);
  id.set_to_lat(to.lat_);
  id.set_to_lon(to.lon_);
  if (from.level_.has_value()) {
    id.set_from_level(*from.level_);
  }
  if (to.level_.has_value()) {
    id.set_to_level(*to.level_);
  }
  id.set_sched_start(sched_start.get_unixtime_seconds());
  id.set_sched_end(sched_end.get_unixtime_seconds());
  id.set_mode(std::string{json::value_from(mode).as_string()});
  id.set_scheduled(scheduled);
  return id;
}

std::string generate_itinerary_id(n::routing::journey const& j,
                                  tag_lookup const& tags,
                                  n::timetable const& tt,
                                  n::rt_timetable const* rtt,
                                  osr::ways const* w,
                                  osr::platforms const* pl,
                                  platform_matches_t const* matches,
                                  adr_ext const* ae,
                                  tz_map_t const* tz_map,
                                  place_t const& start,
                                  place_t const& dest) {
  utl::verify(!j.legs_.empty(), "generate_itinerary_id expects at least 1 leg");

  auto id = proto_id_t{};

  auto const place = [&](auto const& loc) {
    return to_place(&tt, &tags, w, pl, matches, ae, tz_map, n::lang_t{}, loc,
                    start, dest);
  };
  auto const make_non_pt_leg = [&](n::routing::journey::leg const& jl,
                                   api::ModeEnum const mode) {
    return make_leg_id("", std::nullopt, place(tt_location{jl.from_}),
                       place(tt_location{jl.to_}), jl.dep_time_, jl.arr_time_,
                       mode, true);
  };

  for (auto const& jl : j.legs_) {
    id.mutable_legs()->Add(utl::visit(
        jl.uses_,
        [&](n::routing::journey::run_enter_exit const& rex) {
          auto const fr = n::rt::frun{tt, rtt, rex.r_};
          auto const enter = fr[rex.stop_range_.from_];
          auto const exit = fr[rex.stop_range_.to_ - 1U];
          return make_leg_id(
              std::string{enter.display_name(n::event_type::kDep, n::lang_t{})},
              tags.id(tt, enter, n::event_type::kDep), place(enter),
              place(exit), enter.scheduled_time(n::event_type::kDep),
              exit.scheduled_time(n::event_type::kArr),
              to_mode(enter.get_clasz(n::event_type::kDep),
                      kItineraryIdApiVersion),
              fr.is_scheduled());
        },
        [&](n::routing::offset const& o) {
          auto const id = o.type();
          auto const mode = to_mode(id);
          return make_non_pt_leg(jl, mode);
        },
        [&](n::footpath const&) {
          return make_non_pt_leg(jl, api::ModeEnum::WALK);
        }));
  }

  auto data = std::string{};
  utl::verify(id.SerializeToString(&data), "failed to serialize itinerary id");
  return net::encode_base64(data);
}

struct st_candidate {
  friend bool operator==(st_candidate const& a, st_candidate const& b) {
    return std::tie(a.run_.t_, a.run_.rt_) == std::tie(b.run_.t_, b.run_.rt_);
  }

  friend bool operator<(st_candidate const& a, st_candidate const& b) {
    return std::tie(a.run_.t_, a.run_.rt_) < std::tie(b.run_.t_, b.run_.rt_);
  }

  n::rt::frun run_;
  std::optional<openapi::date_time_t> scheduled_arrival_{};
  std::optional<openapi::date_time_t> scheduled_departure_{};
};

std::vector<st_candidate> get_st_candidates_in_radius(
    ep::stop_times const& st_ep,
    geo::latlng const& center,
    std::int64_t const sched_start,
    api::ModeEnum const mode,
    std::int64_t const window,
    int const radius_m,
    bool const arrive_by,
    n::lang_t const& lang) {
  auto query = api::stoptimes_params{};
  query.center_ = fmt::format("{},{}", center.lat_, center.lng_);
  query.time_ = openapi::date_time_t{
      std::chrono::sys_seconds{std::chrono::seconds{sched_start}}};
  query.arriveBy_ = arrive_by;
  query.direction_ = api::directionEnum::LATER;
  query.window_ = window;
  query.radius_ = radius_m;
  query.exactRadius_ = true;
  query.mode_ = {mode};
  query.language_ = lang;

  auto const ev_type =
      query.arriveBy_ ? n::event_type::kArr : n::event_type::kDep;
  auto const query_stop = query.stopId_.and_then([&](std::string const& x) {
    return st_ep.tags_.find_location(st_ep.tt_, x);
  });
  auto const query_center = query.center_.and_then(
      [&](std::string const& x) { return parse_location(x); });

  auto const events =
      st_ep.get_runs(query, nullptr, ev_type, query_stop, query_center, true);

  return utl::to_vec(events, [&](n::rt::run const r) -> st_candidate {
    auto const fr = n::rt::frun{st_ep.tt_, nullptr, r};
    auto const s = fr[0];

    auto res = st_candidate{.run_ = fr};

    if (fr.stop_range_.from_ != 0U) {
      res.scheduled_arrival_ = {s.scheduled_time(n::event_type::kArr)};
    }
    if (fr.stop_range_.from_ != fr.size() - 1U) {
      res.scheduled_departure_ = {s.scheduled_time(n::event_type::kDep)};
    }

    return res;
  });
}

n::rt::frun make_full_frun(n::timetable const& tt,
                           n::rt_timetable const* rtt,
                           n::rt::run run) {
  auto fr = n::rt::frun{tt, rtt, run};
  fr.stop_range_ = {0U, fr.size()};
  return fr;
}

n::rt::frun make_frun_from_trip_id(tag_lookup const& tags,
                                   n::timetable const& tt,
                                   n::rt_timetable const* rtt,
                                   std::string_view const trip_id) {
  auto const [run, _] = tags.get_trip(tt, rtt, trip_id);
  return make_full_frun(tt, rtt, run);
}

std::optional<n::stop_idx_t> find_stop_by_location_time(
    n::rt::frun const& fr,
    n::location_idx_t const loc,
    std::int64_t const target_sec,
    n::event_type const ev_type,
    std::int64_t const allowed_deviation_sec = 0) {
  for (auto i = n::stop_idx_t{0U}; i < fr.size(); ++i) {
    auto const rs = fr[i];
    if (rs.get_location_idx() == loc &&
        std::abs(to_seconds(rs.scheduled_time(ev_type)) - target_sec) <=
            allowed_deviation_sec) {
      return rs.stop_idx_;
    }
  }
  return std::nullopt;
}

std::optional<n::stop_idx_t> find_stop_by_id_time(
    n::rt::frun const& fr,
    tag_lookup const& tags,
    std::string_view const stop_id,
    std::int64_t const target_sec,
    n::event_type const ev_type,
    std::int64_t const allowed_deviation_sec) {
  auto const loc = tags.find_location(*fr.tt_, stop_id);
  if (!loc.has_value()) {
    return std::nullopt;
  }
  return find_stop_by_location_time(fr, *loc, target_sec, ev_type,
                                    allowed_deviation_sec);
}

constexpr auto kWalkTransportModeId = static_cast<n::transport_mode_id_t>(
    static_cast<std::underlying_type_t<osr::search_profile>>(
        osr::search_profile::kFoot));

struct candidate_score {
  bool operator<(candidate_score const& o) const {
    return *candidate_ < *o.candidate_ ||
           (*candidate_ == *o.candidate_ && score_ > o.score_);
  }

  st_candidate const* candidate_;
  double score_;
};

int candidate_cmp(candidate_score const& a, candidate_score const& b) {
  if (*a.candidate_ == *b.candidate_) {
    return 0;
  }
  return a < b ? -1 : 1;
}

struct from_to_candidate {
  st_candidate const* from_;
  st_candidate const* to_;
};

std::optional<from_to_candidate> get_best_candidate(
    std::vector<st_candidate> const& from_resp,
    std::vector<st_candidate> const& to_resp,
    leg_hint const& hint,
    bool const require_display_name_match,
    tag_lookup const& tags) {
  if (from_resp.empty() || to_resp.empty()) {
    return std::nullopt;
  }

  auto from_cands = std::vector<candidate_score>{};
  auto to_cands = std::vector<candidate_score>{};
  from_cands.reserve(from_resp.size());
  to_cands.reserve(to_resp.size());

  auto const [hint_trip_run, _] =
      tags.get_trip(*from_resp.front().run_.tt_, nullptr, hint.trip_id_);

  for (auto const& st : from_resp) {
    if (!st.scheduled_departure_.has_value()) {
      continue;
    }

    if (require_display_name_match &&
        hint.display_name_ !=
            st.run_[0].display_name(n::event_type::kDep, n::lang_t{})) {
      continue;
    }

    from_cands.emplace_back(
        &st,
        -geo::distance(st.run_[0].pos(), hint.from_loc_.pos_) -
            kTimeMul *
                std::abs(
                    hint.sched_start_ -
                    st.scheduled_departure_.value().get_unixtime_seconds()));
  }

  if (from_cands.size() == 0) {
    return std::nullopt;
  }

  for (auto const& st : to_resp) {
    if (!st.scheduled_arrival_.has_value()) {
      continue;
    }

    auto const same_trip =
        hint_trip_run.t_.is_valid() && st.run_.t_ == hint_trip_run.t_;

    to_cands.emplace_back(
        &st,
        (same_trip ? kExactTripIdMatchAddScore : 0.0) -
            geo::distance(st.run_[0].pos(), hint.to_loc_.pos_) -
            kTimeMul *
                std::abs(hint.sched_end_ -
                         st.scheduled_arrival_.value().get_unixtime_seconds()));
  }

  utl::sort(from_cands);
  utl::sort(to_cands);

  auto best_from_to = std::optional<from_to_candidate>{};
  auto best_score = std::numeric_limits<double>::lowest();

  for (auto i_from = 0U, i_to = 0U;
       i_from < from_cands.size() && i_to < to_cands.size();) {
    switch (candidate_cmp(from_cands[i_from], to_cands[i_to])) {
      case -1: ++i_from; break;
      case +1: ++i_to; break;
      case 0:
        auto score = from_cands[i_from].score_ + to_cands[i_to].score_;
        if (score > best_score) {
          best_score = score;
          best_from_to.emplace(from_cands[i_from].candidate_,
                               to_cands[i_to].candidate_);
        }
        ++i_from;
        ++i_to;
        while (i_from < from_cands.size() && i_to < to_cands.size() &&
               candidate_cmp(from_cands[i_from], from_cands[i_from - 1]) == 0 &&
               candidate_cmp(from_cands[i_from], to_cands[i_to]) == 0) {
          ++i_from;
          ++i_to;
        }
        break;
    }
  }

  return best_from_to;
}

first_last_mile_options make_first_last_mile_options(
    api::refreshItinerary_params const& q) {
  return first_last_mile_options{
      .pedestrian_profile_ = q.pedestrianProfile_,
      .elevation_costs_ = q.elevationCosts_,
      .osr_params_ = get_osr_parameters(q),
      .max_matching_distance_ = q.maxMatchingDistance_,
      .max_pre_transit_ = std::chrono::seconds{q.maxPreTransitTime_},
      .max_post_transit_ = std::chrono::seconds{q.maxPostTransitTime_},
      .pre_transit_ = {.form_factors_ = q.preTransitRentalFormFactors_,
                       .propulsion_types_ = q.preTransitRentalPropulsionTypes_,
                       .providers_ = q.preTransitRentalProviders_,
                       .provider_groups_ = q.preTransitRentalProviderGroups_,
                       .ignore_return_constraints_ =
                           q.ignorePreTransitRentalReturnConstraints_},
      .post_transit_ = {.form_factors_ = q.postTransitRentalFormFactors_,
                        .propulsion_types_ =
                            q.postTransitRentalPropulsionTypes_,
                        .providers_ = q.postTransitRentalProviders_,
                        .provider_groups_ = q.postTransitRentalProviderGroups_,
                        .ignore_return_constraints_ =
                            q.ignorePostTransitRentalReturnConstraints_},
      .pre_transit_modes_ = q.preTransitModes_,
      .post_transit_modes_ = q.postTransitModes_};
}

n::routing::journey::leg make_pt_leg(n::rt::frun const& fr,
                                     n::stop_idx_t const from,
                                     n::stop_idx_t const to) {
  auto const rs_from = n::rt::run_stop{&fr, from};
  auto const rs_to = n::rt::run_stop{&fr, to};
  return n::routing::journey::leg{
      n::direction::kForward,
      rs_from.get_location_idx(),
      rs_to.get_location_idx(),
      rs_from.time(n::event_type::kDep),
      rs_to.time(n::event_type::kArr),
      n::routing::journey::run_enter_exit{fr, from, to}};
}

std::expected<n::routing::journey::leg, std::string> reconstruct_pt_leg(
    leg_hint const& lh,
    ep::stop_times const& stop_times_ep,
    n::rt_timetable const* rtt,
    n::lang_t const& lang,
    bool const require_display_name_match) {
  auto const fail = [](std::string_view const msg) {
    return std::unexpected{std::string{msg}};
  };

  if (!lh.scheduled_) {
    if (lh.trip_id_.empty()) {
      return fail("reconstruct_itinerary: additional trip requires trip_id");
    }

    auto const fr = make_frun_from_trip_id(stop_times_ep.tags_,
                                           stop_times_ep.tt_, rtt, lh.trip_id_);
    if (!fr.valid()) {
      return fail("reconstruct_itinerary: additional trip not found");
    }
    if (fr.is_scheduled()) {
      return fail(
          "reconstruct_itinerary: trip_id resolved to scheduled trip while "
          "itinerary id expects additional trip");
    }

    auto const from_idx = find_stop_by_id_time(
        fr, stop_times_ep.tags_, lh.from_stop_id_, lh.sched_start_,
        n::event_type::kDep, kNonSchedAllowedDeviationSeconds);
    if (!from_idx.has_value()) {
      return fail("reconstruct_itinerary: additional trip from stop not found");
    }

    auto const to_idx = find_stop_by_id_time(
        fr, stop_times_ep.tags_, lh.to_stop_id_, lh.sched_end_,
        n::event_type::kArr, kNonSchedAllowedDeviationSeconds);
    if (!to_idx.has_value()) {
      return fail("reconstruct_itinerary: additional trip to stop not found");
    }

    if (*from_idx >= *to_idx) {
      return fail("reconstruct_itinerary: invalid stop order (from >= to)");
    }

    return make_pt_leg(fr, *from_idx, *to_idx);
  } else {
    auto const from_st_res = get_st_candidates_in_radius(
        stop_times_ep, lh.from_loc_.pos_, lh.sched_start_ - kLookbackSeconds,
        lh.mode_, kLookbackSeconds * 2, kSearchRadiusMeters, false, lang);
    auto const to_st_res = get_st_candidates_in_radius(
        stop_times_ep, lh.to_loc_.pos_, lh.sched_end_ - kLookbackSeconds,
        lh.mode_, kLookbackSeconds * 2, kSearchRadiusMeters, true, lang);

    auto const best_from_to =
        get_best_candidate(from_st_res, to_st_res, lh,
                           require_display_name_match, stop_times_ep.tags_);

    if (!best_from_to.has_value()) {
      return fail("no matching route is found");
    }

    // Rebuild with the current RT snapshot
    auto const best_fr =
        make_full_frun(stop_times_ep.tt_, rtt, best_from_to->from_->run_);
    auto const from_idx = best_from_to->from_->run_.stop_range_.from_;
    auto const to_idx = best_from_to->to_->run_.stop_range_.from_;
    if (!best_fr.stop_range_.contains(from_idx) ||
        !best_fr.stop_range_.contains(to_idx)) {
      return fail("reconstruct_itinerary: winning stop index out of range");
    }
    if (from_idx >= to_idx) {
      return fail("reconstruct_itinerary: invalid stop order (from >= to)");
    }

    return make_pt_leg(best_fr, from_idx, to_idx);
  }
}

n::unixtime_t to_unix(openapi::date_time_t const& x) {
  return std::chrono::time_point_cast<n::i32_minutes>(x.time_);
}

n::unixtime_t sched_to_unix(std::int64_t const s) {
  return n::unixtime_t{std::chrono::duration_cast<n::unixtime_t::duration>(
      std::chrono::seconds{s})};
}

// Verifies the decoded itinerary-id legs match the expected shape:
// [first-mile offset] TRANSIT [, TRANSFER, TRANSIT]... [last-mile offset]
template <typename Leg>
void verify_leg_structure(std::vector<Leg> const& legs) {
  utl::verify<net::bad_request_exception>(!legs.empty(),
                                          "itinerary id: empty leg structure");
  utl::verify<net::bad_request_exception>(
      utl::any_of(legs,
                  [](auto&& l) { return l.input_.is_public_transport(); }),
      "itinerary id: no transit leg found");

  for (auto i = std::size_t{0U}; i != legs.size(); ++i) {
    auto const is_transit = legs[i].input_.is_public_transport();
    utl::verify<net::bad_request_exception>(
        i == 0U || is_transit != legs[i - 1U].input_.is_public_transport(),
        "itinerary id: invalid leg structure (two consecutive {} legs)",
        is_transit ? "transit" : "non-transit");
  }
}

api::Itinerary reconstruct_itinerary(
    ep::routing const& routing,
    ep::stop_times const& stop_times_ep,
    rt const& rt,
    std::string const& id_buf,
    bool const require_display_name_match,
    bool const join_interlined_legs,
    bool const detailed_transfers,
    bool const detailed_legs,
    bool const with_scheduled_skipped_stops,
    n::lang_t const& lang,
    std::size_t const num_leg_alternatives,
    n::routing::clasz_mask_t const allowed_claszes,
    bool const require_bike_transport,
    bool const require_car_transport,
    n::profile_idx_t const prf_idx,
    first_last_mile_options const& flm) {
  struct leg {
    leg_hint input_;

    // Resolved leg(s):
    // might be multiple, e.g. for RENTAL, FLEX, TRANSIT with interlining
    std::vector<api::Leg> output_{};

    // For transit legs: the reconstructed journey leg (run + times), used both
    // to render the leg and as the prev/next context for a neighbour's leg
    // alternatives. Empty if reconstruction failed (-> dummy leg).
    std::optional<n::routing::journey::leg> transit_{};

    // Resolved endpoint locations. Transit legs take these from `transit_`;
    // offset/transfer legs inherit them from the adjacent transit leg.
    n::location_idx_t from_{n::location_idx_t::invalid()};
    n::location_idx_t to_{n::location_idx_t::invalid()};

    // For transfer/offset legs: adjusted times so
    // - first: arr aligns with dep of following transit leg
    // - rest: dep aligns with arr of previous transit leg
    n::unixtime_t dep_{}, arr_{};

    // For offset legs:
    // all offsets for alternative lookup
    std::vector<n::routing::offset> offsets_{};
    n::routing::td_offsets_t td_offsets_{};
  };

  auto const non_dummy_match_mode =
      routing.is_osr_loaded() ? n::routing::location_match_mode::kIntermodal
                              : n::routing::location_match_mode::kEquivalent;

  auto const safe_prf_idx =
      stop_times_ep.tt_.locations_.footpaths_out_.at(prf_idx).empty()
          ? n::profile_idx_t{0U}
          : prf_idx;

  auto stats = ep::stats_map_t{};
  auto gbfs_rd = gbfs::gbfs_routing_data{routing.w_, routing.l_, routing.gbfs_};
  auto cache = street_routing_cache_t{};

  // needed to mark blocked elevator nodes in street routing
  auto blocked = osr::bitvec<osr::node_idx_t>{};
  if (routing.is_osr_loaded()) {
    blocked.resize(stop_times_ep.w_->n_nodes());
  }

  // === Helpers ===
  auto const is_transit = [](leg const& l) {
    return !l.input_.trip_id_.empty();
  };
  auto const failed = [](leg const& l) { return !l.transit_.has_value(); };

  auto const get_offsets = [&](leg_hint const& h, bool const is_start,
                               std::vector<api::ModeEnum> const& modes) {
    if (!routing.is_osr_loaded()) {
      return std::vector<n::routing::offset>{};
    }
    auto const& pos = is_start ? h.from_loc_ : h.to_loc_;
    auto const dir =
        is_start ? osr::direction::kForward : osr::direction::kBackward;
    return routing.get_offsets(
        rt.rtt_.get(), place_t{pos}, dir, modes,
        is_start ? flm.pre_transit_ : flm.post_transit_, flm.osr_params_,
        flm.pedestrian_profile_, flm.elevation_costs_,
        is_start ? flm.max_pre_transit_ : flm.max_post_transit_,
        flm.max_matching_distance_, gbfs_rd, stats);
  };

  auto const get_td_offsets = [&](leg_hint const& h, bool const is_start,
                                  n::unixtime_t const anchor_time,
                                  std::vector<api::ModeEnum> const& modes) {
    if (!routing.is_osr_loaded()) {
      return n::routing::td_offsets_t{};
    }
    auto const& pos = is_start ? h.from_loc_ : h.to_loc_;
    auto const dir =
        is_start ? osr::direction::kForward : osr::direction::kBackward;
    return routing.get_td_offsets(
        rt.rtt_.get(), rt.e_.get(), place_t{pos}, dir, modes, flm.osr_params_,
        flm.pedestrian_profile_, flm.elevation_costs_,
        flm.max_matching_distance_,
        is_start ? flm.max_pre_transit_ : flm.max_post_transit_, anchor_time,
        stats);
  };

  auto const reconstruct = [&](n::routing::journey::leg const& l,
                               place_t const& from_place,
                               place_t const& to_place,
                               alternatives_context const& alternatives = {}) {
    return journey_to_response(
               stop_times_ep.w_, routing.l_, stop_times_ep.pl_,
               stop_times_ep.tt_, stop_times_ep.tags_, routing.fa_, rt.e_.get(),
               rt.rtt_.get(), stop_times_ep.matches_, routing.elevations_,
               routing.shapes_, gbfs_rd, stop_times_ep.ae_, stop_times_ep.tz_,
               {.legs_ = {l},
                .start_time_ = l.dep_time_,
                .dest_time_ = l.arr_time_,
                .dest_ = l.to_,
                .transfers_ = 0},
               from_place, to_place, cache, &blocked, prf_idx == n::kCarProfile,
               flm.osr_params_, flm.pedestrian_profile_, flm.elevation_costs_,
               join_interlined_legs, detailed_transfers, detailed_legs,
               /*with_fares=*/false, with_scheduled_skipped_stops,
               stop_times_ep.config_.timetable_.value().max_matching_distance_,
               flm.max_matching_distance_, 6U, false, false, lang,
               /*set_itinerary_id_field=*/false, alternatives)
        .legs_;
  };

  auto const reconstruct_offset =
      [&](std::vector<leg> const& legs,
          std::size_t const i) -> std::vector<api::Leg> {
    auto const& l = legs[i];
    auto const& h = l.input_;
    auto const is_first = i == 0U;
    auto const loc = is_first ? l.to_ : l.from_;
    if (loc == n::location_idx_t::invalid()) {
      return {
          make_dummy_leg(h, "adjacent transit leg couldn't be reconstructed")};
    }

    auto const time = is_first ? l.arr_ : l.dep_;
    auto const side =
        is_first ? n::routing::side::kBoarding : n::routing::side::kAlighting;
    auto const offset =
        n::routing::lookup_offset(loc, time, side, l.offsets_, l.td_offsets_);

    if (!offset.has_value()) {
      return {make_dummy_leg(h, "no offset found")};
    }

    auto const start_place =
        is_first ? place_t{h.from_loc_} : place_t{tt_location{loc}};
    auto const end_place =
        is_first ? place_t{tt_location{loc}} : place_t{h.to_loc_};
    return reconstruct(*offset, start_place, end_place);
  };

  auto const reconstruct_transfer = [&](leg const& l) -> std::vector<api::Leg> {
    auto const& h = l.input_;
    if (l.from_ == n::location_idx_t::invalid() ||
        l.to_ == n::location_idx_t::invalid()) {
      return {
          make_dummy_leg(h, "adjacent transit leg couldn't be reconstructed")};
    }

    auto q = n::routing::query{};
    q.prf_idx_ = safe_prf_idx;
    auto const offs = std::vector<n::routing::offset>{
        {(l.to_), n::duration_t{0}, kWalkTransportModeId}};

    auto const fp_leg = n::routing::lookup_footpath(
        l.from_, l.dep_, n::routing::side::kAlighting, stop_times_ep.tt_,
        rt.rtt_.get(), q, offs, n::routing::location_match_mode::kExact,
        /*use_footpaths=*/true);
    if (!fp_leg.has_value()) {
      return {make_dummy_leg(h, "reconstruct_itinerary: no transfer footpath")};
    }
    return reconstruct(*fp_leg, tt_location{(l.from_)}, tt_location{(l.to_)});
  };

  auto const get_leg_alternatives = [&](std::vector<leg> const& legs,
                                        std::size_t const i,
                                        bool const has_prev_transit,
                                        bool const has_next_transit) {
    auto const& jl = *legs[i].transit_;

    auto const add_equivalents = [&](std::vector<n::routing::offset>& offsets,
                                     n::location_idx_t const l) {
      n::routing::for_each_meta(
          stop_times_ep.tt_, n::routing::location_match_mode::kEquivalent, l,
          [&](n::location_idx_t const c) {
            offsets.emplace_back(c, n::duration_t{0U}, 0U);
          });
    };

    auto q = n::routing::make_alternative_query(
        stop_times_ep.tt_, rt.rtt_.get(),
        n::routing::query{.prf_idx_ = safe_prf_idx,
                          .allowed_claszes_ = allowed_claszes,
                          .require_bike_transport_ = require_bike_transport,
                          .require_car_transport_ = require_car_transport},
        has_prev_transit ? legs[i - 2U].to_ : n::location_idx_t::invalid(),
        has_next_transit ? legs[i + 2U].from_ : n::location_idx_t::invalid());

    if (!has_prev_transit) {
      q.start_match_mode_ = non_dummy_match_mode;
      q.use_start_footpaths_ = !routing.is_osr_loaded();
      if (routing.is_osr_loaded()) {
        auto offsets = get_offsets(legs.front().input_, /*is_start=*/true,
                                   flm.pre_transit_modes_);
        if (is_transit(legs.front())) {
          add_equivalents(offsets, jl.from_);
        }
        q.start_ = std::move(offsets);
        q.td_start_ = get_td_offsets(legs.front().input_, /*is_start=*/true,
                                     sched_to_unix(legs[i].input_.sched_start_),
                                     flm.pre_transit_modes_);
      } else {
        q.start_ = {{jl.from_, n::duration_t{0U}, 0U}};
      }
    }

    if (!has_next_transit) {
      q.dest_match_mode_ = non_dummy_match_mode;
      if (routing.is_osr_loaded()) {
        auto offsets = get_offsets(legs.back().input_, /*is_start=*/false,
                                   flm.post_transit_modes_);
        if (is_transit(legs.back())) {
          add_equivalents(offsets, jl.to_);
        }
        q.destination_ = std::move(offsets);
        q.td_dest_ = get_td_offsets(legs.back().input_, /*is_start=*/false,
                                    sched_to_unix(legs[i].input_.sched_end_),
                                    flm.post_transit_modes_);
      } else {
        q.destination_ = {{jl.to_, n::duration_t{0U}, 0U}};
      }
    }

    auto const& original =
        std::get<n::routing::journey::run_enter_exit>(jl.uses_);
    auto const next_dep =
        has_next_transit ? legs[i + 2U].transit_->dep_time_ : jl.arr_time_;

    // First transit leg with a successor:
    // Search backward from the next leg's departure.
    if (!has_prev_transit && has_next_transit) {
      return n::routing::get_leg_alternatives(
          stop_times_ep.tt_, rt.rtt_.get(), q, n::direction::kBackward,
          next_dep, std::nullopt, original, num_leg_alternatives);
    }

    // Intermediate / single / last transit leg:
    auto const prev_arr = [&]() -> n::unixtime_t {
      if (has_prev_transit) {
        return legs[i - 2U].transit_->arr_time_;
      }
      if (!is_transit(legs.front())) {
        if (auto const access = n::routing::lookup_offset(
                jl.from_, jl.dep_time_, n::routing::side::kBoarding,
                legs.front().offsets_, legs.front().td_offsets_);
            access.has_value()) {
          return access->dep_time_;
        }
        return jl.dep_time_ - (sched_to_unix(legs[i].input_.sched_start_) -
                               sched_to_unix(legs.front().input_.sched_start_));
      }
      return jl.dep_time_;
    }();
    return n::routing::get_leg_alternatives(
        stop_times_ep.tt_, rt.rtt_.get(), q, n::direction::kForward, prev_arr,
        has_next_transit ? std::optional{next_dep} : std::nullopt, original,
        num_leg_alternatives);
  };

  // === Decode & verify ===
  auto legs = utl::to_vec(decode_itinerary_id(id_buf).legs(),
                          [](auto const& x) { return leg{leg_hint{x}}; });
  verify_leg_structure(legs);

  // === Compute first/last mile offsets. ===
  if (!is_transit(legs.front())) {
    auto const& h = legs.front().input_;
    legs.front().offsets_ = get_offsets(h, /*is_start=*/true, {h.mode_});
    legs.front().td_offsets_ = get_td_offsets(
        h, /*is_start=*/true, sched_to_unix(h.sched_end_), {h.mode_});
  }
  if (!is_transit(legs.back())) {
    auto const& h = legs.back().input_;
    legs.back().offsets_ = get_offsets(h, /*is_start=*/false, {h.mode_});
    legs.back().td_offsets_ = get_td_offsets(
        h, /*is_start=*/false, sched_to_unix(h.sched_start_), {h.mode_});
  }

  // ===  Reconstruct transit legs first. ===
  auto transit_err = std::vector<std::string>(legs.size());
  for (auto i = std::size_t{0}; i < legs.size(); ++i) {
    if (!is_transit(legs[i])) {
      continue;
    }
    auto pt = reconstruct_pt_leg(legs[i].input_, stop_times_ep, rt.rtt_.get(),
                                 lang, require_display_name_match);
    if (pt.has_value()) {
      legs[i].from_ = pt->from_;
      legs[i].to_ = pt->to_;
      legs[i].transit_ = std::move(*pt);
    } else {
      transit_err[i] = std::move(pt.error());
    }
  }

  // === Offset/transfer legs inherit from/to from adjacent transit legs. ===
  for (auto i = std::size_t{0}; i < legs.size(); ++i) {
    if (is_transit(legs[i])) {
      continue;
    }
    if (i > 0U) {
      legs[i].from_ = legs[i - 1U].to_;
    }
    if (i + 1U < legs.size()) {
      legs[i].to_ = legs[i + 1U].from_;
    }
  }

  // === Transit legs: Journey to response + leg alternatives. ===
  auto const origin_place = is_transit(legs.front())
                                ? place_t{tt_location{legs.front().from_}}
                                : place_t{legs.front().input_.from_loc_};
  auto const dest_place = is_transit(legs.back())
                              ? place_t{tt_location{legs.back().to_}}
                              : place_t{legs.back().input_.to_loc_};
  for (auto const [i, leg] : utl::enumerate(legs)) {
    if (!is_transit(leg)) {
      continue;
    }

    if (failed(leg)) {
      leg.output_ = {make_dummy_leg(leg.input_, transit_err[i])};
      continue;
    }

    auto const has_prev_transit = i >= 2U;
    auto const has_next_transit = i + 2U < legs.size();
    auto const prev_ok = !has_prev_transit || !failed(legs[i - 2U]);
    auto const next_ok = !has_next_transit || !failed(legs[i + 2U]);
    auto const alternatives =
        (num_leg_alternatives > 0U && prev_ok && next_ok)
            ? get_leg_alternatives(legs, i, has_prev_transit, has_next_transit)
            : alternatives_context{};

    auto const& jl = *leg.transit_;
    leg.output_ = reconstruct(jl, origin_place, dest_place, alternatives);
  }

  // === Anchor non-transit leg times to the adjacent transit legs. ===

  // First -> propagate backward from following transit leg.
  if (!is_transit(legs.front())) {
    auto const duration = legs[0].arr_ - legs[0].dep_;
    legs[0].arr_ = to_unix(legs.at(1).output_.front().startTime_);
    legs[0].dep_ = legs[0].arr_ - duration;
  }

  // Other -> propagate forward from previous transit leg.
  for (auto i = 1U; i < legs.size(); ++i) {
    if (!is_transit(legs[i])) {
      auto const duration = legs[i].arr_ - legs[i].dep_;
      legs[i].dep_ = to_unix(legs[i - 1].output_.back().endTime_);
      legs[i].arr_ = legs[i].dep_ + duration;
    }
  }

  // Resolve non-transit legs.
  for (auto i = std::size_t{0}; i < legs.size(); ++i) {
    if (is_transit(legs[i])) {
      continue;
    }
    legs[i].output_ = (i == 0U || i == legs.size() - 1U)
                          ? reconstruct_offset(legs, i)
                          : reconstruct_transfer(legs[i]);
  }

  // === Assemble itinerary. ===
  auto itinerary = api::Itinerary{};
  for (auto& l : legs) {
    utl::concat(itinerary.legs_, l.output_);
  }
  utl::verify<net::bad_request_exception>(!itinerary.legs_.empty(),
                                          "no legs reconstructed");

  // === Propagate timezone from the nearest transit stop. ===
  // Coordinate boundaries (START / END / mumo transfer points) carry no
  // timezone of their own. Fill them from the closest leg that has one:
  // forward first (so e.g. the last mile inherits the last transit stop's tz),
  // then backward for any remaining leading gaps (so the first mile inherits
  // the first transit stop's tz). Using a single global fallback would wrongly
  // propagate the first mile's timezone to the last mile in cross-timezone
  // journeys.
  {
    auto last = std::optional<std::string>{};
    for (auto& l : itinerary.legs_) {
      l.from_.tz_.has_value() ? (last = l.from_.tz_) : (l.from_.tz_ = last);
      l.to_.tz_.has_value() ? (last = l.to_.tz_) : (l.to_.tz_ = last);
    }
    last.reset();
    for (auto i = itinerary.legs_.size(); i-- > 0U;) {
      auto& l = itinerary.legs_[i];
      l.to_.tz_.has_value() ? (last = l.to_.tz_) : (l.to_.tz_ = last);
      l.from_.tz_.has_value() ? (last = l.from_.tz_) : (l.from_.tz_ = last);
    }
  }

  // === Propagate cancelled + arrival time. ===
  for (auto i = std::size_t{1}; i < itinerary.legs_.size(); ++i) {
    if (itinerary.legs_[i].tripId_.has_value()) {
      continue;
    }

    auto const& prev_to = itinerary.legs_[i - 1].to_;
    auto& curr_from = itinerary.legs_[i].from_;
    if (!curr_from.arrival_.has_value() && prev_to.arrival_.has_value()) {
      curr_from.arrival_ = prev_to.arrival_;
    }
    if (!curr_from.scheduledArrival_.has_value() &&
        prev_to.scheduledArrival_.has_value()) {
      curr_from.scheduledArrival_ = prev_to.scheduledArrival_;
    }
    if (!curr_from.cancelled_.has_value() && prev_to.cancelled_.has_value()) {
      curr_from.cancelled_ = prev_to.cancelled_;
    }
  }

  // === Set meta data. ===
  itinerary.startTime_ = itinerary.legs_.front().startTime_;
  itinerary.endTime_ = itinerary.legs_.back().endTime_;
  itinerary.duration_ = itinerary.endTime_.get_unixtime_seconds() -
                        itinerary.startTime_.get_unixtime_seconds();
  auto const pt_count = utl::count_if(
      itinerary.legs_, [](api::Leg const& l) { return l.tripId_.has_value(); });
  itinerary.transfers_ =
      std::max(std::int64_t{0}, static_cast<std::int64_t>(pt_count) - 1);
  itinerary.id_ = id_buf;

  return itinerary;
}

}  // namespace motis
