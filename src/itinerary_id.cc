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
#include "utl/helpers/algorithm.h"
#include "utl/verify.h"
#include "utl/visit.h"

#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/special_stations.h"
#include "nigiri/td_footpath.h"

#include "osr/location.h"

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
#include "motis/odm/prima.h"

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

proto_id_t decode_itinerary_id(std::string const& id) {
  auto parsed = proto_id_t{};
  auto const data = net::decode_base64(id);
  utl::verify(parsed.ParseFromString(data), "Failed to decode itinerary-id");
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
        from_pos_{l.from_lat(), l.from_lon()},
        from_level_{proto_to_level(l, true)},
        to_stop_id_{l.to_id()},
        to_pos_{l.to_lat(), l.to_lon()},
        to_level_{proto_to_level(l, false)},
        sched_start_{l.sched_start()},
        sched_end_{l.sched_end()},
        mode_{json::value_to<api::ModeEnum>(json::value{l.mode()})},
        scheduled_{l.scheduled()} {}

  bool is_public_transport() const { return !trip_id_.empty(); }

  std::string display_name_;
  std::string trip_id_;
  std::string from_stop_id_;
  geo::latlng from_pos_;
  osr::level_t from_level_;
  std::string to_stop_id_;
  geo::latlng to_pos_;
  osr::level_t to_level_;
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
                .lat_ = lh.from_pos_.lat_,
                .lon_ = lh.from_pos_.lng_,
                .level_ = level_to_api(lh.from_level_),
                .departure_ = start,
                .scheduledDeparture_ = start,
                .cancelled_ = true},
      .to_ = {.stopId_ = lh.to_stop_id_,
              .lat_ = lh.to_pos_.lat_,
              .lon_ = lh.to_pos_.lng_,
              .level_ = level_to_api(lh.to_level_),
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
      .alerts_ =
          std::vector<api::Alert>{api::Alert{.headerText_ = std::move(error)}},
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
          auto const mode =  // input mode for routing
              flex::mode_id::is_flex(id)
                  ? api::ModeEnum::FLEX
                  : (id >= kGbfsTransportModeIdOffset
                         ? api::ModeEnum::RENTAL
                         : to_mode(static_cast<osr::search_profile>(id)));
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

bool is_transit_leg(n::routing::journey::leg const& leg) {
  return std::holds_alternative<n::routing::journey::run_enter_exit>(leg.uses_);
}

place_t endpoint_place(n::routing::journey::leg const& leg,
                       bool const from,
                       std::optional<osr::location> const& pos) {
  auto const loc = from ? leg.from_ : leg.to_;
  auto const special =
      from ? n::special_station::kStart : n::special_station::kEnd;
  if (loc == n::get_special_station(special) && pos.has_value()) {
    return *pos;
  }
  return tt_location{loc};
}

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
        -geo::distance(st.run_[0].pos(), hint.from_pos_) -
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
            geo::distance(st.run_[0].pos(), hint.to_pos_) -
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
      .post_transit_ = {
          .form_factors_ = q.postTransitRentalFormFactors_,
          .propulsion_types_ = q.postTransitRentalPropulsionTypes_,
          .providers_ = q.postTransitRentalProviders_,
          .provider_groups_ = q.postTransitRentalProviderGroups_,
          .ignore_return_constraints_ =
              q.ignorePostTransitRentalReturnConstraints_}};
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
        stop_times_ep, lh.from_pos_, lh.sched_start_ - kLookbackSeconds,
        lh.mode_, kLookbackSeconds * 2, kSearchRadiusMeters, false, lang);
    auto const to_st_res = get_st_candidates_in_radius(
        stop_times_ep, lh.to_pos_, lh.sched_end_ - kLookbackSeconds, lh.mode_,
        kLookbackSeconds * 2, kSearchRadiusMeters, true, lang);

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

api::Itinerary reconstruct_itinerary(
    ep::routing const& routing,
    ep::stop_times const& stop_times_ep,
    rt const& rt,
    std::string const& id,
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
  auto const parsed_id = decode_itinerary_id(id);

  auto const n_legs = parsed_id.legs_size();
  auto start_pos = std::optional<osr::location>{};
  auto end_pos = std::optional<osr::location>{};
  auto first_mile_mode = std::optional<api::ModeEnum>{};
  auto last_mile_mode = std::optional<api::ModeEnum>{};

  auto slots = std::vector<std::variant<n::routing::journey::leg, api::Leg>>{};
  slots.reserve(static_cast<std::size_t>(n_legs));

  for (auto l_idx = 0; l_idx < n_legs; ++l_idx) {
    auto const lh = leg_hint{parsed_id.legs(l_idx)};

    // Case 1: Public transport leg
    if (lh.is_public_transport()) {
      auto run = reconstruct_pt_leg(lh, stop_times_ep, rt.rtt_.get(), lang,
                                    require_display_name_match);
      if (run.has_value()) {
        slots.emplace_back(std::move(*run));
      } else {
        slots.emplace_back(make_dummy_leg(lh, std::move(run.error())));
      }
      continue;
    }

    // Case 2: first/last mile OR transfer.
    auto from_loc = n::location_idx_t::invalid();
    auto to_loc = n::location_idx_t::invalid();

    if (lh.from_stop_id_.empty()) {
      utl::verify(l_idx == 0,
                  "reconstruct_itinerary: non-PT leg without from_id is only "
                  "valid as the first leg (first-mile)");
      from_loc = n::get_special_station(n::special_station::kStart);
      start_pos = osr::location{lh.from_pos_, lh.from_level_};
      first_mile_mode = lh.mode_;
    } else {
      auto const f = stop_times_ep.tags_.find_location(stop_times_ep.tt_,
                                                       lh.from_stop_id_);
      utl::verify(f.has_value(),
                  "reconstruct_itinerary: non-PT leg from stop not found");
      from_loc = *f;
    }

    if (lh.to_stop_id_.empty()) {
      utl::verify(l_idx == n_legs - 1,
                  "reconstruct_itinerary: non-PT leg without to_id is only "
                  "valid as the last leg (last-mile)");
      to_loc = n::get_special_station(n::special_station::kEnd);
      end_pos = osr::location{lh.to_pos_, lh.to_level_};
      last_mile_mode = lh.mode_;
    } else {
      auto const t =
          stop_times_ep.tags_.find_location(stop_times_ep.tt_, lh.to_stop_id_);
      utl::verify(t.has_value(),
                  "reconstruct_itinerary: non-PT leg to stop not found");
      to_loc = *t;
    }

    auto const dep =
        n::unixtime_t{std::chrono::duration_cast<n::unixtime_t::duration>(
            std::chrono::seconds{lh.sched_start_})};
    auto const arr =
        n::unixtime_t{std::chrono::duration_cast<n::unixtime_t::duration>(
            std::chrono::seconds{lh.sched_end_})};
    auto const dur = std::max(
        n::duration_t{0}, std::chrono::duration_cast<n::duration_t>(arr - dep));

    auto const is_offset =
        from_loc == n::get_special_station(n::special_station::kStart) ||
        to_loc == n::get_special_station(n::special_station::kEnd);

    if (is_offset) {
      // Case 2.1: first / last mile.
      auto const fallback_mode =
          (lh.mode_ == api::ModeEnum::RENTAL || lh.mode_ == api::ModeEnum::FLEX)
              ? kWalkTransportModeId  // if not routable -> fallback to walking
              : static_cast<n::transport_mode_id_t>(to_profile(
                    lh.mode_, flm.pedestrian_profile_, flm.elevation_costs_));
      slots.emplace_back(n::routing::journey::leg{
          n::direction::kForward, from_loc, to_loc, dep, arr,
          n::routing::offset{to_loc, dur, fallback_mode}});
    } else {
      // Case 2.2: transfer between public transport legs.
      slots.emplace_back(n::routing::journey::leg{n::direction::kForward,
                                                  from_loc, to_loc, dep, arr,
                                                  n::footpath{to_loc, dur}});
    }
  }

  // === Slot-inspection helpers (transit or dummy alike) ===
  using slot_t = std::variant<n::routing::journey::leg, api::Leg>;
  auto const slot_is_pt = [](slot_t const& s) {
    return std::holds_alternative<api::Leg>(s) ||
           is_transit_leg(std::get<n::routing::journey::leg>(s));
  };
  auto const dt_to_unixtime = [](openapi::date_time_t const t) {
    return n::unixtime_t{std::chrono::duration_cast<n::unixtime_t::duration>(
        std::chrono::seconds{t.get_unixtime_seconds()})};
  };
  auto const slot_dep_time = [&](slot_t const& s) {
    return std::holds_alternative<n::routing::journey::leg>(s)
               ? std::get<n::routing::journey::leg>(s).dep_time_
               : dt_to_unixtime(std::get<api::Leg>(s).scheduledStartTime_);
  };
  auto const slot_arr_time = [&](slot_t const& s) {
    return std::holds_alternative<n::routing::journey::leg>(s)
               ? std::get<n::routing::journey::leg>(s).arr_time_
               : dt_to_unixtime(std::get<api::Leg>(s).scheduledEndTime_);
  };
  auto const find_stop = [&](std::string const& stop_id) {
    auto const loc =
        stop_times_ep.tags_.find_location(stop_times_ep.tt_, stop_id);
    utl::verify(loc.has_value(), "reconstruct_itinerary: stop not found");
    return *loc;
  };
  auto const slot_from_loc = [&](slot_t const& s) {
    return std::holds_alternative<n::routing::journey::leg>(s)
               ? std::get<n::routing::journey::leg>(s).from_
               : find_stop(*std::get<api::Leg>(s).from_.stopId_);
  };
  auto const slot_to_loc = [&](slot_t const& s) {
    return std::holds_alternative<n::routing::journey::leg>(s)
               ? std::get<n::routing::journey::leg>(s).to_
               : find_stop(*std::get<api::Leg>(s).to_.stopId_);
  };

  // === Compute first/last mile offsets ONCE, apply re-routing in place ===
  auto const rtt = rt.rtt_.get();
  auto start_alt_offsets = std::vector<n::routing::offset>{};
  auto dest_alt_offsets = std::vector<n::routing::offset>{};
  auto start_td_alt = n::routing::td_offsets_t{};
  auto dest_td_alt = n::routing::td_offsets_t{};
  auto gbfs_rd = gbfs::gbfs_routing_data{routing.w_, routing.l_, routing.gbfs_};

  if (routing.is_osr_loaded()) {
    auto stats = ep::stats_map_t{};
    auto const reroute = [&](osr::location const& pos, api::ModeEnum const mode,
                             bool const is_start, rental_options const& ro,
                             std::chrono::seconds const max,
                             std::vector<n::routing::offset>& flat_out,
                             n::routing::td_offsets_t& td_out) {
      auto pt_slot_idx = std::optional<std::size_t>{};
      if (is_start) {
        for (auto i = std::size_t{0}; i < slots.size(); ++i) {
          if (slot_is_pt(slots[i])) {
            pt_slot_idx = i;
            break;
          }
        }
      } else {
        for (auto i = slots.size(); i-- > 0;) {
          if (slot_is_pt(slots[i])) {
            pt_slot_idx = i;
            break;
          }
        }
      }
      if (!pt_slot_idx.has_value()) {
        return;
      }
      auto const pt_stop = is_start ? slot_from_loc(slots[*pt_slot_idx])
                                    : slot_to_loc(slots[*pt_slot_idx]);
      auto const pt_time = is_start ? slot_dep_time(slots[*pt_slot_idx])
                                    : slot_arr_time(slots[*pt_slot_idx]);

      auto& slot = is_start ? slots.front() : slots.back();
      if (!std::holds_alternative<n::routing::journey::leg>(slot)) {
        return;
      }
      auto& leg = std::get<n::routing::journey::leg>(slot);
      auto const dir =
          is_start ? osr::direction::kForward : osr::direction::kBackward;
      auto const apply = [&](n::duration_t const dur,
                             n::transport_mode_id_t const tmid) {
        leg.uses_ = n::routing::offset{pt_stop, dur, tmid};
        if (is_start) {
          leg.arr_time_ = pt_time;
          leg.dep_time_ = pt_time - dur;
        } else {
          leg.dep_time_ = pt_time;
          leg.arr_time_ = pt_time + dur;
        }
      };

      if (mode == api::ModeEnum::FLEX) {
        auto td = routing.get_td_offsets(
            rtt, nullptr, place_t{pos}, dir, {mode}, flm.osr_params_,
            flm.pedestrian_profile_, flm.elevation_costs_,
            flm.max_matching_distance_, max, n::routing::start_time_t{pt_time},
            stats);
        if (auto const it = td.find(pt_stop); it != td.end()) {
          auto const r = n::get_td_duration(
              is_start ? n::direction::kForward : n::direction::kBackward,
              it->second, pt_time);
          if (r.has_value()) {
            apply(r->first, r->second.transport_mode_id_);
          }
        }
        td_out = std::move(td);
        return;
      }

      auto offsets = routing.get_offsets(
          rtt, place_t{pos}, dir, {mode}, ro, flm.osr_params_,
          flm.pedestrian_profile_, flm.elevation_costs_, max,
          flm.max_matching_distance_, gbfs_rd, stats);
      auto const it = utl::find_if(
          offsets, [&](auto const& o) { return o.target() == pt_stop; });
      if (it != offsets.end()) {
        apply(it->duration(), it->type());
      }
      flat_out = std::move(offsets);
    };

    if (first_mile_mode.has_value() && start_pos.has_value()) {
      reroute(*start_pos, *first_mile_mode, /*is_start=*/true, flm.pre_transit_,
              flm.max_pre_transit_, start_alt_offsets, start_td_alt);
    }
    if (last_mile_mode.has_value() && end_pos.has_value()) {
      reroute(*end_pos, *last_mile_mode, /*is_start=*/false, flm.post_transit_,
              flm.max_post_transit_, dest_alt_offsets, dest_td_alt);
    }
  }

  // === Anchor non-transit-leg times to surrounding (transit/dummy) PT ===
  auto const anchor_leg_time =
      [](n::routing::journey::leg& leg, n::duration_t const dur,
         n::unixtime_t const adj_time, bool const is_start) {
        if (is_start) {
          leg.arr_time_ = adj_time;
          leg.dep_time_ = adj_time - dur;
        } else {
          leg.dep_time_ = adj_time;
          leg.arr_time_ = adj_time + dur;
        }
      };
  if (slots.size() > 1 &&
      std::holds_alternative<n::routing::journey::leg>(slots.front())) {
    auto& first = std::get<n::routing::journey::leg>(slots.front());
    if (!is_transit_leg(first)) {
      anchor_leg_time(first, first.arr_time_ - first.dep_time_,
                      slot_dep_time(slots[1]), /*is_start=*/true);
    }
  }
  for (auto i = std::size_t{1}; i < slots.size(); ++i) {
    if (!std::holds_alternative<n::routing::journey::leg>(slots[i])) continue;
    auto& leg = std::get<n::routing::journey::leg>(slots[i]);
    if (is_transit_leg(leg)) continue;
    auto const dur = leg.arr_time_ - leg.dep_time_;
    if (dur == n::duration_t{0} &&
        std::holds_alternative<n::footpath>(leg.uses_) &&
        i + 1 < slots.size()) {
      auto const next_dep = slot_dep_time(slots[i + 1]);
      leg.dep_time_ = next_dep;
      leg.arr_time_ = next_dep;
    } else {
      anchor_leg_time(leg, dur, slot_arr_time(slots[i - 1]),
                      /*is_start=*/false);
    }
  }

  // === Snapshot slot context BEFORE per-leg processing ===
  // The per-leg loop moves api::Leg dummies out of `slots`, after which the
  // moved-from variant can't be queried for stop ids. Capture each slot's
  // PT-ness and from/to location now and consult the snapshot for surrounding
  // PT context inside the loop.
  struct slot_ctx {
    bool is_pt;
    n::location_idx_t from_loc;
    n::location_idx_t to_loc;
  };
  auto slot_ctxs = std::vector<slot_ctx>{};
  slot_ctxs.reserve(slots.size());
  for (auto const& s : slots) {
    if (std::holds_alternative<n::routing::journey::leg>(s)) {
      auto const& l = std::get<n::routing::journey::leg>(s);
      slot_ctxs.push_back({is_transit_leg(l), l.from_, l.to_});
    } else {
      auto const& d = std::get<api::Leg>(s);
      slot_ctxs.push_back(
          {true,
           d.from_.stopId_.has_value() ? find_stop(*d.from_.stopId_)
                                       : n::location_idx_t::invalid(),
           d.to_.stopId_.has_value() ? find_stop(*d.to_.stopId_)
                                     : n::location_idx_t::invalid()});
    }
  }

  // === Render each slot via a single-leg journey_to_response call ===
  auto out_legs = std::vector<api::Leg>{};
  auto cache = street_routing_cache_t{};
  auto blocked = osr::bitvec<osr::node_idx_t>{};
  auto const non_dummy_match_mode =
      routing.is_osr_loaded() ? n::routing::location_match_mode::kIntermodal
                              : n::routing::location_match_mode::kEquivalent;
  auto const safe_prf_idx =
      stop_times_ep.tt_.locations_.footpaths_out_.at(prf_idx).empty()
          ? n::profile_idx_t{0U}
          : prf_idx;

  for (auto i = std::size_t{0}; i < slots.size(); ++i) {
    auto& slot = slots[i];

    if (std::holds_alternative<api::Leg>(slot)) {
      out_legs.emplace_back(std::move(std::get<api::Leg>(slot)));
      continue;
    }

    auto& jleg = std::get<n::routing::journey::leg>(slot);
    auto const is_transit = is_transit_leg(jleg);

    auto j = n::routing::journey{};
    j.legs_.push_back(std::move(jleg));
    j.start_time_ = j.legs_.front().dep_time_;
    j.dest_time_ = j.legs_.back().arr_time_;
    j.dest_ = j.legs_.back().to_;
    j.transfers_ = 0U;

    // Per-leg leg-alternatives query: only for transit legs. The surrounding
    // slot context (real transit OR dummy PT) drives `has_prev`/`has_next` --
    // recreating what nigiri's `make_alternative_query` does in plan.
    auto leg_alt_query = std::optional<n::routing::query>{};
    if (is_transit && num_leg_alternatives > 0U) {
      auto prev_pt = std::optional<std::size_t>{};
      for (auto k = i; k-- > 0;) {
        if (slot_ctxs[k].is_pt) {
          prev_pt = k;
          break;
        }
      }
      auto next_pt = std::optional<std::size_t>{};
      for (auto k = i + 1; k < slots.size(); ++k) {
        if (slot_ctxs[k].is_pt) {
          next_pt = k;
          break;
        }
      }
      auto const has_prev = prev_pt.has_value();
      auto const has_next = next_pt.has_value();

      auto const start_match_mode =
          has_prev ? n::routing::location_match_mode::kExact
                   : non_dummy_match_mode;
      auto const dest_match_mode = has_next
                                       ? n::routing::location_match_mode::kExact
                                       : non_dummy_match_mode;

      auto start_offsets =
          has_prev
              ? std::vector<n::routing::offset>{{slot_ctxs[*prev_pt].to_loc,
                                                 n::duration_t{0U}, 0U}}
          : start_alt_offsets.empty()
              ? std::vector<n::routing::offset>{{j.legs_.front().from_,
                                                 n::duration_t{0U}, 0U}}
              : start_alt_offsets;
      auto dest_offsets =
          has_next
              ? std::vector<n::routing::offset>{{slot_ctxs[*next_pt].from_loc,
                                                 n::duration_t{0U}, 0U}}
          : dest_alt_offsets.empty()
              ? std::vector<n::routing::offset>{{j.legs_.back().to_,
                                                 n::duration_t{0U}, 0U}}
              : dest_alt_offsets;

      auto td_start = has_prev ? n::routing::td_offsets_t{} : start_td_alt;
      auto td_dest = has_next ? n::routing::td_offsets_t{} : dest_td_alt;

      leg_alt_query = n::routing::query{
          .start_time_ = j.start_time_,
          .start_match_mode_ = start_match_mode,
          .dest_match_mode_ = dest_match_mode,
          .use_start_footpaths_ = has_prev || !routing.is_osr_loaded(),
          .start_ = std::move(start_offsets),
          .destination_ = std::move(dest_offsets),
          .td_start_ = std::move(td_start),
          .td_dest_ = std::move(td_dest),
          .prf_idx_ = safe_prf_idx,
          .allowed_claszes_ = allowed_claszes,
          .require_bike_transport_ = require_bike_transport,
          .require_car_transport_ = require_car_transport};
    }

    auto const start_place = endpoint_place(j.legs_.front(), true,
                                            i == 0 ? start_pos : std::nullopt);
    auto const end_place = endpoint_place(
        j.legs_.back(), false, i == slots.size() - 1 ? end_pos : std::nullopt);

    auto leg_itin = journey_to_response(
        stop_times_ep.w_, routing.l_, stop_times_ep.pl_, stop_times_ep.tt_,
        stop_times_ep.tags_, routing.fa_, nullptr, rtt, stop_times_ep.matches_,
        nullptr, routing.shapes_, gbfs_rd, stop_times_ep.ae_, stop_times_ep.tz_,
        j, start_place, end_place, cache, &blocked, prf_idx == n::kCarProfile,
        flm.osr_params_, flm.pedestrian_profile_, flm.elevation_costs_,
        join_interlined_legs, detailed_transfers, detailed_legs,
        /*with_fares=*/false, with_scheduled_skipped_stops,
        stop_times_ep.config_.timetable_.value().max_matching_distance_,
        flm.max_matching_distance_, 6U, false, false, lang,
        /*set_itinerary_id_field=*/false,
        leg_alt_query.has_value() ? &*leg_alt_query : nullptr,
        num_leg_alternatives);

    utl::concat(out_legs, leg_itin.legs_);
  }

  utl::verify(!out_legs.empty(),
              "reconstruct_itinerary: no legs reconstructed");

  // Per-leg journey_to_response sees only one leg at a time, so cross-leg
  // context is lost:
  //   * the timezone fallback that `get_first_run_tz` derives from the first
  //     transit leg's stop is unavailable for single-leg offset/footpath
  //     journeys, and
  //   * the from-place arrival/cancellation, normally inherited from the
  //     previous leg's to-place, is not set.
  // Restore both across the assembled out_legs.
  auto fallback_tz = std::optional<std::string>{};
  for (auto const& l : out_legs) {
    if (l.from_.tz_.has_value()) {
      fallback_tz = l.from_.tz_;
      break;
    }
    if (l.to_.tz_.has_value()) {
      fallback_tz = l.to_.tz_;
      break;
    }
  }
  if (fallback_tz.has_value()) {
    for (auto& l : out_legs) {
      if (!l.from_.tz_.has_value()) l.from_.tz_ = fallback_tz;
      if (!l.to_.tz_.has_value()) l.to_.tz_ = fallback_tz;
    }
  }
  for (auto i = std::size_t{1}; i < out_legs.size(); ++i) {
    // Only non-transit legs (walks/transfers/access/egress) inherit the
    // arrival context from the predecessor — boarding stops on transit legs
    // don't carry an `arrival` semantic.
    if (out_legs[i].tripId_.has_value()) continue;
    auto const& prev_to = out_legs[i - 1].to_;
    auto& cur_from = out_legs[i].from_;
    if (!cur_from.arrival_.has_value() && prev_to.arrival_.has_value()) {
      cur_from.arrival_ = prev_to.arrival_;
    }
    if (!cur_from.scheduledArrival_.has_value() &&
        prev_to.scheduledArrival_.has_value()) {
      cur_from.scheduledArrival_ = prev_to.scheduledArrival_;
    }
    if (!cur_from.cancelled_.has_value() && prev_to.cancelled_.has_value()) {
      cur_from.cancelled_ = prev_to.cancelled_;
    }
  }

  auto res = api::Itinerary{};
  res.startTime_ = out_legs.front().startTime_;
  res.endTime_ = out_legs.back().endTime_;
  res.duration_ = res.endTime_.get_unixtime_seconds() -
                  res.startTime_.get_unixtime_seconds();
  auto const pt_count = utl::count_if(
      out_legs, [](api::Leg const& l) { return l.tripId_.has_value(); });
  res.transfers_ =
      std::max(std::int64_t{0}, static_cast<std::int64_t>(pt_count) - 1);
  res.id_ = id;
  res.legs_ = std::move(out_legs);
  return res;
}

n::unixtime_t to_unix(openapi::date_time_t const& x) {
  return std::chrono::time_point_cast<n::i32_minutes>(x.time_);
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif

api::Itinerary reconstruct_itinerary_1(
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
    first_last_mile_options const& flm,
    unsigned const api_version) {
  // ==== Helpers ====
  struct leg {
    leg_hint input_;
    // Only for non-transit legs (offset/transfer):
    // updated departure/arrival times to not overlap with times
    // of resolved transit legs
    n::unixtime_t dep_{}, arr_{};
    std::vector<api::Leg> output_{};
  };

  auto const is_transit = [](leg const& l) {
    return !l.input_.trip_id_.empty();
  };
  auto const get_offsets = [&](leg_hint const& l, osr::direction const dir) {
    CISTA_UNUSED_PARAM(l)
    CISTA_UNUSED_PARAM(dir)
    return std::vector<n::routing::offset>{};
  };
  auto const get_td_offsets = [&](leg_hint const& l, osr::direction const dir) {
    CISTA_UNUSED_PARAM(l)
    CISTA_UNUSED_PARAM(dir)
    return n::routing::td_offsets_t{};
  };
  auto gbfs_rd = gbfs::gbfs_routing_data{routing.w_, routing.l_, routing.gbfs_};
  auto cache = street_routing_cache_t{};
  auto blocked = osr::bitvec<osr::node_idx_t>{};
  auto const reconstruct =
      [&](n::routing::journey::leg const& l) -> std::vector<api::Leg> {
    auto j = n::routing::journey{.legs_ = {l},
                                 .start_time_ = l.dep_time_,
                                 .dest_time_ = l.arr_time_,
                                 .dest_ = l.to_,
                                 .transfers_ = 0};
    return journey_to_response(
               stop_times_ep.w_, routing.l_, stop_times_ep.pl_,
               stop_times_ep.tt_, stop_times_ep.tags_, routing.fa_, rt.e_.get(),
               rt.rtt_.get(), stop_times_ep.matches_, routing.elevations_,
               routing.shapes_, gbfs_rd, stop_times_ep.ae_, stop_times_ep.tz_,
               j, tt_location{l.from_}, tt_location{l.to_}, cache, &blocked,
               prf_idx == n::kCarProfile, flm.osr_params_,
               flm.pedestrian_profile_, flm.elevation_costs_,
               join_interlined_legs, detailed_transfers, detailed_legs,
               /*with_fares=*/false, with_scheduled_skipped_stops,
               stop_times_ep.config_.timetable_.value().max_matching_distance_,
               flm.max_matching_distance_, api_version, false, false, lang,
               /*set_itinerary_id_field=*/false, nullptr, 0U)
        .legs_;
  };
  auto const reconstruct_transit = [&](leg const& l) -> std::vector<api::Leg> {
    auto x =
        reconstruct_pt_leg(l.input_, stop_times_ep, rt.rtt_.get(), lang,
                           require_display_name_match)
            .transform_error([&](std::string error) -> std::vector<api::Leg> {
              return {make_dummy_leg(l.input_, std::move(error))};
            })
            .transform(reconstruct);
    return x.value_or(x.error());
  };
  auto const reconstruct_offset = [](leg const& l) -> std::vector<api::Leg> {
    CISTA_UNUSED_PARAM(l)
    return {api::Leg{}};
  };
  auto const reconstruct_transfer = [](leg const& l) -> std::vector<api::Leg> {
    CISTA_UNUSED_PARAM(l)
    return {api::Leg{}};
  };

  // Structure:
  // - [first mile offset]
  // - PT [, TRANSFER, PT]...
  // - [last mile offset]
  // => !id.legs().empty()
  auto legs = utl::to_vec(decode_itinerary_id(id_buf).legs(),
                          [](auto const& x) { return leg{leg_hint{x}}; });

  // Reconstruct transit legs first.
  // Transit real-times are needed for non-transit leg lookup.
  for (auto& l : legs) {
    if (is_transit(l)) {
      l.output_ = reconstruct_transit(l);
      assert(!l.output_.empty());
    }
  }

  // Update non-transit leg times.
  // First -> propagate backward from following transit leg.
  if (!is_transit(legs.front())) {
    auto const duration = legs[0].arr_ - legs[0].dep_;
    legs[0].arr_ = to_unix(legs.at(1).output_.front().startTime_);
    legs[0].dep_ = to_unix(legs[0].arr_ - duration);
  }

  // Other -> propagate forward from previous transit leg.
  for (auto i = 1U; i < legs.size(); ++i) {
    if (!is_transit(legs[i])) {
      auto const duration = legs[0].arr_ - legs[0].dep_;
      legs[i].dep_ = to_unix(legs[i - 1].output_.back().endTime_);
      legs[i].arr_ = to_unix(legs[i].dep_ + duration);
    }
  }

  // Compute offsets.
  auto const& first = legs.front().input_;
  auto const& last = legs.back().input_;
  auto q = n::routing::query{};
  if (!is_transit(legs.front())) {
    q.start_ = get_offsets(first, osr::direction::kBackward);
    q.td_start_ = get_td_offsets(first, osr::direction::kBackward);
    q.start_match_mode_ = n::routing::location_match_mode::kIntermodal;
  }
  if (!is_transit(legs.back())) {
    q.destination_ = get_offsets(last, osr::direction::kForward);
    q.td_dest_ = get_td_offsets(last, osr::direction::kForward);
    q.dest_match_mode_ = n::routing::location_match_mode::kIntermodal;
  }

  // Resolve non-transit legs.
  for (auto [i, l] : utl::enumerate(legs)) {
    if (is_transit(l)) {
      continue;
    }

    if (i == 0 || i == legs.size() - 1) {
      // First/last mile offset.
      l.output_ = reconstruct_offset(l);
    } else {
      // Transfers between two public transport legs.
      l.output_ = reconstruct_transfer(l);
    }
  }

  // Assemble itinerary.
  auto itinerary = api::Itinerary{};
  for (auto& l : legs) {
    itinerary.legs_.insert(itinerary.legs_.end(), l.output_.begin(),
                           l.output_.end());
  }
  // Set itinerary meta data derived from legs.
  // TODO

  return itinerary;
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

}  // namespace motis
