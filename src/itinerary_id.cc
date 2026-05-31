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

#include "nigiri/routing/direct.h"
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

n::unixtime_t to_unix(openapi::date_time_t const& x) {
  return std::chrono::time_point_cast<n::i32_minutes>(x.time_);
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
  auto const find_loc = [&](std::string const& stop_id) -> n::location_idx_t {
    return stop_times_ep.tags_.find_location(stop_times_ep.tt_, stop_id)
        .value_or(n::location_idx_t::invalid());
  };
  // Some profiles have no footpaths -> fall back to the default profile.
  auto const safe_prf_idx =
      stop_times_ep.tt_.locations_.footpaths_out_.at(prf_idx).empty()
          ? n::profile_idx_t{0U}
          : prf_idx;

  auto gbfs_rd = gbfs::gbfs_routing_data{routing.w_, routing.l_, routing.gbfs_};
  auto cache = street_routing_cache_t{};
  // `street_routing` marks blocked elevator nodes into this bitvec via
  // `set_blocked`, which does not resize -> it must cover all OSR nodes.
  auto blocked = osr::bitvec<osr::node_idx_t>{};
  if (stop_times_ep.w_ != nullptr) {
    blocked.resize(stop_times_ep.w_->n_nodes());
  }
  auto stats = ep::stats_map_t{};

  // Street-routed first/last mile offsets. `dir` is the Dijkstra direction:
  // first mile is many-to-one (backward), last mile one-to-many (forward).
  auto const get_offsets = [&](leg_hint const& h, bool const is_start,
                               osr::direction const dir) {
    if (!routing.is_osr_loaded()) {
      return std::vector<n::routing::offset>{};
    }
    auto const pos = is_start ? osr::location{h.from_pos_, h.from_level_}
                              : osr::location{h.to_pos_, h.to_level_};
    return routing.get_offsets(
        rt.rtt_.get(), place_t{pos}, dir, {h.mode_},
        is_start ? flm.pre_transit_ : flm.post_transit_, flm.osr_params_,
        flm.pedestrian_profile_, flm.elevation_costs_,
        is_start ? flm.max_pre_transit_ : flm.max_post_transit_,
        flm.max_matching_distance_, gbfs_rd, stats);
  };
  auto const get_td_offsets = [&](leg_hint const& h, bool const is_start,
                                  osr::direction const dir,
                                  n::unixtime_t const anchor) {
    if (!routing.is_osr_loaded()) {
      return n::routing::td_offsets_t{};
    }
    auto const pos = is_start ? osr::location{h.from_pos_, h.from_level_}
                              : osr::location{h.to_pos_, h.to_level_};
    return routing.get_td_offsets(
        rt.rtt_.get(), rt.e_.get(), place_t{pos}, dir, {h.mode_},
        flm.osr_params_, flm.pedestrian_profile_, flm.elevation_costs_,
        flm.max_matching_distance_,
        is_start ? flm.max_pre_transit_ : flm.max_post_transit_,
        n::routing::start_time_t{anchor}, stats);
  };

  auto const non_dummy_match_mode =
      routing.is_osr_loaded() ? n::routing::location_match_mode::kIntermodal
                              : n::routing::location_match_mode::kEquivalent;

  // First/last mile alternative offsets for the leg-alternatives query of the
  // first/last transit leg. Populated below (before transit reconstruction)
  // when the journey actually starts/ends with an offset.
  auto start_alt_offsets = std::vector<n::routing::offset>{};
  auto start_td_alt = n::routing::td_offsets_t{};
  auto dest_alt_offsets = std::vector<n::routing::offset>{};
  auto dest_td_alt = n::routing::td_offsets_t{};

  auto const reconstruct =
      [&](n::routing::journey::leg const& l, place_t const& from_place,
          place_t const& to_place,
          n::routing::query const* const leg_alt_query = nullptr,
          std::size_t const num_alts = 0U) -> std::vector<api::Leg> {
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
               j, from_place, to_place, cache, &blocked,
               prf_idx == n::kCarProfile, flm.osr_params_,
               flm.pedestrian_profile_, flm.elevation_costs_,
               join_interlined_legs, detailed_transfers, detailed_legs,
               /*with_fares=*/false, with_scheduled_skipped_stops,
               stop_times_ep.config_.timetable_.value().max_matching_distance_,
               flm.max_matching_distance_, 6U, false, false, lang,
               /*set_itinerary_id_field=*/false, leg_alt_query, num_alts)
        .legs_;
  };

  // Recreates nigiri's `make_alternative_query` for a single transit leg:
  // the surrounding PT context (real transit OR dummy PT) drives
  // has_prev/has_next, which selects exact stop matching vs. the first/last
  // mile offsets.
  auto const make_leg_alt_query =
      [&](std::vector<leg> const& legs, std::size_t const i,
          n::routing::journey::leg const& jl) {
    auto prev_pt = std::optional<std::size_t>{};
    for (auto k = i; k-- > 0;) {
      if (is_transit(legs[k])) {
        prev_pt = k;
        break;
      }
    }
    auto next_pt = std::optional<std::size_t>{};
    for (auto k = i + 1; k < legs.size(); ++k) {
      if (is_transit(legs[k])) {
        next_pt = k;
        break;
      }
    }
    auto const has_prev = prev_pt.has_value();
    auto const has_next = next_pt.has_value();

    auto start_offsets =
        has_prev ? std::vector<n::routing::offset>{{find_loc(
                       legs[*prev_pt].input_.to_stop_id_),
                       n::duration_t{0U}, 0U}}
        : start_alt_offsets.empty()
            ? std::vector<n::routing::offset>{{jl.from_, n::duration_t{0U}, 0U}}
            : start_alt_offsets;
    auto dest_offsets =
        has_next ? std::vector<n::routing::offset>{{find_loc(
                       legs[*next_pt].input_.from_stop_id_),
                       n::duration_t{0U}, 0U}}
        : dest_alt_offsets.empty()
            ? std::vector<n::routing::offset>{{jl.to_, n::duration_t{0U}, 0U}}
            : dest_alt_offsets;

    return n::routing::query{
        .start_time_ = jl.dep_time_,
        .start_match_mode_ = has_prev
                                 ? n::routing::location_match_mode::kExact
                                 : non_dummy_match_mode,
        .dest_match_mode_ = has_next
                                ? n::routing::location_match_mode::kExact
                                : non_dummy_match_mode,
        .use_start_footpaths_ = has_prev || !routing.is_osr_loaded(),
        .start_ = std::move(start_offsets),
        .destination_ = std::move(dest_offsets),
        .td_start_ = has_prev ? n::routing::td_offsets_t{} : start_td_alt,
        .td_dest_ = has_next ? n::routing::td_offsets_t{} : dest_td_alt,
        .prf_idx_ = safe_prf_idx,
        .allowed_claszes_ = allowed_claszes,
        .require_bike_transport_ = require_bike_transport,
        .require_car_transport_ = require_car_transport};
  };
  auto const reconstruct_transit =
      [&](std::vector<leg> const& legs,
          std::size_t const i) -> std::vector<api::Leg> {
    auto pt = reconstruct_pt_leg(legs[i].input_, stop_times_ep, rt.rtt_.get(),
                                 lang, require_display_name_match);
    if (!pt.has_value()) {
      return {make_dummy_leg(legs[i].input_, std::move(pt.error()))};
    }
    auto const& jl = *pt;
    auto leg_alt_query = std::optional<n::routing::query>{};
    if (num_leg_alternatives > 0U) {
      leg_alt_query = make_leg_alt_query(legs, i, jl);
    }
    return reconstruct(jl, tt_location{jl.from_}, tt_location{jl.to_},
                       leg_alt_query.has_value() ? &*leg_alt_query : nullptr,
                       num_leg_alternatives);
  };

  // First/last mile: street-route to/from the adjacent PT stop and pick the
  // offset that targets it. Timing is anchored at the adjacent PT
  // departure/arrival; `journey_to_response` keeps it.
  auto const reconstruct_offset =
      [&](std::vector<leg> const& legs,
          std::size_t const i) -> std::vector<api::Leg> {
    auto const& l = legs[i];
    auto const& h = l.input_;
    auto const is_first = i == 0U;
    auto const loc = find_loc(is_first ? h.to_stop_id_ : h.from_stop_id_);
    auto const t = is_first ? l.arr_ : l.dep_;
    auto const s =
        is_first ? n::routing::side::kBoarding : n::routing::side::kAlighting;
    auto const dir =
        is_first ? osr::direction::kBackward : osr::direction::kForward;

    auto const offsets = get_offsets(h, is_first, dir);
    auto const td = get_td_offsets(h, is_first, dir, t);
    auto const found = n::routing::lookup_offset(loc, t, s, offsets, td);
    if (!found.has_value()) {
      return {make_dummy_leg(h, "reconstruct_itinerary: no offset found")};
    }

    auto const start_place =
        is_first ? place_t{osr::location{h.from_pos_, h.from_level_}}
                 : place_t{tt_location{loc}};
    auto const end_place =
        is_first ? place_t{tt_location{loc}}
                 : place_t{osr::location{h.to_pos_, h.to_level_}};
    return reconstruct(*found, start_place, end_place);
  };

  // Transfer between two PT legs: derive timing from the footpath graph
  // (real-time td-footpath override first, else static footpath) instead of
  // RAPTOR. Anchored at the previous PT arrival.
  auto const reconstruct_transfer =
      [&](std::vector<leg> const& legs,
          std::size_t const i) -> std::vector<api::Leg> {
    auto const& l = legs[i];
    auto const& h = l.input_;
    auto const from_stop = find_loc(h.from_stop_id_);
    auto const to_stop = find_loc(h.to_stop_id_);

    auto q = n::routing::query{};
    q.prf_idx_ = safe_prf_idx;
    auto const offs = std::vector<n::routing::offset>{
        {to_stop, n::duration_t{0}, kWalkTransportModeId}};

    // Forward search from the previous PT arrival: leaving `from_stop` when
    // the traveller actually alights. This is the moment a blocked elevator
    // matters; a backward search from the next departure would instead find
    // any feasible slot across the whole transfer buffer. `kAlighting`
    // yields from=from_stop, to=to_stop, dep=arrival, arr=arrival+duration.
    auto const found = n::routing::lookup_footpath(
        from_stop, l.dep_, n::routing::side::kAlighting, stop_times_ep.tt_,
        rt.rtt_.get(), q, offs, n::routing::location_match_mode::kExact,
        /*use_footpaths=*/true);
    if (!found.has_value()) {
      return {make_dummy_leg(h, "reconstruct_itinerary: no transfer footpath")};
    }
    return reconstruct(*found, tt_location{from_stop}, tt_location{to_stop});
  };

  // Structure:
  // - [first mile offset]
  // - PT [, TRANSFER, PT]...
  // - [last mile offset]
  // => !id.legs().empty()
  auto legs = utl::to_vec(decode_itinerary_id(id_buf).legs(),
                          [](auto const& x) { return leg{leg_hint{x}}; });

  // Precompute first/last mile alternative offsets (consumed by the
  // leg-alternatives query of the first/last transit leg). Anchored at the
  // encoded boarding/alighting time of the adjacent PT leg.
  auto const sched_to_unix = [](std::int64_t const s) {
    return n::unixtime_t{
        std::chrono::duration_cast<n::unixtime_t::duration>(
            std::chrono::seconds{s})};
  };
  if (num_leg_alternatives > 0U && !is_transit(legs.front())) {
    auto const& h = legs.front().input_;
    start_alt_offsets = get_offsets(h, /*is_start=*/true, osr::direction::kBackward);
    start_td_alt = get_td_offsets(h, /*is_start=*/true, osr::direction::kBackward,
                                  sched_to_unix(h.sched_end_));
  }
  if (num_leg_alternatives > 0U && !is_transit(legs.back())) {
    auto const& h = legs.back().input_;
    dest_alt_offsets = get_offsets(h, /*is_start=*/false, osr::direction::kForward);
    dest_td_alt = get_td_offsets(h, /*is_start=*/false, osr::direction::kForward,
                                 sched_to_unix(h.sched_start_));
  }

  // Reconstruct transit legs first.
  // Transit real-times are needed for non-transit leg lookup.
  for (auto i = std::size_t{0}; i < legs.size(); ++i) {
    if (is_transit(legs[i])) {
      legs[i].output_ = reconstruct_transit(legs, i);
      assert(!legs[i].output_.empty());
    }
  }

  // Anchor non-transit leg times to the adjacent transit legs. The actual
  // durations are recomputed from the offset/footpath lookup below; here we
  // only fix the endpoint that touches the neighbouring PT leg.
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
                          : reconstruct_transfer(legs, i);
  }

  // Assemble itinerary.
  auto itinerary = api::Itinerary{};
  for (auto& l : legs) {
    itinerary.legs_.insert(itinerary.legs_.end(), l.output_.begin(),
                           l.output_.end());
  }
  utl::verify(!itinerary.legs_.empty(),
              "reconstruct_itinerary: no legs reconstructed");

  // Per-leg journey_to_response sees only one leg at a time, so cross-leg
  // context is lost:
  //   * the timezone fallback that `get_first_run_tz` derives from the first
  //     transit leg's stop is unavailable for single-leg offset/footpath
  //     journeys, and
  //   * the from-place arrival/cancellation, normally inherited from the
  //     previous leg's to-place, is not set.
  // Restore both across the assembled legs.
  auto fallback_tz = std::optional<std::string>{};
  for (auto const& l : itinerary.legs_) {
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
    for (auto& l : itinerary.legs_) {
      if (!l.from_.tz_.has_value()) l.from_.tz_ = fallback_tz;
      if (!l.to_.tz_.has_value()) l.to_.tz_ = fallback_tz;
    }
  }
  for (auto i = std::size_t{1}; i < itinerary.legs_.size(); ++i) {
    // Only non-transit legs (walks/transfers/access/egress) inherit the
    // arrival context from the predecessor — boarding stops on transit legs
    // don't carry an `arrival` semantic.
    if (itinerary.legs_[i].tripId_.has_value()) continue;
    auto const& prev_to = itinerary.legs_[i - 1].to_;
    auto& cur_from = itinerary.legs_[i].from_;
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

  // Set itinerary meta data derived from legs.
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
