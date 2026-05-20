#include "motis/itinerary_id.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "boost/json.hpp"
#include "boost/url/url_view.hpp"

#include "fmt/chrono.h"
#include "fmt/format.h"

#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/rt/frun.h"
#include "nigiri/special_stations.h"

#include "osr/location.h"

#include "net/base64.h"

#include "motis/constants.h"
#include "motis/data.h"
#include "motis/gbfs/routing_data.h"
#include "motis/journey_to_response.h"
#include "motis/osr/mode_to_profile.h"
#include "motis/parse_location.h"
#include "motis/place.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/time_conv.h"

#include "itinerary_id.pb.h"

namespace n = nigiri;
namespace json = boost::json;

namespace motis {

using proto_id_t = ItineraryId;
using proto_leg_t = LegId;

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
        scheduled_{l.scheduled()} {
    if (!trip_id_.empty()) {
      utl::verify(!display_name_.empty(), "itinerary id: display_name missing");
      utl::verify(!from_stop_id_.empty(), "itinerary id: from_id missing");
      utl::verify(!to_stop_id_.empty(), "itinerary id: to_id missing");
    }
  }

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

proto_leg_t get_leg_id_proto(api::Leg const& l,
                             std::string const& leg_display_name) {
  if (l.tripId_.has_value()) {
    utl::verify(l.from_.stopId_.has_value(),
                "itinerary id: PT leg missing 'from' stopId");
    utl::verify(l.to_.stopId_.has_value(),
                "itinerary id: PT leg missing 'to' stopId");
  }
  auto id = proto_leg_t{};
  id.set_display_name(leg_display_name);
  id.set_trip_id(l.tripId_.value_or(""));
  id.set_from_id(l.from_.stopId_.value_or(""));
  id.set_to_id(l.to_.stopId_.value_or(""));
  id.set_from_lat(l.from_.lat_);
  id.set_from_lon(l.from_.lon_);
  id.set_to_lat(l.to_.lat_);
  id.set_to_lon(l.to_.lon_);
  if (l.from_.level_.has_value()) {
    id.set_from_level(*l.from_.level_);
  }
  if (l.to_.level_.has_value()) {
    id.set_to_level(*l.to_.level_);
  }
  id.set_sched_start(l.scheduledStartTime_.get_unixtime_seconds());
  id.set_sched_end(l.scheduledEndTime_.get_unixtime_seconds());
  id.set_mode(std::string{json::value_from(l.mode_).as_string()});
  id.set_scheduled(l.scheduled_);
  return id;
}

std::string get_single_leg_id(api::Leg const& l,
                              std::string const& leg_display_name) {
  auto id = proto_id_t{};
  id.mutable_legs()->Add(get_leg_id_proto(l, leg_display_name));

  auto data = std::string{};
  utl::verify(id.SerializeToString(&data), "failed to serialize itinerary id");
  return net::encode_base64(data);
}

std::string generate_itinerary_id(
    api::Itinerary const& itin,
    std::vector<std::string> const& default_display_names,
    std::vector<std::size_t> const& default_display_names_indices) {
  utl::verify(itin.legs_.size() != 0,
              "generate_itinerary_id expects at least 1 leg");
  auto id = proto_id_t{};
  auto default_name_idx = std::size_t{0};
  for (auto l_idx = std::size_t{0}; l_idx < itin.legs_.size(); ++l_idx) {
    auto const& leg = itin.legs_[l_idx];
    auto display_name =
        (default_name_idx != default_display_names_indices.size() &&
         l_idx == default_display_names_indices.at(default_name_idx))
            ? default_display_names.at(default_name_idx++)
            : leg.displayName_.value_or("");
    id.mutable_legs()->Add(get_leg_id_proto(leg, display_name));
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
    nigiri::rt_timetable const* rtt,
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
      st_ep.get_runs(query, rtt, ev_type, query_stop, query_center, true);

  return utl::to_vec(events, [&](n::rt::run const r) -> st_candidate {
    auto const fr = n::rt::frun{st_ep.tt_, rtt, r};
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

api::Itinerary build_itinerary_from_legs(
    std::vector<n::routing::journey::leg>& legs,
    std::optional<osr::location> const& start_pos,
    std::optional<osr::location> const& end_pos,
    ep::stop_times const& stop_times,
    osr::lookup const* l,
    n::shapes_storage const* shapes,
    n::rt_timetable const* rtt,
    bool const join_interlined_legs,
    bool const detailed_transfers,
    bool const detailed_legs,
    bool const with_fares,
    bool const with_scheduled_skipped_stops,
    n::lang_t const& lang,
    std::size_t const num_leg_alternatives) {
  utl::verify(!legs.empty(), "build_itinerary_from_legs: no legs");

  for (auto i = std::size_t{1}; i < legs.size(); ++i) {
    auto& leg = legs[i];
    if (is_transit_leg(leg)) {
      continue;
    }
    auto const dur = leg.arr_time_ - leg.dep_time_;
    if (dur == n::duration_t{0} &&
        std::holds_alternative<n::footpath>(leg.uses_) && i + 1 < legs.size()) {
      leg.dep_time_ = legs[i + 1].dep_time_;
      leg.arr_time_ = legs[i + 1].dep_time_;
    } else {
      leg.dep_time_ = legs[i - 1].arr_time_;
      leg.arr_time_ = leg.dep_time_ + dur;
    }
  }

  auto const pt_count =
      static_cast<std::size_t>(utl::count_if(legs, is_transit_leg));

  auto j = n::routing::journey{};
  j.start_time_ = legs.front().dep_time_;
  j.dest_time_ = legs.back().arr_time_;
  j.dest_ = legs.back().to_;
  j.transfers_ =
      pt_count == 0 ? std::uint8_t{0} : static_cast<std::uint8_t>(pt_count - 1);

  auto const start_place = endpoint_place(legs.front(), true, start_pos);
  auto const end_place = endpoint_place(legs.back(), false, end_pos);

  j.legs_ = std::move(legs);

  auto cache = street_routing_cache_t{};
  auto blocked = osr::bitvec<osr::node_idx_t>{};
  auto gbfs_rd = gbfs::gbfs_routing_data{};

  auto const is_transit = [](n::routing::journey::leg const& l) {
    return std::holds_alternative<n::routing::journey::run_enter_exit>(l.uses_);
  };
  auto const first_transit =
      std::find_if(begin(j.legs_), end(j.legs_), is_transit);
  auto const last_transit =
      std::find_if(rbegin(j.legs_), rend(j.legs_), is_transit);
  auto leg_alternatives_query = std::optional<n::routing::query>{};
  if (num_leg_alternatives > 0U && first_transit != end(j.legs_) &&
      last_transit != rend(j.legs_)) {
    leg_alternatives_query = n::routing::query{
        .start_time_ = j.start_time_,
        .start_match_mode_ = n::routing::location_match_mode::kExact,
        .dest_match_mode_ = n::routing::location_match_mode::kExact,
        .use_start_footpaths_ = true,
        .start_ = {{first_transit->from_, n::duration_t{0U}, 0U}},
        .destination_ = {{last_transit->to_, n::duration_t{0U}, 0U}}};
  }

  return journey_to_response(
      stop_times.w_, l, stop_times.pl_, stop_times.tt_, stop_times.tags_,
      nullptr, nullptr, rtt, stop_times.matches_, nullptr, shapes, gbfs_rd,
      stop_times.ae_, stop_times.tz_, j, start_place, end_place, cache,
      &blocked, false, osr_parameters{}, api::PedestrianProfileEnum::FOOT,
      api::ElevationCostsEnum::NONE, join_interlined_legs, detailed_transfers,
      detailed_legs, with_fares, with_scheduled_skipped_stops,
      stop_times.config_.timetable_.value().max_matching_distance_,
      kMaxMatchingDistance, 6U, false, false, lang, false,
      leg_alternatives_query ? &*leg_alternatives_query : nullptr,
      num_leg_alternatives);
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
    tag_lookup const& tags,
    n::rt_timetable const* rtt) {
  if (from_resp.empty() || to_resp.empty()) {
    return std::nullopt;
  }

  auto from_cands = std::vector<candidate_score>{};
  auto to_cands = std::vector<candidate_score>{};
  from_cands.reserve(from_resp.size());
  to_cands.reserve(to_resp.size());

  auto const [hint_trip_run, _] =
      tags.get_trip(*from_resp.front().run_.tt_, rtt, hint.trip_id_);

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

api::Itinerary reconstruct_itinerary(ep::stop_times const& stop_times_ep,
                                     osr::lookup const* l,
                                     nigiri::shapes_storage const* shapes,
                                     rt const& rt,
                                     std::string const& id,
                                     bool const require_display_name_match,
                                     bool const join_interlined_legs,
                                     bool const detailed_transfers,
                                     bool const detailed_legs,
                                     bool const with_fares,
                                     bool const with_scheduled_skipped_stops,
                                     n::lang_t const& lang,
                                     std::size_t const num_leg_alternatives) {
  auto stop_times_rt = std::atomic_load(&stop_times_ep.rt_);
  auto stop_times_rtt = stop_times_rt->rtt_.get();
  auto const parsed_id = decode_itinerary_id(id);

  auto const make_pt_leg = [](n::rt::frun const& fr, n::stop_idx_t const from,
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
  };

  auto const get_run = [&](leg_hint const& lh)
      -> std::expected<n::routing::journey::leg, std::string> {
    auto const fail = [](std::string_view const msg) {
      return std::unexpected{std::string{msg}};
    };

    if (!lh.scheduled_) {
      if (lh.trip_id_.empty()) {
        return fail("reconstruct_itinerary: additional trip requires trip_id");
      }

      auto const fr = make_frun_from_trip_id(
          stop_times_ep.tags_, stop_times_ep.tt_, rt.rtt_.get(), lh.trip_id_);
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
        return fail(
            "reconstruct_itinerary: additional trip from stop not found");
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
          stop_times_ep, stop_times_rtt, lh.from_pos_,
          lh.sched_start_ - kLookbackSeconds, lh.mode_, kLookbackSeconds * 2,
          kSearchRadiusMeters, false, lang);
      auto const to_st_res = get_st_candidates_in_radius(
          stop_times_ep, stop_times_rtt, lh.to_pos_,
          lh.sched_end_ - kLookbackSeconds, lh.mode_, kLookbackSeconds * 2,
          kSearchRadiusMeters, true, lang);

      auto const best_from_to = get_best_candidate(
          from_st_res, to_st_res, lh, require_display_name_match,
          stop_times_ep.tags_, stop_times_rtt);

      if (!best_from_to.has_value()) {
        return fail("no matching route is found");
      }

      // Rebuild with the current RT snapshot
      auto const best_fr = make_full_frun(stop_times_ep.tt_, rt.rtt_.get(),
                                          best_from_to->from_->run_);
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
  };

  auto const n_legs = parsed_id.legs_size();
  auto start_pos = std::optional<osr::location>{};
  auto end_pos = std::optional<osr::location>{};

  auto slots = std::vector<std::variant<n::routing::journey::leg, api::Leg>>{};
  slots.reserve(static_cast<std::size_t>(n_legs));
  auto any_dummy = false;

  for (auto l_idx = 0; l_idx < n_legs; ++l_idx) {
    auto const lh = leg_hint{parsed_id.legs(l_idx)};
    if (lh.is_public_transport()) {
      auto run = get_run(lh);
      if (run.has_value()) {
        slots.emplace_back(std::move(*run));
      } else {
        slots.emplace_back(make_dummy_leg(lh, std::move(run.error())));
        any_dummy = true;
      }
      continue;
    }

    auto from_loc = n::location_idx_t::invalid();
    auto to_loc = n::location_idx_t::invalid();

    if (lh.from_stop_id_.empty()) {
      utl::verify(l_idx == 0,
                  "reconstruct_itinerary: non-PT leg without from_id is only "
                  "valid as the first leg (first-mile)");
      from_loc = n::get_special_station(n::special_station::kStart);
      start_pos = osr::location{lh.from_pos_, lh.from_level_};
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
      slots.emplace_back(n::routing::journey::leg{
          n::direction::kForward, from_loc, to_loc, dep, arr,
          n::routing::offset{to_loc, dur, kWalkTransportModeId}});
    } else {
      slots.emplace_back(n::routing::journey::leg{n::direction::kForward,
                                                  from_loc, to_loc, dep, arr,
                                                  n::footpath{to_loc, dur}});
    }
  }

  // Fully reconstructable journey: keep the original single-pass behaviour.
  if (!any_dummy) {
    auto legs = std::vector<n::routing::journey::leg>{};
    legs.reserve(slots.size());
    for (auto& slot : slots) {
      legs.emplace_back(std::move(std::get<n::routing::journey::leg>(slot)));
    }
    auto res = build_itinerary_from_legs(
        legs, start_pos, end_pos, stop_times_ep, l, shapes, rt.rtt_.get(),
        join_interlined_legs, detailed_transfers, detailed_legs, with_fares,
        with_scheduled_skipped_stops, lang, num_leg_alternatives);
    res.id_ = id;
    return res;
  }

  // Partially reconstructable journey:
  // reconstruct consecutive segments and stitch them together.
  auto out_legs = std::vector<api::Leg>{};
  auto pt_leg_count = std::int64_t{0};
  auto segment = std::vector<n::routing::journey::leg>{};
  auto first_segment = true;

  auto const flush_segment = [&](bool const is_last) {
    if (segment.empty()) {
      return;
    }
    pt_leg_count += utl::count_if(segment, is_transit_leg);
    auto seg_itinerary = build_itinerary_from_legs(
        segment, first_segment ? start_pos : std::nullopt,
        is_last ? end_pos : std::nullopt, stop_times_ep, l, shapes,
        rt.rtt_.get(), join_interlined_legs, detailed_transfers, detailed_legs,
        with_fares, with_scheduled_skipped_stops, lang, num_leg_alternatives);
    for (auto& leg : seg_itinerary.legs_) {
      out_legs.emplace_back(std::move(leg));
    }
    segment.clear();
    first_segment = false;
  };

  for (auto& slot : slots) {
    if (std::holds_alternative<n::routing::journey::leg>(slot)) {
      segment.emplace_back(std::move(std::get<n::routing::journey::leg>(slot)));
    } else {
      flush_segment(false);
      out_legs.emplace_back(std::move(std::get<api::Leg>(slot)));
      ++pt_leg_count;
    }
  }
  flush_segment(true);

  utl::verify(!out_legs.empty(),
              "reconstruct_itinerary: no legs reconstructed");

  auto res = api::Itinerary{};
  res.startTime_ = out_legs.front().startTime_;
  res.endTime_ = out_legs.back().endTime_;
  res.duration_ = res.endTime_.get_unixtime_seconds() -
                  res.startTime_.get_unixtime_seconds();
  res.transfers_ = std::max(std::int64_t{0}, pt_leg_count - 1);
  res.id_ = id;
  res.legs_ = std::move(out_legs);
  return res;
}

}  // namespace motis
