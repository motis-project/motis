#include "motis/itinerary_id.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "boost/json.hpp"
#include "boost/url/url_view.hpp"

#include "fmt/chrono.h"
#include "fmt/format.h"

#include "utl/verify.h"

#include "geo/polyline_format.h"

#include "nigiri/rt/frun.h"

#include "net/base64.h"

#include "motis/constants.h"
#include "motis/data.h"
#include "motis/gbfs/routing_data.h"
#include "motis/journey_to_response.h"
#include "motis/parse_location.h"
#include "motis/place.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/time_conv.h"

#include "itinerary_id.pb.h"

namespace n = nigiri;
namespace json = boost::json;

namespace motis {

using proto_id_t = ::motis::proto::SingleLegItineraryId;
using proto_leg_t = ::motis::proto::LegId;

constexpr auto kTimeMul = 1.0 / 60.0 * 7;
constexpr auto kSearchRadiusMeters = 100;
constexpr auto kLookbackSeconds = std::int64_t{8 * 60};
constexpr auto kNonSchedAllowedDeviationSeconds = std::int64_t{15 * 60};

constexpr auto kExactTripIdMatchAddScore = 50.0;
constexpr auto kExactStopIdMatchAddScore = 15.0;
constexpr auto kExactTripNameMatchAddScore = 150.0;

proto_id_t decode_itinerary_id(std::string const& id) {
  auto parsed = proto_id_t{};
  auto const data = net::decode_base64(id);
  utl::verify(parsed.ParseFromString(data), "Failed to decode itinerary-id");
  return parsed;
}

n::lang_t lang_from_str(std::string const& lang) {
  return lang.empty() ? n::lang_t{} : n::lang_t{std::vector<std::string>{lang}};
}

void encode_lang(proto_id_t& id, n::lang_t const& lang) {
  if (lang.has_value() && lang->size() > 0) {
    id.set_lang(lang->front());
  }
}

double exact_stop_id_match_score(api::Place const& place,
                                 std::string_view const expected_stop_id) {
  return place.stopId_.has_value() && *place.stopId_ == expected_stop_id
             ? kExactStopIdMatchAddScore
             : 0.0;
}

struct leg_hint {
  explicit leg_hint(json::object const& l)
      : display_name_{l.at("display_name").as_string()},
        trip_id_{l.at("trip_id").as_string()},
        from_stop_id_{l.at("from_id").as_string()},
        to_stop_id_{l.at("to_id").as_string()},
        sched_start_{l.at("sched_start").as_int64()},
        sched_end_{l.at("sched_end").as_int64()},
        mode_{json::value_to<api::ModeEnum>(l.at("mode"))},
        scheduled_{l.at("scheduled").as_bool()} {
    decode_coords(l.at("coords").as_string());
  }

  explicit leg_hint(proto_leg_t const& l)
      : display_name_{l.display_name()},
        trip_id_{l.trip_id()},
        from_stop_id_{l.from_id()},
        to_stop_id_{l.to_id()},
        sched_start_{l.sched_start()},
        sched_end_{l.sched_end()},
        mode_{json::value_to<api::ModeEnum>(json::value{l.mode()})},
        scheduled_{l.scheduled()} {
    utl::verify(!display_name_.empty(), "itinerary id: display_name missing");
    utl::verify(!trip_id_.empty(), "itinerary id: trip_id missing");
    utl::verify(!from_stop_id_.empty(), "itinerary id: from_id missing");
    utl::verify(!to_stop_id_.empty(), "itinerary id: to_id missing");
    decode_coords(l.coords());
  }

  std::string display_name_;
  std::string trip_id_;
  std::string from_stop_id_;
  std::string to_stop_id_;
  std::int64_t sched_start_;
  std::int64_t sched_end_;
  geo::latlng from_pos_;
  geo::latlng to_pos_;
  api::ModeEnum mode_;
  bool scheduled_;

private:
  void decode_coords(std::string_view const encoded_coords) {
    auto const coords = geo::decode_polyline(encoded_coords);
    utl::verify(coords.size() == 2,
                "itinerary id: coords must decode to exactly 2 points, got {}",
                coords.size());

    from_pos_ = coords[0];
    to_pos_ = coords[1];
  }
};

proto_leg_t get_leg_id_proto(api::Leg const& l) {
  auto id = proto_leg_t{};
  id.set_display_name(l.displayName_.value());
  id.set_trip_id(l.tripId_.value());
  id.set_from_id(l.from_.stopId_.value());
  id.set_to_id(l.to_.stopId_.value());
  id.set_coords(geo::encode_polyline(
      {{l.from_.lat_, l.from_.lon_}, {l.to_.lat_, l.to_.lon_}}));
  id.set_sched_start(l.scheduledStartTime_.get_unixtime_seconds());
  id.set_sched_end(l.scheduledEndTime_.get_unixtime_seconds());
  id.set_mode(std::string{json::value_from(l.mode_).as_string()});
  id.set_scheduled(l.scheduled_);
  return id;
}

std::string get_single_leg_id(api::Leg const& l, n::lang_t const& lang) {
  auto id = proto_id_t{};
  encode_lang(id, lang);
  id.mutable_leg()->CopyFrom(get_leg_id_proto(l));

  auto data = std::string{};
  utl::verify(id.SerializeToString(&data), "failed to serialize itinerary id");
  return net::encode_base64(data);
}

struct st_candidate {
  api::Place place_{};
  std::string tripId_{};
  std::string displayName_{};
  n::rt::run run_{};
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
      st_ep.get_runs(query, rtt, ev_type, query_stop, query_center);

  return utl::to_vec(events, [&](n::rt::run const r) -> st_candidate {
    auto const fr = n::rt::frun{st_ep.tt_, rtt, r};
    auto const s = fr[0];
    auto place =
        to_place(&st_ep.tt_, &st_ep.tags_, st_ep.w_, st_ep.pl_, st_ep.matches_,
                 st_ep.ae_, st_ep.tz_, query.language_, s);
    if (fr.stop_range_.from_ != 0U) {
      place.arrival_ = {s.time(n::event_type::kArr)};
      place.scheduledArrival_ = {s.scheduled_time(n::event_type::kArr)};
    }
    if (fr.stop_range_.from_ != fr.size() - 1U) {
      place.departure_ = {s.time(n::event_type::kDep)};
      place.scheduledDeparture_ = {s.scheduled_time(n::event_type::kDep)};
    }

    auto const trip_id = st_ep.tags_.id(st_ep.tt_, s, ev_type);

    return {.place_ = std::move(place),
            .tripId_ = trip_id,
            .displayName_ = std::string{s.display_name(ev_type, lang)},
            .run_ = r};
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

api::Itinerary build_itinerary_from_frun(
    std::tuple<n::rt::frun, n::stop_idx_t, n::stop_idx_t> const& run,
    ep::stop_times const& stop_times,
    n::shapes_storage const* shapes,
    n::rt_timetable const* rtt,
    n::lang_t const& lang) {
  auto const& [fr, from_idx, to_idx] = run;
  utl::verify(fr.stop_range_.contains(from_idx), "from_idx={} out of range {}",
              from_idx, fr.stop_range_);
  utl::verify(fr.stop_range_.contains(to_idx), "to_idx={} out of range {}",
              to_idx, fr.stop_range_);

  auto const from = n::rt::run_stop{&fr, from_idx};
  auto const to = n::rt::run_stop{&fr, to_idx};

  auto const dep = from.time(n::event_type::kDep);
  auto const arr = to.time(n::event_type::kArr);

  auto j = n::routing::journey{
      .legs_ = {{n::direction::kForward, from.get_location_idx(),
                 to.get_location_idx(), dep, arr,
                 n::routing::journey::run_enter_exit{fr, from_idx, to_idx}}},
      .start_time_ = dep,
      .dest_time_ = arr,
      .dest_ = to.get_location_idx(),
      .transfers_ = 0,
  };

  auto cache = street_routing_cache_t{};
  auto blocked = osr::bitvec<osr::node_idx_t>{};
  auto gbfs_rd = gbfs::gbfs_routing_data{};

  // routing defaults
  constexpr auto join_interlined_legs = true;
  constexpr auto detailed_transfers = true;
  constexpr auto with_fares = false;
  constexpr auto with_scheduled_skipped_stops = false;
  constexpr auto api_version = 0;

  return journey_to_response(
      stop_times.w_, nullptr, stop_times.pl_, stop_times.tt_, stop_times.tags_,
      nullptr, nullptr, rtt, stop_times.matches_, nullptr, shapes, gbfs_rd,
      stop_times.ae_, stop_times.tz_, j,
      tt_location{from.get_location_idx(),
                  n::rt::run_stop{&fr, from_idx}.get_scheduled_location_idx()},
      tt_location{to.get_location_idx()}, cache, &blocked, false,
      osr_parameters{}, api::PedestrianProfileEnum::FOOT,
      api::ElevationCostsEnum::NONE, join_interlined_legs, detailed_transfers,
      with_fares, with_scheduled_skipped_stops,
      stop_times.config_.timetable_.value().max_matching_distance_,
      kMaxMatchingDistance, api_version, false, false, lang);
}

struct candidate_score {
  bool operator<(candidate_score const& o) const {
    return candidate_->tripId_ < o.candidate_->tripId_ ||
           (candidate_->tripId_ == o.candidate_->tripId_ && score_ > o.score_);
  }

  st_candidate const* candidate_;
  double score_;
};

int candidate_score_cmp_ids(candidate_score const& a,
                            candidate_score const& b) {
  int cmp = a.candidate_->tripId_.compare(b.candidate_->tripId_);
  return (cmp > 0) - (cmp < 0);
}

struct from_to_candidate {
  st_candidate const* from_;
  st_candidate const* to_;
};

std::optional<from_to_candidate> get_best_candidate(
    std::vector<st_candidate> const& from_resp,
    std::vector<st_candidate> const& to_resp,
    leg_hint const& hint) {
  auto from_cands = std::vector<candidate_score>{};
  auto to_cands = std::vector<candidate_score>{};
  from_cands.reserve(from_resp.size());
  to_cands.reserve(to_resp.size());

  for (auto const& st : from_resp) {
    if (!st.place_.scheduledDeparture_.has_value()) {
      continue;
    }

    from_cands.emplace_back(
        &st,
        (st.displayName_ == hint.display_name_ ? kExactTripNameMatchAddScore
                                               : 0.0) +
            exact_stop_id_match_score(st.place_, hint.from_stop_id_) -
            geo::distance({st.place_.lat_, st.place_.lon_}, hint.from_pos_) -
            kTimeMul * std::abs(hint.sched_start_ -
                                st.place_.scheduledDeparture_.value()
                                    .get_unixtime_seconds()));
  }
  for (auto const& st : to_resp) {
    if (!st.place_.scheduledArrival_.has_value()) {
      continue;
    }

    to_cands.emplace_back(
        &st, (st.tripId_ == hint.trip_id_ ? kExactTripIdMatchAddScore : 0.0) +
                 exact_stop_id_match_score(st.place_, hint.to_stop_id_) -
                 geo::distance(geo::latlng{st.place_.lat_, st.place_.lon_},
                               hint.to_pos_) -
                 kTimeMul * std::abs(hint.sched_end_ -
                                     st.place_.scheduledArrival_.value()
                                         .get_unixtime_seconds()));
  }

  utl::sort(from_cands);
  utl::sort(to_cands);

  auto best_from_to = std::optional<from_to_candidate>{};
  auto best_score = std::numeric_limits<double>::lowest();

  for (auto i_from = 0U, i_to = 0U;
       i_from < from_cands.size() && i_to < to_cands.size();) {

    switch (candidate_score_cmp_ids(from_cands[i_from], to_cands[i_to])) {
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
               candidate_score_cmp_ids(from_cands[i_from],
                                       from_cands[i_from - 1]) == 0 &&
               candidate_score_cmp_ids(from_cands[i_from], to_cands[i_to]) ==
                   0) {
          ++i_from;
          ++i_to;
        }
        break;
    }
  }

  return best_from_to;
}

api::Itinerary reconstruct_itinerary(ep::stop_times const& stop_times_ep,
                                     nigiri::shapes_storage const* shapes,
                                     rt const& rt,
                                     std::string const& id) {
  auto stop_times_rt = std::atomic_load(&stop_times_ep.rt_);
  auto stop_times_rtt = stop_times_rt->rtt_.get();
  auto const parsed_id = decode_itinerary_id(id);
  utl::verify(parsed_id.has_leg(),
              "reconstruct_itinerary: itinerary id is missing leg");
  auto const lang = lang_from_str(parsed_id.lang());
  auto const lh = leg_hint{parsed_id.leg()};
  auto const get_run =
      [&]() -> std::tuple<n::rt::frun, n::stop_idx_t, n::stop_idx_t> {
    if (!lh.scheduled_) {
      utl::verify(!lh.trip_id_.empty(),
                  "reconstruct_itinerary: additional trip requires trip_id");

      auto const fr = make_frun_from_trip_id(
          stop_times_ep.tags_, stop_times_ep.tt_, rt.rtt_.get(), lh.trip_id_);
      utl::verify(fr.valid(),
                  "reconstruct_itinerary: additional trip not found");
      utl::verify(!fr.is_scheduled(),
                  "reconstruct_itinerary: trip_id resolved to scheduled trip "
                  "while itinerary id expects additional trip");

      auto const from_idx = find_stop_by_id_time(
          fr, stop_times_ep.tags_, lh.from_stop_id_, lh.sched_start_,
          n::event_type::kDep, kNonSchedAllowedDeviationSeconds);
      utl::verify(from_idx.has_value(),
                  "reconstruct_itinerary: additional trip from stop not found");

      auto const to_idx = find_stop_by_id_time(
          fr, stop_times_ep.tags_, lh.to_stop_id_, lh.sched_end_,
          n::event_type::kArr, kNonSchedAllowedDeviationSeconds);
      utl::verify(to_idx.has_value(),
                  "reconstruct_itinerary: additional trip to stop not found");

      utl::verify(*from_idx < *to_idx,
                  "reconstruct_itinerary: invalid stop order (from >= to)");

      return {fr, *from_idx, *to_idx};
    } else {
      auto const from_st_res = get_st_candidates_in_radius(
          stop_times_ep, stop_times_rtt, lh.from_pos_,
          lh.sched_start_ - kLookbackSeconds, lh.mode_, kLookbackSeconds * 2,
          kSearchRadiusMeters, false, lang);
      auto const to_st_res = get_st_candidates_in_radius(
          stop_times_ep, stop_times_rtt, lh.to_pos_,
          lh.sched_end_ - kLookbackSeconds, lh.mode_, kLookbackSeconds * 2,
          kSearchRadiusMeters, true, lang);

      auto const best_from_to = get_best_candidate(from_st_res, to_st_res, lh);

      utl::verify(best_from_to.has_value(), "no matching route is found");

      // Rebuild with the current RT snapshot
      auto best_fr = make_full_frun(stop_times_ep.tt_, rt.rtt_.get(),
                                    best_from_to->from_->run_);
      auto const from_idx = best_from_to->from_->run_.stop_range_.from_;
      auto const to_idx = best_from_to->to_->run_.stop_range_.from_;
      utl::verify(best_fr.stop_range_.contains(from_idx) &&
                      best_fr.stop_range_.contains(to_idx),
                  "reconstruct_itinerary: winning stop index out of range");
      utl::verify(from_idx < to_idx,
                  "reconstruct_itinerary: invalid stop order (from >= to)");

      return {best_fr, from_idx, to_idx};
    }
  };

  auto res = build_itinerary_from_frun(get_run(), stop_times_ep, shapes,
                                       rt.rtt_.get(), lang);
  res.id_ = id;
  return res;
}

}  // namespace motis
