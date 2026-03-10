#include "motis/itinerary_id.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <optional>
#include <sstream>
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

#include "motis/constants.h"
#include "motis/data.h"
#include "motis/gbfs/routing_data.h"
#include "motis/journey_to_response.h"
#include "motis/place.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/time_conv.h"

namespace n = nigiri;
namespace json = boost::json;

namespace motis {

constexpr auto kTimeMul = 1.0 / 60.0 * 7;
constexpr auto kSearchRadiusMeters = 100;
constexpr auto kExactTripIdMatchAddScore = 50.0;
constexpr auto kExactStopIdMatchAddScore = 15.0;

double exact_stop_id_match_score(api::Place const& place,
                                 std::string_view const expected_stop_id) {
  return place.stopId_.has_value() && *place.stopId_ == expected_stop_id
             ? kExactStopIdMatchAddScore
             : 0.0;
}

struct leg_hint {
  explicit leg_hint(json::object const& l)
      : trip_id_{l.at("trip_id").as_string()},
        from_stop_id_{l.at("from_id").as_string()},
        to_stop_id_{l.at("to_id").as_string()},
        sched_start_{l.at("sched_start").as_int64()},
        sched_end_{l.at("sched_end").as_int64()},
        mode_{json::value_to<api::ModeEnum>(l.at("mode"))},
        scheduled_{l.at("scheduled").as_bool()} {
    auto const& encoded_coords = l.at("coords").as_string();
    auto const coords = geo::decode_polyline(encoded_coords);
    utl::verify(coords.size() == 2,
                "itinerary id: coords must decode to exactly 2 points, got {}",
                coords.size());

    from_pos_ = coords[0];
    to_pos_ = coords[1];
  }

  std::string trip_id_;
  std::string from_stop_id_;
  std::string to_stop_id_;
  std::int64_t sched_start_;
  std::int64_t sched_end_;
  geo::latlng from_pos_;
  geo::latlng to_pos_;
  api::ModeEnum mode_;
  bool scheduled_;
};

std::string get_leg_id(api::Leg const& l) {
  return json::serialize(json::object{
      {"trip_id", l.tripId_.value()},
      {"from_id", l.from_.stopId_.value()},
      {"to_id", l.to_.stopId_.value()},
      {"coords", geo::encode_polyline(
                     {{l.from_.lat_, l.from_.lon_}, {l.to_.lat_, l.to_.lon_}})},
      {"sched_start", l.scheduledStartTime_.get_unixtime_seconds()},
      {"sched_end", l.scheduledEndTime_.get_unixtime_seconds()},
      {"mode", json::value_from(l.mode_)},
      {"scheduled", l.scheduled_},
  });
}

api::stoptimes_response get_stop_times_in_radius(ep::stop_times const& st_ep,
                                                 geo::latlng const& center,
                                                 std::int64_t const sched_start,
                                                 api::ModeEnum const mode,
                                                 std::int64_t const window,
                                                 int const radius_m,
                                                 bool const arrive_by) {
  return st_ep(boost::urls::url_view{fmt::format(
      "?center={},{}&time={:%FT%TZ}&arriveBy={}&direction=LATER&window={}"
      "&radius={}&exactRadius=true&fetchStops=true&mode={}",
      center.lat_, center.lng_,
      std::chrono::sys_seconds{std::chrono::seconds{sched_start}},
      arrive_by ? "true" : "false", window, radius_m, fmt::streamed(mode))});
}

n::rt::frun make_frun_from_stoptime(tag_lookup const& tags,
                                    n::timetable const& tt,
                                    n::rt_timetable const* rtt,
                                    std::string_view const trip_id) {
  auto const [run, _] = tags.get_trip(tt, rtt, trip_id);
  return n::rt::frun{tt, rtt, run};
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

std::optional<n::stop_idx_t> find_stop_by_place(n::rt::frun const& fr,
                                                tag_lookup const& tags,
                                                api::Place const& p,
                                                n::event_type const ev_type) {
  if (!p.stopId_) {
    return std::nullopt;
  }
  auto const t = (ev_type == n::event_type::kArr ? p.scheduledArrival_
                                                 : p.scheduledDeparture_);
  if (!t.has_value()) {
    return std::nullopt;
  }

  auto const loc = tags.find_location(*fr.tt_, *p.stopId_);
  if (!loc.has_value()) {
    return std::nullopt;
  }

  return find_stop_by_location_time(fr, *loc, t.value().get_unixtime_seconds(),
                                    ev_type);
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
    n::rt_timetable const* rtt) {
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
      kMaxMatchingDistance, api_version, false, false, std::nullopt);
}

struct id_score {
  bool operator<(id_score const& o) const {
    return id_ < o.id_ || (id_ == o.id_ && score_ > o.score_);
  }

  std::string_view id_;
  double score_;
  api::Place const* place_;
};

struct from_to {
  api::Place const* from_;
  api::Place const* to_;
};

std::tuple<std::optional<std::string_view>, std::optional<from_to>>
get_best_candidate(api::stoptimes_response const& from_stop_times,
                   api::stoptimes_response const& to_stop_times,
                   leg_hint const& hint) {
  auto to_cands = std::vector<id_score>{};
  for (auto const& st : to_stop_times.stopTimes_) {
    if (!st.place_.scheduledArrival_.has_value()) {
      continue;
    }

    to_cands.emplace_back(
        st.tripId_,
        (st.tripId_ == hint.trip_id_ ? kExactTripIdMatchAddScore : 0.0) +
            exact_stop_id_match_score(st.place_, hint.to_stop_id_) -
            geo::distance(geo::latlng{st.place_.lat_, st.place_.lon_},
                          hint.to_pos_) -
            kTimeMul *
                std::abs(
                    hint.sched_end_ -
                    st.place_.scheduledArrival_.value().get_unixtime_seconds()),
        &st.place_);
  }
  utl::sort(to_cands);

  auto best_trip_id = std::optional<std::string_view>{};
  auto best_from_to = std::optional<from_to>{};
  auto best_score = std::numeric_limits<double>::lowest();
  for (auto const& from_st : from_stop_times.stopTimes_) {
    if (!from_st.place_.scheduledDeparture_.has_value()) {
      continue;
    }
    auto it = std::lower_bound(
        begin(to_cands), end(to_cands),
        id_score{from_st.tripId_, std::numeric_limits<double>::max(), nullptr});
    if (it == end(to_cands) || it->id_ != from_st.tripId_) {
      continue;
    }
    auto score = it->score_ +
                 exact_stop_id_match_score(from_st.place_, hint.from_stop_id_) -
                 geo::distance({from_st.place_.lat_, from_st.place_.lon_},
                               hint.from_pos_) -
                 kTimeMul * std::abs(hint.sched_start_ -
                                     from_st.place_.scheduledDeparture_.value()
                                         .get_unixtime_seconds());
    if (score > best_score) {
      best_score = score;
      best_trip_id = from_st.tripId_;
      best_from_to.emplace(&from_st.place_, it->place_);
    }
  }
  return {best_trip_id, best_from_to};
}

api::Itinerary reconstruct_itinerary(ep::stop_times const& stop_times_ep,
                                     nigiri::shapes_storage const* shapes,
                                     rt const& rt,
                                     std::string const& id) {
  constexpr auto kLookbackSeconds = std::int64_t{8 * 60};

  auto const& lh = leg_hint{json::parse(id).as_object()};
  auto const get_run =
      [&]() -> std::tuple<n::rt::frun, n::stop_idx_t, n::stop_idx_t> {
    if (!lh.scheduled_) {
      utl::verify(!lh.trip_id_.empty(),
                  "reconstruct_itinerary: additional trip requires trip_id");

      auto const fr = make_frun_from_stoptime(
          stop_times_ep.tags_, stop_times_ep.tt_, rt.rtt_.get(), lh.trip_id_);
      utl::verify(fr.valid(),
                  "reconstruct_itinerary: additional trip not found");
      utl::verify(!fr.is_scheduled(),
                  "reconstruct_itinerary: trip_id resolved to scheduled trip "
                  "while itinerary id expects additional trip");

      auto const from_idx = find_stop_by_id_time(
          fr, stop_times_ep.tags_, lh.from_stop_id_, lh.sched_start_,
          n::event_type::kDep, kLookbackSeconds);
      utl::verify(from_idx.has_value(),
                  "reconstruct_itinerary: additional trip from stop not found");

      auto const to_idx = find_stop_by_id_time(
          fr, stop_times_ep.tags_, lh.to_stop_id_, lh.sched_end_,
          n::event_type::kArr, kLookbackSeconds);
      utl::verify(to_idx.has_value(),
                  "reconstruct_itinerary: additional trip to stop not found");

      utl::verify(*from_idx < *to_idx,
                  "reconstruct_itinerary: invalid stop order (from >= to)");

      return {fr, *from_idx, *to_idx};
    } else {
      auto const from_st_res = get_stop_times_in_radius(
          stop_times_ep, lh.from_pos_, lh.sched_start_ - kLookbackSeconds,
          lh.mode_, kLookbackSeconds * 2, kSearchRadiusMeters, false);
      auto const to_st_res = get_stop_times_in_radius(
          stop_times_ep, lh.to_pos_, lh.sched_end_ - kLookbackSeconds, lh.mode_,
          kLookbackSeconds * 2, kSearchRadiusMeters, true);

      auto const [best_trip_id, best_from_to] =
          get_best_candidate(from_st_res, to_st_res, lh);

      utl::verify(best_trip_id.has_value() && best_from_to.has_value(),
                  "no matching route is found");
      auto const best_fr = make_frun_from_stoptime(
          stop_times_ep.tags_, stop_times_ep.tt_, rt.rtt_.get(), *best_trip_id);

      auto const from_idx =
          find_stop_by_place(best_fr, stop_times_ep.tags_,
                             *(best_from_to->from_), n::event_type::kDep);
      auto const to_idx =
          find_stop_by_place(best_fr, stop_times_ep.tags_, *(best_from_to->to_),
                             n::event_type::kArr);
      utl::verify(from_idx.has_value() && to_idx.has_value(),
                  "reconstruct_itinerary: could not map from/to stop in frun");
      utl::verify(*from_idx < *to_idx,
                  "reconstruct_itinerary: invalid stop order (from >= to)");

      return {best_fr, *from_idx, *to_idx};
    }
  };

  auto res = build_itinerary_from_frun(get_run(), stop_times_ep, shapes,
                                       rt.rtt_.get());
  res.id_ = id;
  return res;
}

}  // namespace motis
