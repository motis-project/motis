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

#include "nigiri/rt/frun.h"
#include "geo/polyline_format.h"

#include "motis/constants.h"
#include "motis/gbfs/routing_data.h"
#include "motis/journey_to_response.h"
#include "motis/place.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/time_conv.h"

namespace n = nigiri;

namespace {
constexpr auto kItineraryTimeBase = std::chrono::sys_days{
    std::chrono::year{2026} / std::chrono::month{1} / std::chrono::day{1}};
constexpr auto kTimeMul = 1.0 / 60.0 * 7;
constexpr auto kExactTripIdMatchAddScore = 50.0;
constexpr auto kExactStopIdMatchAddScore = 15.0;

double exact_stop_id_match_score(motis::api::Place const& place,
                                 std::string_view expected_stop_id) {
  return place.stopId_.has_value() && *place.stopId_ == expected_stop_id
             ? kExactStopIdMatchAddScore
             : 0.0;
}

std::int64_t to_epoch_seconds(openapi::date_time_t const& t) {
  auto const sys = static_cast<std::chrono::sys_seconds>(t);
  return std::chrono::duration_cast<std::chrono::seconds>(
             sys.time_since_epoch())
      .count();
}

std::int64_t to_itinerary_minutes(std::int64_t const epoch_seconds) {
  return std::chrono::duration_cast<std::chrono::minutes>(
             std::chrono::seconds{epoch_seconds} -
             kItineraryTimeBase.time_since_epoch())
      .count();
}

std::int64_t to_epoch_seconds_from_itinerary_minutes(
    std::int64_t const minutes) {
  return std::chrono::duration_cast<std::chrono::seconds>(
             (kItineraryTimeBase + std::chrono::minutes{minutes})
                 .time_since_epoch())
      .count();
}

std::string_view require_stop_id(motis::api::Place const& p,
                                 std::size_t const leg_idx,
                                 std::string_view field) {
  utl::verify(p.stopId_.has_value() && !p.stopId_->empty(),
              "itinerary id: leg {} missing {}.stopId", leg_idx, field);
  return *p.stopId_;
}

struct leg_hint {
  std::string trip_id_;
  std::string from_stop_id_;
  std::string to_stop_id_;
  std::int64_t sched_start_;
  std::int64_t sched_end_;
  geo::latlng from_latlng_;
  geo::latlng to_latlng_;
  motis::api::ModeEnum mode_;

  explicit leg_hint(boost::json::object const& l) {
    auto const& encoded_coords = l.at("coords").as_string();
    auto const coords = geo::decode_polyline(encoded_coords);
    utl::verify(coords.size() == 2,
                "itinerary id: coords must decode to exactly 2 points, got {}",
                coords.size());

    trip_id_ = l.at("trip_id").as_string().c_str();
    from_stop_id_ = l.at("from_id").as_string().c_str();
    to_stop_id_ = l.at("to_id").as_string().c_str();
    // time
    auto const sched_delta = l.at("sched_delta");
    auto const sched_start_m = l.at("sched_start").as_int64();
    auto const sched_delta_m = sched_delta.as_int64();
    utl::verify(sched_delta_m >= 0, "itinerary id: sched_delta < 0");
    sched_start_ = to_epoch_seconds_from_itinerary_minutes(sched_start_m);
    sched_end_ =
        to_epoch_seconds_from_itinerary_minutes(sched_start_m + sched_delta_m);

    from_latlng_ = coords[0];
    to_latlng_ = coords[1];
    mode_ = static_cast<motis::api::ModeEnum>(l.at("mode").as_int64());
  }
};

motis::api::stoptimes_response stoptimes_in_radius(
    motis::ep::stop_times const& st_ep,
    geo::latlng const& center,
    std::int64_t sched_start,
    motis::api::ModeEnum mode,
    int window,
    int radius_m,
    bool arrive_by) {
  std::ostringstream oss;
  oss << mode;

  auto const t =
      std::chrono::system_clock::time_point{std::chrono::seconds{sched_start}};
  auto const iso = fmt::format("{:%FT%TZ}", t);

  auto const query = fmt::format(
      "?center={},{}&time={}&arriveBy={}&direction=LATER&window={}"
      "&radius={}&exactRadius=true&fetchStops=true&mode={}",
      center.lat_, center.lng_, iso, arrive_by ? "true" : "false", window,
      radius_m, oss.str());

  return st_ep(boost::urls::url_view{query});
}

n::rt::frun make_frun_from_stoptime(motis::tag_lookup const& tags,
                                    n::timetable const& tt,
                                    n::rt_timetable const* rtt,
                                    std::string_view const trip_id) {

  auto const [run, _] = tags.get_trip(tt, rtt, trip_id);
  return n::rt::frun{tt, rtt, run};
}

std::optional<n::stop_idx_t> find_stop_by_location_time(
    n::rt::frun const& fr,
    n::location_idx_t const loc,
    openapi::date_time_t const& scheduled_time,
    n::event_type const ev_type) {
  auto const target_sec = to_epoch_seconds(scheduled_time);

  for (auto i = n::stop_idx_t{0U}; i < fr.size(); ++i) {
    auto const rs = fr[i];
    if (rs.get_location_idx() == loc &&
        motis::to_seconds(rs.scheduled_time(ev_type)) == target_sec) {
      return rs.stop_idx_;
    }
  }
  return std::nullopt;
}

std::optional<n::stop_idx_t> find_stop_by_place(n::rt::frun const& fr,
                                                motis::tag_lookup const& tags,
                                                motis::api::Place const& p,
                                                n::event_type const ev_type) {

  if (!p.stopId_) {
    return std::nullopt;
  }
  auto t = (ev_type == n::event_type::kArr ? p.scheduledArrival_
                                           : p.scheduledDeparture_);
  if (!t.has_value()) {
    return std::nullopt;
  }

  auto const loc = tags.find_location(*fr.tt_, *p.stopId_);
  if (!loc.has_value()) {
    return std::nullopt;
  }

  return find_stop_by_location_time(fr, *loc, *t, ev_type);
}

motis::api::Itinerary build_itinerary_from_frun(
    std::vector<std::tuple<n::rt::frun, n::stop_idx_t, n::stop_idx_t>> runs,
    motis::ep::stop_times const& stoptimes_ep,
    n::rt_timetable const* rtt,
    n::unixtime_t journey_start) {

  std::vector<n::routing::journey::leg> legs;
  auto const first_from_rs =
      n::rt::run_stop{&std::get<0>(runs.front()), std::get<1>(runs.front())};

  for (auto const& run : runs) {
    auto const& [fr, from_idx, to_idx] = run;
    utl::verify(
        fr.stop_range_.contains(from_idx) && fr.stop_range_.contains(to_idx),
        "build_itinerary_from_frun: stop idx out of range");

    auto const from_rs = n::rt::run_stop{&fr, from_idx};
    auto const to_rs = n::rt::run_stop{&fr, to_idx};

    auto const dep = from_rs.time(n::event_type::kDep);
    auto const arr = to_rs.time(n::event_type::kArr);

    legs.push_back({n::direction::kForward, from_rs.get_location_idx(),
                    to_rs.get_location_idx(), dep, arr,
                    n::routing::journey::run_enter_exit{fr, from_idx, to_idx}});
  }

  auto j = n::routing::journey{.legs_ = legs,
                               .start_time_ = journey_start,
                               .dest_time_ = legs.back().arr_time_,
                               .dest_ = legs.back().to_,
                               .transfers_ = 0};

  auto cache = motis::street_routing_cache_t{};
  auto blocked = osr::bitvec<osr::node_idx_t>{};
  auto gbfs_rd = motis::gbfs::gbfs_routing_data{};

  // routing defaults
  constexpr auto join_interlined_legs = true;
  constexpr auto detailed_transfers = true;
  constexpr auto with_fares = false;
  constexpr auto with_scheduled_skipped_stops = false;
  constexpr auto api_version = 0;

  return journey_to_response(
      stoptimes_ep.w_,
      nullptr,  // osr::lookup **
      stoptimes_ep.pl_, stoptimes_ep.tt_, stoptimes_ep.tags_, nullptr, nullptr,
      rtt, stoptimes_ep.matches_, nullptr,
      nullptr,  // shapes **
      gbfs_rd, stoptimes_ep.ae_, stoptimes_ep.tz_, j,
      motis::tt_location{legs.front().from_,
                         first_from_rs.get_scheduled_location_idx()},
      motis::tt_location{legs.back().to_}, cache, &blocked, false,
      motis::osr_parameters{}, motis::api::PedestrianProfileEnum::FOOT,
      motis::api::ElevationCostsEnum::NONE, join_interlined_legs,
      detailed_transfers, with_fares, with_scheduled_skipped_stops,
      stoptimes_ep.config_.timetable_.value().max_matching_distance_,
      motis::kMaxMatchingDistance, api_version, false, false, std::nullopt);
}

struct id_score {
  std::string_view id_;
  double score_;
  motis::api::Place const* place_;

  bool operator<(id_score const& o) const {
    return id_ < o.id_ || (id_ == o.id_ && score_ > o.score_);
  }
};
struct from_to {
  motis::api::Place const* from_;
  motis::api::Place const* to_;
};
}  // namespace

namespace motis {

std::string generate_itinerary_id(api::Itinerary const& itin) {
  utl::verify(itin.legs_.size() == 1, "itin.legs_.size() must be one");

  auto legs = boost::json::array{};
  legs.reserve(itin.legs_.size());

  int64_t max_sched_t = 0;

  {
    auto const& leg = itin.legs_[0];

    auto const trip_id = leg.tripId_.value_or("");
    auto const from_id = require_stop_id(leg.from_, 0, "from");
    auto const to_id = require_stop_id(leg.to_, 0, "to");

    auto const sched_start = to_epoch_seconds(leg.scheduledStartTime_);
    auto const sched_end = to_epoch_seconds(leg.scheduledEndTime_);
    utl::verify(sched_start != 0 && sched_end != 0,
                "itinerary id: leg {} missing scheduled times", 0);
    utl::verify(sched_end >= sched_start,
                "itinerary id: leg {} scheduledEndTime < scheduledStartTime",
                0);
    auto const sched_start_m = to_itinerary_minutes(sched_start);
    auto const sched_end_m = to_itinerary_minutes(sched_end);
    max_sched_t = sched_end;

    auto const sched_delta_m = sched_end_m - sched_start_m;

    auto leg_obj = boost::json::object{};
    leg_obj["trip_id"] = trip_id;
    leg_obj["from_id"] = from_id;
    leg_obj["to_id"] = to_id;
    leg_obj["coords"] = geo::encode_polyline(
        geo::polyline{geo::latlng{leg.from_.lat_, leg.from_.lon_},
                      geo::latlng{leg.to_.lat_, leg.to_.lon_}});
    leg_obj["sched_start"] = sched_start_m;
    leg_obj["sched_delta"] = sched_delta_m;
    leg_obj["mode"] = static_cast<int>(leg.mode_);

    legs.emplace_back(std::move(leg_obj));
  }

  auto root = boost::json::object{};
  root["legs"] = std::move(legs);
  root["journey_start"] = max_sched_t - itin.duration_;
  return boost::json::serialize(root);
}

api::Itinerary reconstruct_itinerary(motis::ep::stop_times const& stoptimes_ep,
                                     std::string const& itin_id) {
  constexpr auto kLookbackSeconds = 8 * 60;

  auto const root = boost::json::parse(itin_id).as_object();
  auto const& legs = root.at("legs").as_array();
  utl::verify(legs.size() == 1, "reconstruct_itinerary: legs.size() must be 1");

  std::vector<leg_hint> lh_vk;
  for (auto const& lj : legs) {
    lh_vk.emplace_back(lj.as_object());
  }

  auto const journey_start =
      n::unixtime_t{std::chrono::duration_cast<n::i32_minutes>(
          std::chrono::seconds{root.at("journey_start").as_int64()})};
  auto const rtt = stoptimes_ep.rt_ ? stoptimes_ep.rt_->rtt_.get() : nullptr;

  std::vector<std::tuple<n::rt::frun, n::stop_idx_t, n::stop_idx_t>> runs;

  {
    auto const& lh = *begin(lh_vk);
    auto const from_str = stoptimes_in_radius(
        stoptimes_ep, lh.from_latlng_, lh.sched_start_ - kLookbackSeconds,
        lh.mode_, kLookbackSeconds * 2, 100, false);
    auto const to_str = stoptimes_in_radius(
        stoptimes_ep, lh.to_latlng_, lh.sched_end_ - kLookbackSeconds, lh.mode_,
        kLookbackSeconds * 2, 100, true);

    std::vector<id_score> to_cands;
    for (auto const& st : to_str.stopTimes_) {
      if (!st.place_.scheduledArrival_.has_value()) {
        continue;
      }
      to_cands.push_back(
          {st.tripId_,
           (st.tripId_ == lh.trip_id_ ? kExactTripIdMatchAddScore : 0.0) +
               exact_stop_id_match_score(st.place_, lh.to_stop_id_) -
               geo::distance(geo::latlng{st.place_.lat_, st.place_.lon_},
                             lh.to_latlng_) -
               kTimeMul * std::abs(lh.sched_end_ -
                                   to_epoch_seconds(
                                       st.place_.scheduledArrival_.value())),
           &st.place_});
    }
    std::sort(begin(to_cands), end(to_cands));

    // pick the best candidate
    std::optional<std::string_view> best_trip_id;
    std::optional<from_to> best_from_to;
    double best_score = std::numeric_limits<double>::lowest();

    for (auto const& from_st : from_str.stopTimes_) {
      if (!from_st.place_.scheduledDeparture_.has_value()) {
        continue;
      }
      auto it = std::lower_bound(
          begin(to_cands), end(to_cands),
          id_score{from_st.tripId_, std::numeric_limits<double>::max(),
                   nullptr});
      if (it == end(to_cands) || it->id_ != from_st.tripId_) {
        continue;
      }
      double score =
          it->score_ +
          exact_stop_id_match_score(from_st.place_, lh.from_stop_id_) -
          geo::distance(geo::latlng{from_st.place_.lat_, from_st.place_.lon_},
                        lh.from_latlng_) -
          kTimeMul * std::abs(lh.sched_start_ -
                              to_epoch_seconds(
                                  from_st.place_.scheduledDeparture_.value()));
      if (score > best_score) {
        best_score = score;
        best_trip_id = from_st.tripId_;
        best_from_to.emplace(&from_st.place_, it->place_);
      }
    }

    utl::verify(best_trip_id.has_value() && best_from_to.has_value(),
                "no matching route is found");
    auto const best_fr = make_frun_from_stoptime(
        stoptimes_ep.tags_, stoptimes_ep.tt_, rtt, *best_trip_id);

    auto const from_idx =
        find_stop_by_place(best_fr, stoptimes_ep.tags_, *(best_from_to->from_),
                           n::event_type::kDep);
    auto const to_idx = find_stop_by_place(
        best_fr, stoptimes_ep.tags_, *(best_from_to->to_), n::event_type::kArr);
    utl::verify(from_idx.has_value() && to_idx.has_value(),
                "reconstruct_itinerary: could not map from/to stop in frun");
    utl::verify(*from_idx < *to_idx,
                "reconstruct_itinerary: invalid stop order (from >= to)");

    runs.push_back({best_fr, *from_idx, *to_idx});
  }

  auto res = build_itinerary_from_frun(runs, stoptimes_ep, rtt, journey_start);
  res.id_ = itin_id;
  return res;
}
}  // namespace motis
