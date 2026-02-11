#include "motis/itinerary_id.h"

#include "motis/constants.h"
#include "motis/gbfs/routing_data.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "boost/json.hpp"
#include "utl/verify.h"

#include "nigiri/rt/frun.h"

#include "motis/journey_to_response.h"
#include "motis/timetable/time_conv.h"

#include "motis/place.h"
#include "motis/tag_lookup.h"

namespace {
std::int64_t to_epoch_seconds(openapi::date_time_t const& t) {
  auto const sys = static_cast<std::chrono::sys_seconds>(t);
  return std::chrono::duration_cast<std::chrono::seconds>(
             sys.time_since_epoch())
      .count();
}

std::string_view require_stop_id(motis::api::Place const& p,
                                 std::size_t const leg_idx,
                                 std::string_view field) {
  utl::verify(p.stopId_.has_value() && !p.stopId_->empty(),
              "itinerary id: leg {} missing {}.stopId", leg_idx, field);
  return *p.stopId_;
}

std::string_view require_trip_id(motis::api::Leg const& leg,
                                 std::size_t const leg_idx) {
  utl::verify(leg.tripId_.has_value() && !leg.tripId_->empty(),
              "itinerary id: leg {} missing tripId (mode={})", leg_idx,
              static_cast<int>(leg.mode_));
  return *leg.tripId_;
}

struct leg_hint {
  std::string trip_id;
  std::string from_stop_id;
  std::string to_stop_id;
  std::int64_t sched_start;
  std::int64_t sched_end;
  //
  geo::latlng from_latlng;
  geo::latlng to_latlng;
  motis::api::ModeEnum mode;

  explicit leg_hint(boost::json::object const& l) {
    auto const& from_coord = l.at("from_coord").as_array();
    auto const& to_coord = l.at("to_coord").as_array();

    trip_id = l.at("trip_id").as_string().c_str();
    from_stop_id = l.at("from_id").as_string().c_str();
    to_stop_id = l.at("to_id").as_string().c_str();
    sched_start = l.at("sched_start").as_int64();
    sched_end = l.at("sched_end").as_int64();
    from_latlng =
        geo::latlng{from_coord.at(0).as_double(), from_coord.at(1).as_double()};
    to_latlng =
        geo::latlng{to_coord.at(0).as_double(), to_coord.at(1).as_double()};
    mode = static_cast<motis::api::ModeEnum>(l.at("mode").as_int64());
  }
};

motis::api::stoptimes_response stoptimes_in_radius(
    motis::ep::stop_times const& st_ep,
    std::string_view center_stop_id,
    std::int64_t sched_start,
    motis::api::ModeEnum mode,
    int n,
    int radius_m) {
  std::ostringstream oss;
  oss << mode;

  auto const t =
      std::chrono::system_clock::time_point{std::chrono::seconds{sched_start}};
  auto const iso = fmt::format("{:%FT%TZ}", t);

  auto const query = fmt::format(
      "?stopId={}&time={}&arriveBy=false&direction=LATER&n={}"
      "&radius={}&exactRadius=true&fetchStops=true&mode={}",
      center_stop_id, iso, n, radius_m, oss.str());

  return st_ep(boost::urls::url_view{query});
}

struct stop_match {
  motis::api::StopTime st;
  motis::api::Place to;
};

std::optional<stop_match> pick_best_candidate(
    motis::api::stoptimes_response const& response, leg_hint const& lh) {
  std::optional<stop_match> best;
  double best_score = -1e18;

  for (auto const& st : response.stopTimes_) {
    if (st.mode_ != lh.mode) continue;
    if (!st.nextStops_) continue;

    //
    double base_score =
        -geo::distance({st.place_.lat_, st.place_.lon_}, lh.from_latlng);

    //
    // double prev_dist = 1e18;
    for (auto const& ns : *st.nextStops_) {
      if (!ns.stopId_) continue;
      // LATER
      // if (*ns.stopId_ == lh.to_stop_id) {
      //}
      auto dist = geo::distance(lh.to_latlng, {ns.lat_, ns.lon_});
      auto score = base_score - dist;
      if (best_score < score) {
        best_score = score;
        best = {st, ns};
      }
      // early break if we're getting further away
      // if (dist > prev_dist) break;
      // prev_dist = dist;
    }
  }
  return best;
}

nigiri::rt::frun make_frun_from_stoptime(motis::tag_lookup const& tags,
                                         nigiri::timetable const& tt,
                                         nigiri::rt_timetable const* rtt,
                                         motis::api::StopTime const& st) {

  auto const [run, _] = tags.get_trip(tt, rtt, st.tripId_);
  return nigiri::rt::frun{tt, rtt, run};
}

std::optional<nigiri::stop_idx_t> find_stop_by_location(
    nigiri::rt::frun const& fr, nigiri::location_idx_t const loc) {
  for (auto i = nigiri::stop_idx_t{0U}; i < fr.size(); ++i) {
    auto const rs = fr[i];
    if (rs.get_location_idx() == loc) {
      return rs.stop_idx_;
    }
  }
  return std::nullopt;
}

std::optional<nigiri::stop_idx_t> find_stop_by_place(
    nigiri::rt::frun const& fr,
    motis::tag_lookup const& tags,
    motis::api::Place const& p) {

  if (!p.stopId_) {
    return std::nullopt;
  }
  std::cout << "branch: " << p.stopId_.value() << std::endl;  // TEMP
  if (p.stopId_->find('_') != std::string::npos) {
    auto const loc = tags.get_location(*fr.tt_, *p.stopId_);
    return find_stop_by_location(fr, loc);
  }
  for (auto i = nigiri::stop_idx_t{0U}; i < fr.size(); ++i) {
    if (fr[i].get_location_id() == *p.stopId_) {
      return i;
    }
  }

  return std::nullopt;
}

motis::api::Itinerary build_itinerary_from_frun(
    nigiri::rt::frun const& fr,
    nigiri::stop_idx_t from_idx,
    nigiri::stop_idx_t to_idx,
    motis::ep::stop_times const& stoptimes_ep,
    nigiri::rt_timetable const* rtt,
    std::optional<nigiri::unixtime_t> journey_start) {

  utl::verify(
      fr.stop_range_.contains(from_idx) && fr.stop_range_.contains(to_idx),
      "build_itinerary_from_frun: stop idx out of range");

  auto const from_rs = nigiri::rt::run_stop{&fr, from_idx};
  auto const to_rs = nigiri::rt::run_stop{&fr, to_idx};

  auto const dep = from_rs.time(nigiri::event_type::kDep);
  auto const arr = to_rs.time(nigiri::event_type::kArr);

  auto const start_time = journey_start.value_or(dep);

  auto j = nigiri::routing::journey{
      .legs_ = {nigiri::routing::journey::leg{
          nigiri::direction::kForward, from_rs.get_location_idx(),
          to_rs.get_location_idx(), dep, arr,
          nigiri::routing::journey::run_enter_exit{fr, from_idx, to_idx}}},
      .start_time_ = start_time,
      .dest_time_ = arr,
      .dest_ = to_rs.get_location_idx(),
      .transfers_ = 0U};

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
      motis::tt_location{from_rs.get_location_idx(),
                         from_rs.get_scheduled_location_idx()},
      motis::tt_location{to_rs.get_location_idx()}, cache, &blocked, false,
      motis::osr_parameters{}, motis::api::PedestrianProfileEnum::FOOT,
      motis::api::ElevationCostsEnum::NONE, join_interlined_legs,
      detailed_transfers, with_fares, with_scheduled_skipped_stops,
      stoptimes_ep.config_.timetable_.value().max_matching_distance_,
      motis::kMaxMatchingDistance, api_version, false, false, std::nullopt);
}
}  // namespace

namespace motis {

std::string generate_itinerary_id(api::Itinerary const& itin) {
  // utl::verify(!itin.legs_.empty(), "generate_itinerary_id: itinerary has no
  // legs");
  utl::verify(itin.legs_.size() == 1, "itin.legs_.size() != 1");  // TEMP

  auto legs = boost::json::array{};
  legs.reserve(itin.legs_.size());

  for (std::size_t i = 0; i < itin.legs_.size(); ++i) {
    auto const& leg = itin.legs_[i];

    auto const trip_id = require_trip_id(leg, i);
    auto const from_id = require_stop_id(leg.from_, i, "from");
    auto const to_id = require_stop_id(leg.to_, i, "to");

    auto const sched_start = to_epoch_seconds(leg.scheduledStartTime_);
    auto const sched_end = to_epoch_seconds(leg.scheduledEndTime_);
    utl::verify(sched_start != 0 && sched_end != 0,
                "itinerary id: leg {} missing scheduled times", i);
    utl::verify(sched_end >= sched_start,
                "itinerary id: leg {} scheduledEndTime < scheduledStartTime",
                i);

    auto leg_obj = boost::json::object{};
    leg_obj["trip_id"] = trip_id;
    leg_obj["from_id"] = from_id;
    leg_obj["from_coord"] = boost::json::array{leg.from_.lat_, leg.from_.lon_};
    leg_obj["to_id"] = to_id;
    leg_obj["to_coord"] = boost::json::array{leg.to_.lat_, leg.to_.lon_};
    leg_obj["sched_start"] = sched_start;
    leg_obj["sched_end"] = sched_end;
    leg_obj["mode"] = static_cast<int>(leg.mode_);

    legs.emplace_back(std::move(leg_obj));
  }

  auto root = boost::json::object{};
  root["legs"] = std::move(legs);
  root["duration"] = itin.duration_;
  return boost::json::serialize(root);
}
api::Itinerary reconstruct_itinerary(motis::ep::stop_times const& stoptimes_ep,
                                     std::string const& itin_id) {
  constexpr auto lookback_t = 3 * 60;
  // reconstruction. Assuming single leg // TEMP
  auto const root = boost::json::parse(itin_id).as_object();
  auto const& legs = root.at("legs").as_array();
  utl::verify(legs.size() == 1,
              "reconstruct_itinerary: legs.size() != 1");  // TEMP

  auto const& l = legs.at(0).as_object();
  auto const& lh = leg_hint(l);

  //
  auto const total_duration = root.at("duration").as_int64();
  auto const leg_duration = lh.sched_end - lh.sched_start;
  auto const query_start_sec = lh.sched_start - (total_duration - leg_duration);

  auto const journey_start =
      nigiri::unixtime_t{std::chrono::duration_cast<nigiri::i32_minutes>(
          std::chrono::seconds{query_start_sec})};

  //
  auto const response =
      stoptimes_in_radius(stoptimes_ep, lh.from_stop_id,
                          lh.sched_start - lookback_t, lh.mode, 70, 100);

  auto const best = pick_best_candidate(response, lh);
  utl::verify(best.has_value(),
              "reconstruct_itinerary: no matching trip found");

  auto const fr =
      make_frun_from_stoptime(stoptimes_ep.tags_, stoptimes_ep.tt_,
                              stoptimes_ep.rt_->rtt_.get(), best->st);

  auto const from_idx =
      find_stop_by_place(fr, stoptimes_ep.tags_, best->st.place_);
  auto const to_idx = find_stop_by_place(fr, stoptimes_ep.tags_, best->to);
  utl::verify(
      from_idx.has_value() && to_idx.has_value(),
      //              "form_idx.has_value() && to_idx.has_value() isn't true");
      "reconstruct_itinerary: could not map from/to stop in frun");

  return build_itinerary_from_frun(
      fr, *from_idx, *to_idx, stoptimes_ep,
      stoptimes_ep.rt_ ? stoptimes_ep.rt_->rtt_.get() : nullptr, journey_start);
}
}  // namespace motis
