#include "motis/itinerary_id.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

#include "utl/overloaded.h"
#include "utl/verify.h"

#include "geo/polyline_format.h"

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

using proto_id_t = ::motis::ItineraryId;
using proto_leg_t = ::motis::LegId;
using resolved_run = std::tuple<n::rt::frun, n::stop_idx_t, n::stop_idx_t>;

struct walk_segment {
  bool is_first_mile() const {
    return from_loc_ == n::get_special_station(n::special_station::kStart);
  }
  bool is_last_mile() const {
    return to_loc_ == n::get_special_station(n::special_station::kEnd);
  }
  bool is_offset() const { return is_first_mile() || is_last_mile(); }

  n::location_idx_t from_loc_;
  n::location_idx_t to_loc_;
  std::optional<geo::latlng> from_pos_;
  std::optional<geo::latlng> to_pos_;
  n::unixtime_t dep_;
  n::unixtime_t arr_;
};

using leg_item = std::variant<resolved_run, walk_segment>;

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

struct leg_hint {
  explicit leg_hint(proto_leg_t const& l)
      : display_name_{l.display_name()},
        trip_id_{l.trip_id()},
        from_stop_id_{l.from_id()},
        to_stop_id_{l.to_id()},
        sched_start_{l.sched_start()},
        sched_end_{l.sched_end()},
        mode_{json::value_to<api::ModeEnum>(json::value{l.mode()})},
        scheduled_{l.scheduled()} {
    if (!trip_id_.empty()) {
      utl::verify(!display_name_.empty(), "itinerary id: display_name missing");
      utl::verify(!from_stop_id_.empty(), "itinerary id: from_id missing");
      utl::verify(!to_stop_id_.empty(), "itinerary id: to_id missing");
    }
    decode_coords(l.coords());
  }

  bool is_public_transport() const { return !trip_id_.empty(); }

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
  id.set_coords(geo::encode_polyline(
      {{l.from_.lat_, l.from_.lon_}, {l.to_.lat_, l.to_.lon_}}));
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
         l_idx == default_display_names_indices[default_name_idx])
            ? default_display_names[default_name_idx++]
            : leg.displayName_.value_or("");
    id.mutable_legs()->Add(get_leg_id_proto(leg, display_name));
  }
  //

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
  std::optional<openapi::date_time_t> scheduledArrival_{};
  std::optional<openapi::date_time_t> scheduledDeparture_{};
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
      res.scheduledArrival_ = {s.scheduled_time(n::event_type::kArr)};
    }
    if (fr.stop_range_.from_ != fr.size() - 1U) {
      res.scheduledDeparture_ = {s.scheduled_time(n::event_type::kDep)};
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

n::unixtime_t item_dep(leg_item const& item) {
  return std::visit(
      utl::overloaded{[](resolved_run const& r) {
                        auto const& [fr, from_idx, _] = r;
                        return n::rt::run_stop{&fr, from_idx}.time(
                            n::event_type::kDep);
                      },
                      [](walk_segment const& w) { return w.dep_; }},
      item);
}

n::unixtime_t item_arr(leg_item const& item) {
  return std::visit(
      utl::overloaded{[](resolved_run const& r) {
                        auto const& [fr, _, to_idx] = r;
                        return n::rt::run_stop{&fr, to_idx}.time(
                            n::event_type::kArr);
                      },
                      [](walk_segment const& w) { return w.arr_; }},
      item);
}

constexpr auto kWalkTransportModeId = static_cast<n::transport_mode_id_t>(
    static_cast<std::underlying_type_t<osr::search_profile>>(
        osr::search_profile::kFoot));

place_t endpoint_place(leg_item const& it, bool const from) {
  if (auto const* w = std::get_if<walk_segment>(&it)) {
    if (from && w->is_first_mile() && w->from_pos_.has_value()) {
      return osr::location{*w->from_pos_, osr::kNoLevel};
    }
    if (!from && w->is_last_mile() && w->to_pos_.has_value()) {
      return osr::location{*w->to_pos_, osr::kNoLevel};
    }
    return tt_location{from ? w->from_loc_ : w->to_loc_};
  }
  auto const& [fr, from_idx, to_idx] = std::get<resolved_run>(it);
  return tt_location{
      n::rt::run_stop{&fr, from ? from_idx : to_idx}.get_location_idx()};
}

api::Itinerary build_itinerary_from_items(
    std::vector<leg_item>& items,
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
    unsigned int const api_version) {
  utl::verify(!items.empty(), "build_itinerary_from_items: no items");

  // TODO: check for viability
  // Re-time walk segments:
  for (auto i = std::size_t{0}; i < items.size(); ++i) {
    if (auto* w = std::get_if<walk_segment>(&items[i])) {
      if (i > 0) {
        w->dep_ = item_arr(items[i - 1]);
      }
      if (i + 1 < items.size()) {
        w->arr_ = item_dep(items[i + 1]);
      }
    }
  }

  auto j = n::routing::journey{};
  j.legs_.reserve(items.size());
  auto pt_count = std::size_t{0};
  for (auto const& item : items) {
    std::visit(
        utl::overloaded{
            [&](resolved_run const& r) {
              auto const& [fr, from_idx, to_idx] = r;
              utl::verify(fr.stop_range_.contains(from_idx),
                          "from_idx={} out of range {}", from_idx,
                          fr.stop_range_);
              utl::verify(fr.stop_range_.contains(to_idx),
                          "to_idx={} out of range {}", to_idx, fr.stop_range_);
              auto const rs_from = n::rt::run_stop{&fr, from_idx};
              auto const rs_to = n::rt::run_stop{&fr, to_idx};
              j.legs_.emplace_back(n::routing::journey::leg{
                  n::direction::kForward, rs_from.get_location_idx(),
                  rs_to.get_location_idx(), rs_from.time(n::event_type::kDep),
                  rs_to.time(n::event_type::kArr),
                  n::routing::journey::run_enter_exit{fr, from_idx, to_idx}});
              ++pt_count;
            },
            [&](walk_segment const& w) {
              auto const dur = std::max(
                  n::duration_t{0},
                  std::chrono::duration_cast<n::duration_t>(w.arr_ - w.dep_));
              if (w.is_offset()) {
                j.legs_.emplace_back(n::routing::journey::leg{
                    n::direction::kForward, w.from_loc_, w.to_loc_, w.dep_,
                    w.arr_,
                    n::routing::offset{w.to_loc_, dur, kWalkTransportModeId}});
              } else {
                j.legs_.emplace_back(n::routing::journey::leg{
                    n::direction::kForward, w.from_loc_, w.to_loc_, w.dep_,
                    w.arr_, n::footpath{w.to_loc_, dur}});
              }
            }},
        item);
  }
  j.start_time_ = j.legs_.front().dep_time_;
  j.dest_time_ = j.legs_.back().arr_time_;
  j.dest_ = j.legs_.back().to_;
  j.transfers_ =
      pt_count == 0 ? std::uint8_t{0} : static_cast<std::uint8_t>(pt_count - 1);

  auto cache = street_routing_cache_t{};
  auto blocked = osr::bitvec<osr::node_idx_t>{};
  auto gbfs_rd = gbfs::gbfs_routing_data{};

  return journey_to_response(
      stop_times.w_, l, stop_times.pl_, stop_times.tt_, stop_times.tags_,
      nullptr, nullptr, rtt, stop_times.matches_, nullptr, shapes, gbfs_rd,
      stop_times.ae_, stop_times.tz_, j, endpoint_place(items.front(), true),
      endpoint_place(items.back(), false), cache, &blocked, false,
      osr_parameters{}, api::PedestrianProfileEnum::FOOT,
      api::ElevationCostsEnum::NONE, join_interlined_legs, detailed_transfers,
      detailed_legs, with_fares, with_scheduled_skipped_stops,
      stop_times.config_.timetable_.value().max_matching_distance_,
      kMaxMatchingDistance, api_version, false, false, lang, false);
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
    if (!st.scheduledDeparture_.has_value()) {
      continue;
    }

    if (require_display_name_match &&
        hint.display_name_ !=
            st.run_[0].display_name(n::event_type::kDep, n::lang_t{})) {
      continue;
    }

    from_cands.emplace_back(
        &st, -geo::distance(st.run_[0].pos(), hint.from_pos_) -
                 kTimeMul *
                     std::abs(hint.sched_start_ - st.scheduledDeparture_.value()
                                                      .get_unixtime_seconds()));
  }

  if (from_cands.size() == 0) {
    return std::nullopt;
  }

  for (auto const& st : to_resp) {
    if (!st.scheduledArrival_.has_value()) {
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
                         st.scheduledArrival_.value().get_unixtime_seconds()));
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
                                     unsigned int const api_version) {
  auto stop_times_rt = std::atomic_load(&stop_times_ep.rt_);
  auto stop_times_rtt = stop_times_rt->rtt_.get();
  auto const parsed_id = decode_itinerary_id(id);

  auto const get_run = [&](leg_hint const& lh) -> resolved_run {
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

      auto const best_from_to = get_best_candidate(
          from_st_res, to_st_res, lh, require_display_name_match,
          stop_times_ep.tags_, stop_times_rtt);

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

  auto const n_legs = parsed_id.legs_size();
  auto items = std::vector<leg_item>{};
  items.reserve(static_cast<std::size_t>(n_legs));
  for (auto l_idx = 0; l_idx < n_legs; ++l_idx) {
    auto const lh = leg_hint{parsed_id.legs(l_idx)};
    if (lh.is_public_transport()) {
      items.emplace_back(get_run(lh));
      continue;
    }

    auto from_loc = n::location_idx_t::invalid();
    auto to_loc = n::location_idx_t::invalid();
    auto from_pos = std::optional<geo::latlng>{};
    auto to_pos = std::optional<geo::latlng>{};

    if (lh.from_stop_id_.empty()) {
      utl::verify(l_idx == 0,
                  "reconstruct_itinerary: non-PT leg without from_id is only "
                  "valid as the first leg (first-mile)");
      from_loc = n::get_special_station(n::special_station::kStart);
      from_pos = lh.from_pos_;
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
      to_pos = lh.to_pos_;
    } else {
      auto const t =
          stop_times_ep.tags_.find_location(stop_times_ep.tt_, lh.to_stop_id_);
      utl::verify(t.has_value(),
                  "reconstruct_itinerary: non-PT leg to stop not found");
      to_loc = *t;
    }

    items.emplace_back(walk_segment{
        from_loc, to_loc, from_pos, to_pos,
        n::unixtime_t{std::chrono::duration_cast<n::unixtime_t::duration>(
            std::chrono::seconds{lh.sched_start_})},
        n::unixtime_t{std::chrono::duration_cast<n::unixtime_t::duration>(
            std::chrono::seconds{lh.sched_end_})}});
  }

  auto res = build_itinerary_from_items(
      items, stop_times_ep, l, shapes, rt.rtt_.get(), join_interlined_legs,
      detailed_transfers, detailed_legs, with_fares,
      with_scheduled_skipped_stops, lang, api_version);
  res.id_ = id;
  return res;
}

}  // namespace motis
