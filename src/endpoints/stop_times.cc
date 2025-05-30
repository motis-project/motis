#include "motis/endpoints/stop_times.h"

#include <memory>

#include "utl/concat.h"
#include "utl/enumerate.h"
#include "utl/erase_duplicates.h"

#include "nigiri/routing/clasz_mask.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "motis/data.h"
#include "motis/journey_to_response.h"
#include "motis/parse_location.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/clasz_to_mode.h"
#include "motis/timetable/modes_to_clasz_mask.h"
#include "motis/timetable/time_conv.h"

namespace n = nigiri;

namespace motis::ep {

struct ev_iterator {
  ev_iterator() = default;
  ev_iterator(ev_iterator const&) = delete;
  ev_iterator(ev_iterator&&) = delete;
  ev_iterator& operator=(ev_iterator const&) = delete;
  ev_iterator& operator=(ev_iterator&&) = delete;
  virtual ~ev_iterator() = default;
  virtual bool finished() const = 0;
  virtual n::unixtime_t time() const = 0;
  virtual n::rt::run get() const = 0;
  virtual void increment() = 0;
};

struct static_ev_iterator : public ev_iterator {
  static_ev_iterator(n::timetable const& tt,
                     n::rt_timetable const* rtt,
                     n::route_idx_t const r,
                     n::stop_idx_t const stop_idx,
                     n::unixtime_t const start,
                     n::event_type const ev_type,
                     n::direction const dir)
      : tt_{tt},
        rtt_{rtt},
        day_{to_idx(tt_.day_idx_mam(start).first)},
        end_day_{dir == n::direction::kForward
                     ? to_idx(tt.day_idx(tt_.date_range_.to_))
                     : to_idx(tt.day_idx(tt_.date_range_.from_) - 1U)},
        size_{tt_.route_transport_ranges_[r].size()},
        i_{0},
        r_{r},
        stop_idx_{stop_idx},
        ev_type_{ev_type},
        dir_{dir} {
    seek_next(start);
  }

  ~static_ev_iterator() override = default;

  static_ev_iterator(static_ev_iterator const&) = delete;
  static_ev_iterator(static_ev_iterator&&) = delete;
  static_ev_iterator& operator=(static_ev_iterator const&) = delete;
  static_ev_iterator& operator=(static_ev_iterator&&) = delete;

  void seek_next(std::optional<n::unixtime_t> const start = std::nullopt) {
    while (!finished()) {
      for (; i_ < size_; ++i_) {
        if (start.has_value() &&
            (dir_ == n::direction::kForward ? time() < *start
                                            : time() > *start)) {
          continue;
        }
        if (is_active()) {
          return;
        }
      }
      dir_ == n::direction::kForward ? ++day_ : --day_;
      i_ = 0;
    }
  }

  bool finished() const override { return day_ == end_day_; }

  n::unixtime_t time() const override {
    return tt_.event_time(t(), stop_idx_, ev_type_);
  }

  n::rt::run get() const override {
    assert(is_active());
    return n::rt::run{
        .t_ = t(),
        .stop_range_ = {stop_idx_, static_cast<n::stop_idx_t>(stop_idx_ + 1U)}};
  }

  void increment() override {
    ++i_;
    seek_next();
  }

private:
  bool is_active() const {
    auto const x = t();
    auto const in_static =
        tt_.bitfields_[tt_.transport_traffic_days_[x.t_idx_]].test(
            to_idx(x.day_));
    return rtt_ == nullptr
               ? in_static
               : in_static &&
                     rtt_->resolve_rt(x) ==  // only when no RT/cancelled
                         n::rt_transport_idx_t::invalid();
  }

  n::transport t() const {
    auto const idx = dir_ == n::direction::kForward ? i_ : size_ - i_ - 1;
    auto const t = tt_.route_transport_ranges_[r_][idx];
    auto const day_offset = tt_.event_mam(r_, t, stop_idx_, ev_type_).days();
    return n::transport{tt_.route_transport_ranges_[r_][idx],
                        n::day_idx_t{to_idx(day_) - day_offset}};
  }

  n::timetable const& tt_;
  n::rt_timetable const* rtt_;
  n::day_idx_t day_, end_day_;
  std::uint32_t size_;
  std::uint32_t i_;
  n::route_idx_t r_;
  n::stop_idx_t stop_idx_;
  n::event_type ev_type_;
  n::direction dir_;
};

struct rt_ev_iterator : public ev_iterator {
  rt_ev_iterator(n::rt_timetable const& rtt,
                 n::rt_transport_idx_t const rt_t,
                 n::stop_idx_t const stop_idx,
                 n::unixtime_t const start,
                 n::event_type const ev_type,
                 n::direction const dir,
                 n::routing::clasz_mask_t const allowed_clasz)
      : rtt_{rtt},
        stop_idx_{stop_idx},
        rt_t_{rt_t},
        ev_type_{ev_type},
        finished_{
            !n::routing::is_allowed(
                allowed_clasz, rtt.rt_transport_section_clasz_[rt_t].at(0)) ||
            (dir == n::direction::kForward ? time() < start : time() > start)} {
    assert((ev_type == n::event_type::kDep &&
            stop_idx_ < rtt_.rt_transport_location_seq_[rt_t].size() - 1U) ||
           (ev_type == n::event_type::kArr && stop_idx_ > 0U));
  }

  ~rt_ev_iterator() override = default;

  rt_ev_iterator(rt_ev_iterator const&) = delete;
  rt_ev_iterator(rt_ev_iterator&&) = delete;
  rt_ev_iterator& operator=(rt_ev_iterator const&) = delete;
  rt_ev_iterator& operator=(rt_ev_iterator&&) = delete;

  bool finished() const override { return finished_; }

  n::unixtime_t time() const override {
    return rtt_.unix_event_time(rt_t_, stop_idx_, ev_type_);
  }

  n::rt::run get() const override {
    return n::rt::run{
        .stop_range_ = {stop_idx_, static_cast<n::stop_idx_t>(stop_idx_ + 1U)},
        .rt_ = rt_t_};
  }

  void increment() override { finished_ = true; }

  n::rt_timetable const& rtt_;
  n::stop_idx_t stop_idx_;
  n::rt_transport_idx_t rt_t_;
  n::event_type ev_type_;
  bool finished_{false};
};

std::vector<n::rt::run> get_events(
    std::vector<n::location_idx_t> const& locations,
    n::timetable const& tt,
    n::rt_timetable const* rtt,
    n::unixtime_t const time,
    n::event_type const ev_type,
    n::direction const dir,
    std::size_t const count,
    n::routing::clasz_mask_t const allowed_clasz) {
  auto iterators = std::vector<std::unique_ptr<ev_iterator>>{};

  if (rtt != nullptr) {
    for (auto const x : locations) {
      for (auto const rt_t : rtt->location_rt_transports_[x]) {
        auto const location_seq = rtt->rt_transport_location_seq_[rt_t];
        for (auto const [stop_idx, s] : utl::enumerate(location_seq)) {
          if (n::stop{s}.location_idx() == x &&
              ((ev_type == n::event_type::kDep &&
                stop_idx != location_seq.size() - 1U) ||
               (ev_type == n::event_type::kArr && stop_idx != 0U))) {
            iterators.emplace_back(std::make_unique<rt_ev_iterator>(
                *rtt, rt_t, static_cast<n::stop_idx_t>(stop_idx), time, ev_type,
                dir, allowed_clasz));
          }
        }
      }
    }
  }

  auto seen = n::hash_set<std::pair<n::route_idx_t, n::stop_idx_t>>{};
  for (auto const x : locations) {
    for (auto const r : tt.location_routes_[x]) {
      if (!n::routing::is_allowed(allowed_clasz, tt.route_clasz_[r])) {
        continue;
      }
      auto const location_seq = tt.route_location_seq_[r];
      for (auto const [stop_idx, s] : utl::enumerate(location_seq)) {
        if (n::stop{s}.location_idx() == x &&
            ((ev_type == n::event_type::kDep &&
              stop_idx != location_seq.size() - 1U) ||
             (ev_type == n::event_type::kArr && stop_idx != 0U)) &&
            seen.emplace(r, static_cast<n::stop_idx_t>(stop_idx)).second) {
          iterators.emplace_back(std::make_unique<static_ev_iterator>(
              tt, rtt, r, static_cast<n::stop_idx_t>(stop_idx), time, ev_type,
              dir));
        }
      }
    }
  }

  auto const all_finished = [&]() {
    return utl::all_of(iterators,
                       [](auto const& it) { return it->finished(); });
  };

  auto const fwd = dir == n::direction::kForward;
  auto evs = std::vector<n::rt::run>{};
  auto last_time = n::unixtime_t{};
  while (!all_finished()) {
    auto const it = std::min_element(
        begin(iterators), end(iterators), [&](auto const& a, auto const& b) {
          if (a->finished() || b->finished()) {
            return a->finished() < b->finished();
          }
          return fwd ? a->time() < b->time() : a->time() > b->time();
        });
    assert(!(*it)->finished());
    if (evs.size() >= count && (*it)->time() != last_time) {
      break;
    }
    evs.emplace_back((*it)->get());
    last_time = (*it)->time();
    (*it)->increment();
  }
  return evs;
}

api::stoptimes_response stop_times::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::stoptimes_params{url.params()};

  auto const max_results = config_.limits_.value().stoptimes_max_results_;
  utl::verify(query.n_ < max_results, "n={} > {} not allowed", query.n_,
              max_results);

  auto const x = tags_.get_location(tt_, query.stopId_);
  auto const p = tt_.locations_.parents_[x];
  auto const l = p == n::location_idx_t::invalid() ? x : p;
  auto const allowed_clasz = to_clasz_mask(query.mode_);
  auto const [dir, time] = parse_cursor(query.pageCursor_.value_or(fmt::format(
      "{}|{}",
      query.direction_
          .transform([](auto&& x) {
            return x == api::directionEnum::EARLIER ? "EARLIER" : "LATER";
          })
          .value_or(query.arriveBy_ ? "EARLIER" : "LATER"),
      std::chrono::duration_cast<std::chrono::seconds>(
          query.time_.value_or(openapi::now())->time_since_epoch())
          .count())));

  auto locations = std::vector{l};
  auto const add = [&](n::location_idx_t const l) {
    auto const l_name = tt_.locations_.names_[l].view();
    utl::concat(locations, tt_.locations_.children_[l]);
    for (auto const eq : tt_.locations_.equivalences_[l]) {
      if (tt_.locations_.names_[eq].view() == l_name) {
        locations.emplace_back(eq);
        utl::concat(locations, tt_.locations_.children_[eq]);
      }
    }
  };

  if (query.radius_) {
    loc_rtree_.in_radius(tt_.locations_.coordinates_[x],
                         static_cast<double>(*query.radius_),
                         [&](n::location_idx_t const y) {
                           if (query.exactRadius_ == true) {
                             locations.emplace_back(y);
                           } else {
                             add(y);
                           }
                         });
  } else {
    add(x);
  }
  utl::erase_duplicates(locations);

  auto const rt = rt_;
  auto const rtt = rt->rtt_.get();
  auto const ev_type =
      query.arriveBy_ ? n::event_type::kArr : n::event_type::kDep;
  auto events = get_events(locations, tt_, rtt, time, ev_type, dir,
                           static_cast<std::size_t>(query.n_), allowed_clasz);
  utl::sort(events, [&](n::rt::run const& a, n::rt::run const& b) {
    auto const fr_a = n::rt::frun{tt_, rtt, a};
    auto const fr_b = n::rt::frun{tt_, rtt, b};
    return fr_a[0].time(ev_type) < fr_b[0].time(ev_type);
  });
  return {
      .stopTimes_ = utl::to_vec(
          events,
          [&](n::rt::run const r) -> api::StopTime {
            auto const fr = n::rt::frun{tt_, rtt, r};
            auto const s = fr[0];
            auto const& agency = s.get_provider(ev_type);
            auto place = to_place(&tt_, &tags_, w_, pl_, matches_,
                                  tt_location{s.get_location_idx(),
                                              s.get_scheduled_location_idx()});
            place.alerts_ =
                get_alerts(fr, std::pair{s, fr.stop_range_.from_ != 0U
                                                ? n::event_type::kArr
                                                : n::event_type::kDep});
            if (fr.stop_range_.from_ != 0U) {
              place.arrival_ = {s.time(n::event_type::kArr)};
              place.scheduledArrival_ = {s.scheduled_time(n::event_type::kArr)};
            }
            if (fr.stop_range_.from_ != fr.size() - 1U) {
              place.departure_ = {s.time(n::event_type::kDep)};
              place.scheduledDeparture_ = {
                  s.scheduled_time(n::event_type::kDep)};
            }
            auto const in_out_allowed =
                !fr.is_cancelled() &&
                (ev_type == n::event_type::kArr ? s.out_allowed()
                                                : s.in_allowed());
            auto const stop_cancelled =
                fr.is_cancelled() ||
                (ev_type == n::event_type::kArr
                     ? !s.out_allowed() && s.get_scheduled_stop().out_allowed()
                     : !s.in_allowed() && s.get_scheduled_stop().in_allowed());

            return {
                .place_ = std::move(place),
                .mode_ = to_mode(s.get_clasz(ev_type)),
                .realTime_ = r.is_rt(),
                .headsign_ = std::string{s.direction(ev_type)},
                .agencyId_ = std::string{tt_.strings_.get(agency.short_name_)},
                .agencyName_ = std::string{tt_.strings_.get(agency.long_name_)},
                .agencyUrl_ = std::string{tt_.strings_.get(agency.url_)},
                .routeColor_ = to_str(s.get_route_color(ev_type).color_),
                .routeTextColor_ =
                    to_str(s.get_route_color(ev_type).text_color_),
                .tripId_ = tags_.id(tt_, s, ev_type),
                .routeShortName_ = std::string{s.trip_display_name(ev_type)},
                .pickupDropoffType_ =
                    in_out_allowed ? api::PickupDropoffTypeEnum::NORMAL
                                   : api::PickupDropoffTypeEnum::NOT_ALLOWED,
                .cancelled_ = stop_cancelled,
                .source_ = fmt::format("{}", fmt::streamed(fr.dbg()))};
          }),
      .previousPageCursor_ =
          events.empty()
              ? ""
              : fmt::format(
                    "EARLIER|{}",
                    to_seconds(
                        n::rt::frun{tt_, rtt, events.front()}[0].time(ev_type) -
                        std::chrono::minutes{1})),
      .nextPageCursor_ =
          events.empty()
              ? ""
              : fmt::format(
                    "LATER|{}",
                    to_seconds(
                        n::rt::frun{tt_, rtt, events.back()}[0].time(ev_type) +
                        std::chrono::minutes{1}))};
}

}  // namespace motis::ep