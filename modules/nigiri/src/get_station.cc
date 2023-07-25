#include "motis/nigiri/trip_to_connection.h"

#include <memory>

#include "utl/concat.h"
#include "utl/enumerate.h"

#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
#include "nigiri/timetable.h"

#include "motis/core/conv/position_conv.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/extern_trip.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/nigiri/extern_trip.h"
#include "motis/nigiri/location.h"
#include "motis/nigiri/resolve_run.h"
#include "motis/nigiri/unixtime_conv.h"

#ifdef CreateEvent
#undef CreateEvent
#endif

namespace n = nigiri;
namespace mm = motis::module;
namespace fbs = flatbuffers;

namespace motis::nigiri {

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
  static_ev_iterator(n::timetable const& tt, n::rt_timetable const* rtt,
                     n::route_idx_t const r, n::stop_idx_t const stop_idx,
                     n::unixtime_t const start, n::event_type const ev_type,
                     n::direction const dir)
      : tt_{tt},
        rtt_{rtt},
        day_{to_idx(tt_.day_idx_mam(start).first)},
        end_day_{dir == n::direction::kForward
                     ? to_idx(tt.day_idx(tt.date_range_.to_))
                     : -1},
        size_{static_cast<std::int32_t>(
            to_idx(tt.route_transport_ranges_[r].size()))},
        i_{dir == n::direction::kForward ? 0 : size_ - 1},
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
    if (dir_ == n::direction::kForward) {
      while (!finished()) {
        for (; i_ < size_; ++i_) {
          if (start.has_value() && time() < *start) {
            continue;
          }
          if (is_active()) {
            return;
          }
        }
        ++day_;
        i_ = 0;
      }
    } else {
      while (!finished()) {
        for (; i_ > 0; --i_) {
          if (start.has_value() && time() > *start) {
            continue;
          }
          if (is_active()) {
            return;
          }
        }
        --day_;
        i_ = size_ - 1;
      }
    }
  }

  bool finished() const override { return day_ == end_day_; }

  n::unixtime_t time() const override {
    return tt_.event_time(
        n::transport{tt_.route_transport_ranges_[r_][i_], n::day_idx_t{day_}},
        stop_idx_, ev_type_);
  }

  n::rt::run get() const override {
    assert(is_active());
    return n::rt::run{
        .t_ = n::transport{tt_.route_transport_ranges_[r_][i_],
                           n::day_idx_t{day_}},
        .stop_range_ = {stop_idx_, static_cast<n::stop_idx_t>(stop_idx_ + 1U)}};
  }

  void increment() override {
    dir_ == n::direction::kForward ? ++i_ : --i_;
    seek_next();
  }

private:
  bool is_active() const {
    auto const x = t();
    return (rtt_ == nullptr
                ? tt_.bitfields_[tt_.transport_traffic_days_[x.t_idx_]]
                : rtt_->bitfields_[rtt_->transport_traffic_days_[x.t_idx_]])
        .test(to_idx(x.day_));
  }

  n::transport t() const {
    auto const t = tt_.route_transport_ranges_[r_][i_];
    auto const day_offset = tt_.event_mam(r_, t, stop_idx_, ev_type_).days();
    return n::transport{tt_.route_transport_ranges_[r_][i_],
                        n::day_idx_t{day_ - day_offset}};
  }

  n::timetable const& tt_;
  n::rt_timetable const* rtt_;
  std::int32_t day_, end_day_, size_, i_;
  n::route_idx_t r_;
  n::stop_idx_t stop_idx_;
  n::event_type ev_type_;
  n::direction dir_;
};

struct rt_ev_iterator : public ev_iterator {
  rt_ev_iterator(n::rt_timetable const& rtt, n::rt_transport_idx_t const rt_t,
                 n::stop_idx_t const stop_idx, n::unixtime_t const start,
                 n::event_type const ev_type, n::direction const dir)
      : rtt_{rtt}, stop_idx_{stop_idx}, rt_t_{rt_t}, ev_type_{ev_type} {
    finished_ = dir == n::direction::kForward ? time() < start : time() > start;
    assert((ev_type == n::event_type::kDep &&
            stop_idx_ < rtt.rt_transport_location_seq_[rt_t].size() - 1U) ||
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
  bool finished_{false};
  n::stop_idx_t stop_idx_;
  n::rt_transport_idx_t rt_t_;
  n::event_type ev_type_;
};

std::vector<n::rt::run> get_events(
    std::vector<n::location_idx_t> const& locations, n::timetable const& tt,
    n::rt_timetable const* rtt, n::unixtime_t const time,
    n::event_type const ev_type, n::direction const dir,
    std::size_t const count) {
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
                dir));
          }
        }
      }
    }
  }

  auto seen = n::hash_set<std::pair<n::route_idx_t, n::stop_idx_t>>{};
  for (auto const x : locations) {
    for (auto const r : tt.location_routes_[x]) {
      auto const location_seq = tt.route_location_seq_[r];
      for (auto const [stop_idx, s] : utl::enumerate(location_seq)) {
        if (n::stop{s}.location_idx() == x &&
            ((ev_type == n::event_type::kDep &&
              stop_idx != location_seq.size() - 1U) ||
             (ev_type == n::event_type::kArr && stop_idx != 0U)) &&
            seen.emplace(r, stop_idx).second) {
          iterators.emplace_back(std::make_unique<static_ev_iterator>(
              tt, rtt, r, stop_idx, time, ev_type, dir));
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
  while (!all_finished() && evs.size() < count) {
    auto const it = std::min_element(
        begin(iterators), end(iterators), [&](auto const& a, auto const& b) {
          if (a->finished() || b->finished()) {
            return a->finished() < b->finished();
          }
          return fwd ? a->time() < b->time() : a->time() > b->time();
        });
    assert(!(*it)->finished());
    evs.emplace_back((*it)->get());
    (*it)->increment();
  }
  return evs;
}

mm::msg_ptr get_station(tag_lookup const& tags, n::timetable const& tt,
                        n::rt_timetable const* rtt, mm::msg_ptr const& msg) {
  using railviz::RailVizStationRequest;
  auto const req = motis_content(RailVizStationRequest, msg);

  auto const time = to_nigiri_unixtime(req->time());
  auto const l = get_location_idx(tags, tt, req->station_id()->view());
  auto const l_name = tt.locations_.names_[l].view();

  auto locations = std::vector{l};
  utl::concat(locations, tt.locations_.children_[l]);
  for (auto const eq : tt.locations_.equivalences_[l]) {
    if (tt.locations_.names_[eq].view() == l_name) {
      locations.emplace_back(eq);
    }
  }

  auto const dir = req->direction() != railviz::Direction_EARLIER
                       ? n::direction::kForward
                       : n::direction::kBackward;
  auto const deps = get_events(locations, tt, rtt, time, n::event_type::kDep,
                               dir, req->event_count());
  auto const arrs = get_events(locations, tt, rtt, time, n::event_type::kArr,
                               dir, req->event_count());

  mm::message_creator fbb;

  auto const write = [&](n::rt::run const r, n::event_type const ev_type) {
    auto const fr = n::rt::frun{tt, rtt, r};
    auto const range = Range{0, 0};
    return railviz::CreateEvent(
        fbb,
        fbb.CreateVector(std::vector{CreateTripInfo(
            fbb,
            to_fbs(fbb, nigiri_trip_to_extern_trip(
                            tags, tt, fr[0].get_trip_idx(ev_type), fr.t_)),
            CreateTransport(
                fbb, &range, static_cast<std::uint32_t>(fr[0].get_clasz()),
                fbb.CreateString(fr[0].line(ev_type)),
                fbb.CreateString(fr.name()),
                fbb.CreateString(fr[0].get_provider(ev_type).long_name_),
                fbb.CreateString(fr[0].direction(ev_type))))}),
        ev_type == n::event_type::kDep ? EventType_DEP : EventType_ARR,
        CreateEventInfo(
            fbb, to_motis_unixtime(fr[0].time(ev_type)),
            to_motis_unixtime(fr[0].scheduled_time(ev_type)),
            fbb.CreateString(fr[0].track()), fbb.CreateString(fr[0].track()),
            true,
            fr.is_rt() ? TimestampReason_FORECAST : TimestampReason_SCHEDULE));
  };

  auto const pos = to_fbs(tt.locations_.coordinates_[l]);
  auto const p = tt.locations_.parents_[l] == n::location_idx_t::invalid()
                     ? l
                     : tt.locations_.parents_[l];

  auto events =
      std::vector<fbs::Offset<railviz::Event>>(deps.size() + arrs.size());
  auto i = 0U;
  for (auto const& dep : deps) {
    events[i++] = write(dep, n::event_type::kDep);
  }
  for (auto const& arr : arrs) {
    events[i++] = write(arr, n::event_type::kArr);
  }
  fbb.create_and_finish(
      MsgContent_RailVizStationResponse,
      railviz::CreateRailVizStationResponse(
          fbb,
          CreateStation(fbb, fbb.CreateString(tt.locations_.ids_[l].view()),
                        fbb.CreateString(tt.locations_.names_[p].view()), &pos),
          fbb.CreateVector(events))
          .Union());
  return make_msg(fbb);
}

}  // namespace motis::nigiri