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

namespace n = nigiri;
namespace mm = motis::module;
namespace fbs = flatbuffers;

namespace motis::nigiri {

struct ev_iterator {
  virtual bool finished() const = 0;
  virtual n::unixtime_t time() const = 0;
  virtual n::rt::run get() const = 0;
  virtual void increment() = 0;
  virtual void decrement() = 0;
};

struct static_ev_iterator : public ev_iterator {
  static_ev_iterator(n::timetable const& tt, n::rt_timetable const* rtt,
                     n::route_idx_t const r, n::stop_idx_t const stop_idx,
                     n::unixtime_t const start, n::event_type const ev_type) {}

  bool finished() const override { return true; }
  n::unixtime_t time() const override { return {}; }
  n::rt::run get() const override { return {}; }
  void increment() override {}
  void decrement() override {}
};

struct rt_ev_iterator : public ev_iterator {
  rt_ev_iterator(n::timetable const& tt, n::rt_timetable const& rtt,
                 n::stop_idx_t const stop_idx, n::unixtime_t const start,
                 n::event_type const ev_type) {}

  bool finished() const override { return true; }
  n::unixtime_t time() const override { return {}; }
  n::rt::run get() const override { return {}; }
  void increment() override {}
  void decrement() override {}
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
        for (auto const [stop_idx, s] :
             utl::enumerate(rtt->rt_transport_location_seq_[rt_t])) {
          if (n::stop{s}.location_idx() == x) {
            iterators.emplace_back(std::make_unique<rt_ev_iterator>(
                tt, *rtt, static_cast<n::stop_idx_t>(stop_idx), time, ev_type));
          }
        }
      }
    }
  }

  for (auto const x : locations) {
    for (auto const r : tt.location_routes_[x]) {
      for (auto const [stop_idx, s] :
           utl::enumerate(tt.route_location_seq_[r])) {
        if (n::stop{s}.location_idx() == x) {
          iterators.emplace_back(std::make_unique<static_ev_iterator>(
              tt, rtt, r, stop_idx, time, ev_type));
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
  while (!all_finished() || evs.size() >= count) {
    auto const it = std::min_element(
        begin(iterators), end(iterators), [&](auto const& a, auto const& b) {
          if (a->finished() || b->finished()) {
            return a->finished() < b->finished();
          }
          return fwd ? a->time() < b->time() : a->time() > b->time();
        });
    assert(!(*it)->finished());
    evs.emplace_back((*it)->get());
    fwd ? (*it)->increment() : (*it)->decrement();
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
  auto const departures = get_events(
      locations, tt, rtt, time, n::event_type::kDep, dir, req->event_count());
  auto const arrivals = get_events(
      locations, tt, rtt, time, n::event_type::kArr, dir, req->event_count());

  mm::message_creator fbb;

  auto const write = [&](n::rt::run const r, n::event_type const ev_type) {
    auto const fr = n::rt::frun{tt, rtt, r};
    auto const range = Range{0, 0};
    return railviz::CreateEvent(
        fbb,
        fbb.CreateVector(std::vector{CreateTripInfo(
            fbb,
            to_fbs(fbb, nigiri_trip_to_extern_trip(tags, tt, fr.trip_idx(),
                                                   fr.t_.day_)),
            CreateTransport(
                fbb, &range, static_cast<std::uint32_t>(fr[0].get_clasz()),
                fbb.CreateString(fr[0].line()), fbb.CreateString(fr.name()),
                fbb.CreateString(fr[0].get_provider().long_name_),
                fbb.CreateString(fr[0].direction())))}),
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

  auto events = std::vector<fbs::Offset<railviz::Event>>{departures.size() +
                                                         arrivals.size()};
  auto i = 0U;
  for (auto const& dep : departures) {
    events[i++] = write(dep, n::event_type::kDep);
  }
  for (auto const& arr : arrivals) {
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