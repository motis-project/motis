#include "motis/nigiri/trip_to_connection.h"

#include <memory>

#include "utl/concat.h"
#include "utl/enumerate.h"

#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
#include "nigiri/timetable.h"

#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/extern_trip.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/nigiri/location.h"
#include "motis/nigiri/resolve_run.h"
#include "motis/nigiri/unixtime_conv.h"

namespace n = nigiri;
namespace mm = motis::module;

namespace motis::nigiri {

struct ev_iterator {
  virtual bool finished() const = 0;
  virtual n::unixtime_t time() const = 0;
};

struct static_ev_iterator : public ev_iterator {
  static_ev_iterator(n::timetable const& tt, n::rt_timetable const* rtt,
                     n::route_idx_t const r, n::stop_idx_t const stop_idx,
                     n::unixtime_t const start) {}

  bool finished() const override { return true; }
  n::unixtime_t time() const override { return {}; }
};

struct rt_ev_iterator : public ev_iterator {
  rt_ev_iterator(n::timetable const& tt, n::rt_timetable const& rtt,
                 n::stop_idx_t const stop_idx, n::unixtime_t const start) {}

  bool finished() const override { return true; }
  n::unixtime_t time() const override { return {}; }
};

motis::module::msg_ptr get_station(tag_lookup const& tags,
                                   n::timetable const& tt,
                                   n::rt_timetable const* rtt,
                                   motis::module::msg_ptr const& msg) {
  using railviz::RailVizStationRequest;
  auto const req = motis_content(RailVizStationRequest, msg);
  CISTA_UNUSED_PARAM(tags)  // TODO(felix)
  CISTA_UNUSED_PARAM(tt)  // TODO(felix)
  CISTA_UNUSED_PARAM(rtt)  // TODO(felix)

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

  auto iterators = std::vector<std::unique_ptr<ev_iterator>>{};

  if (rtt != nullptr) {
    for (auto const x : locations) {
      for (auto const rt_t : rtt->location_rt_transports_[x]) {
        for (auto const [stop_idx, s] :
             utl::enumerate(rtt->rt_transport_location_seq_[rt_t])) {
          if (n::stop{s}.location_idx() == x) {
            iterators.emplace_back(std::make_unique<rt_ev_iterator>(
                tt, *rtt, static_cast<n::stop_idx_t>(stop_idx), time));
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
          iterators.emplace_back(
              std::make_unique<static_ev_iterator>(tt, rtt, r, stop_idx, time));
        }
      }
    }
  }

  auto const all_finished = [&]() {
    return utl::all_of(iterators,
                       [](auto const& it) { return it->finished(); });
  };

  auto const ev = std::vector<n::rt::run>{};
  while (!all_finished()) {
    std::min_element(
        begin(iterators), end(iterators),
        [](auto const& a, auto const& b) { return a->time() < b->time(); });
  }

  return mm::make_success_msg();
}

}  // namespace motis::nigiri