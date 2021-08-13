#pragma once

#include <cstdint>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

#include "motis/core/schedule/event_type.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"

#include "motis/module/message.h"

#include "motis/paxmon/paxmon_data.h"

namespace motis::paxmon {

struct trip_ev_key {
  std::uint32_t station_id_{};
  time schedule_time_{};
  event_type type_{event_type::DEP};
  event_node* node_{nullptr};
};

inline std::ostream& operator<<(std::ostream& o, trip_ev_key const& tek) {
  return o << "{station=" << tek.station_id_
           << ", time=" << format_time(tek.schedule_time_)
           << ", type=" << tek.type_ << ", node=" << tek.node_ << "}";
}

inline bool operator==(trip_ev_key const& lhs, trip_ev_key const& rhs) {
  return std::tie(lhs.station_id_, lhs.schedule_time_, lhs.type_) ==
         std::tie(rhs.station_id_, rhs.schedule_time_, rhs.type_);
}

enum class diff_op : std::uint8_t { KEEP, REMOVE, INSERT };

inline std::ostream& operator<<(std::ostream& o, diff_op const op) {
  switch (op) {
    case diff_op::KEEP: return o << "=";
    case diff_op::REMOVE: return o << "-";
    case diff_op::INSERT: return o << "+";
  }
  return o;
}

std::vector<trip_ev_key> to_trip_ev_keys(trip_data const& td, graph const& g);

std::vector<trip_ev_key> to_trip_ev_keys(
    schedule const& sched,
    flatbuffers::Vector<flatbuffers::Offset<motis::rt::RtEventInfo>> const&
        events);

void apply_reroute(paxmon_data& data, schedule const& sched, trip const* trp,
                   trip_data& td, std::vector<trip_ev_key> const& old_route,
                   std::vector<trip_ev_key> const& new_route,
                   std::vector<edge*>& updated_interchange_edges);

}  // namespace motis::paxmon
