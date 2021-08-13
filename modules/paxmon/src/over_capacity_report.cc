#include "motis/paxmon/over_capacity_report.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>

#include "fmt/ostream.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/access/service_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/hash_map.h"

#include "motis/paxmon/get_load.h"

namespace motis::paxmon {

std::set<std::string> get_service_names(schedule const& sched,
                                        trip const* trp) {
  std::set<std::string> names;
  for (auto const& section : motis::access::sections(trp)) {
    for (connection_info const* ci = section.fcon().con_info_; ci != nullptr;
         ci = ci->merged_with_) {
      names.insert(get_service_name(sched, ci));
    }
  }
  return names;
}

std::set<std::uint32_t> get_train_nrs(trip const* trp) {
  std::set<std::uint32_t> train_nrs;
  for (auto const& section : motis::access::sections(trp)) {
    for (connection_info const* ci = section.fcon().con_info_; ci != nullptr;
         ci = ci->merged_with_) {
      train_nrs.insert(ci->train_nr_);
    }
  }
  return train_nrs;
}

std::string_view capacity_source_str(capacity_source const src) {
  switch (src) {
    case capacity_source::TRIP_EXACT: return "trip";
    case capacity_source::TRAIN_NR: return "train_nr";
    case capacity_source::CATEGORY: return "category";
    case capacity_source::CLASZ: return "clasz";
    case capacity_source::SPECIAL: return "special";
  }
  return "???";
}

void write_over_capacity_report(paxmon_data const& data, schedule const& sched,
                                std::string const& filename) {
  std::ofstream out{filename};
  mcd::hash_map<trip const*, std::vector<edge const*>> over_capacity;
  auto const& g = data.graph_;

  for (auto const& n : g.nodes_) {
    for (auto const& e : n->outgoing_edges(g)) {
      if (!e->is_trip() || e->is_canceled(g)) {
        continue;
      }
      auto const passengers = get_base_load(e->get_pax_connection_info());
      auto const capacity = e->capacity();
      if (e->has_capacity() && passengers > capacity) {
        for (auto const& trp : e->get_trips(sched)) {
          over_capacity[trp].emplace_back(e.get());
        }
      }
    }
  }

  for (auto& [trp, edges] : over_capacity) {
    std::sort(begin(edges), end(edges), [&g](edge const* lhs, edge const* rhs) {
      return lhs->from(g)->schedule_time() < rhs->from(g)->schedule_time();
    });
    auto const& trp_start_station =
        sched.stations_.at(trp->id_.primary_.get_station_id());
    auto const& trp_target_station =
        sched.stations_.at(trp->id_.secondary_.target_station_id_);
    fmt::print(
        out,
        "Trip: train_nr={:<6} line_id={:<6}  train_nrs={:<30}  "
        "service_names={}\n"
        "From: {:16}    {:8} {:50}\nTo:   {:16}    {:8} {:50}\n{:->148}\n",
        trp->id_.primary_.train_nr_, trp->id_.secondary_.line_id_,
        fmt::join(get_train_nrs(trp), ", "),
        fmt::join(get_service_names(sched, trp), ", "),
        format_unix_time(motis_to_unixtime(sched.schedule_begin_,
                                           trp->id_.primary_.get_time())),
        trp_start_station->eva_nr_, trp_start_station->name_,
        format_unix_time(motis_to_unixtime(sched.schedule_begin_,
                                           trp->id_.secondary_.target_time_)),
        trp_target_station->eva_nr_, trp_target_station->name_, "");

    for (auto const& e : edges) {
      auto const& from_station = e->from(g)->get_station(sched);
      auto const& to_station = e->to(g)->get_station(sched);
      auto const passengers = get_base_load(e->get_pax_connection_info());
      auto const capacity = e->capacity();
      auto const additional = static_cast<int>(passengers - capacity);
      auto const percentage = static_cast<double>(passengers) /
                              static_cast<double>(capacity) * 100.0;
      fmt::print(
          out, "{:4}/{:4} [{:+4} {:3.0f}%] [{:8}] | {:8} {:50} => {:8} {:50}\n",
          passengers, capacity, additional, percentage,
          capacity_source_str(e->get_capacity_source()), from_station.eva_nr_,
          from_station.name_, to_station.eva_nr_, to_station.name_);
    }

    fmt::print(out, "\n\n");
  }
}

}  // namespace motis::paxmon
