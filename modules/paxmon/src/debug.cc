#include "motis/paxmon/debug.h"

#include "fmt/core.h"

#include "motis/core/access/realtime_access.h"

namespace motis::paxmon {

void print_trip(trip const* trp) {
  fmt::print("    trip: {:7} / {:10} / {:5} / {:7} / {:10} / {}\n",
             trp->id_.primary_.get_station_id(), trp->id_.primary_.get_time(),
             trp->id_.primary_.train_nr_,
             trp->id_.secondary_.target_station_id_,
             trp->id_.secondary_.target_time_, trp->id_.secondary_.line_id_);
}

void print_leg(schedule const& sched, journey_leg const& leg) {
  auto const enter_station_name =
      sched.stations_.at(leg.enter_station_id_)->name_.str();
  auto const exit_station_name =
      sched.stations_.at(leg.exit_station_id_)->name_.str();
  fmt::print("  {:7} {:50} -> {:7} {:50}", format_time(leg.enter_time_),
             enter_station_name, format_time(leg.exit_time_),
             exit_station_name);
  if (leg.enter_transfer_) {
    fmt::print("  enter_transfer={:2} {}\n", leg.enter_transfer_->duration_,
               leg.enter_transfer_->type_ == transfer_info::type::SAME_STATION
                   ? "SAME_STATION"
                   : "FOOTPATH");
  } else {
    fmt::print("\n");
  }
  print_trip(leg.trip_);
}

void print_trip_section(schedule const& sched,
                        motis::access::trip_section const& ts) {
  auto const cur_dep_time = ts.ev_key_from().get_time();
  auto const sched_dep_time = get_schedule_time(sched, ts.ev_key_from());
  auto const from_station_name = ts.from_station(sched).name_.str();
  auto const cur_arr_time = ts.ev_key_to().get_time();
  auto const sched_arr_time = get_schedule_time(sched, ts.ev_key_to());
  auto const to_station_name = ts.to_station(sched).name_.str();
  auto const& merged_trips = sched.merged_trips_.at(ts.lcon().trips_);
  fmt::print("  {:7} [{:7}] {:50} -> {:7} [{:7}] {:50}",
             format_time(cur_dep_time), format_time(sched_dep_time),
             from_station_name, format_time(cur_arr_time),
             format_time(sched_arr_time), to_station_name);
  if (merged_trips->size() > 1) {
    fmt::print(" [{} trips]\n", merged_trips->size());
  } else {
    fmt::print("\n");
  }
}

void print_trip_edge(schedule const& sched, graph const& g, edge const* e) {
  auto const cur_dep_time = e->from(g)->current_time();
  auto const cur_sched_time = e->from(g)->schedule_time();
  auto const from_station_name = e->from(g)->get_station(sched).name_.str();
  auto const cur_arr_time = e->to(g)->current_time();
  auto const sched_arr_time = e->to(g)->schedule_time();
  auto const to_station_name = e->to(g)->get_station(sched).name_.str();
  auto const& merged_trips = sched.merged_trips_.at(e->get_merged_trips_idx());
  fmt::print("  {:7} [{:7}] {:50} -> {:7} [{:7}] {:50}",
             format_time(cur_dep_time), format_time(cur_sched_time),
             from_station_name, format_time(cur_arr_time),
             format_time(sched_arr_time), to_station_name);
  if (merged_trips->size() > 1) {
    fmt::print(" [{} trips]\n", merged_trips->size());
  } else {
    fmt::print("\n");
  }
}

void print_trip_sections(graph const& g, schedule const& sched, trip const* trp,
                         trip_data const* td) {
  std::cout << "paxmon trip:\n";
  for (auto const e : td->edges_) {
    print_trip_edge(sched, g, e);
  }
  std::cout << "motis trip:" << std::endl;
  for (auto const& sec : motis::access::sections(trp)) {
    print_trip_section(sched, sec);
  }
}

}  // namespace motis::paxmon
