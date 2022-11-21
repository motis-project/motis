#include "motis/paxmon/debug.h"

#include "fmt/core.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/trip_access.h"

#include "motis/core/debug/trip.h"

namespace motis::paxmon {

void print_trip(schedule const& sched, trip const* trp) {
  fmt::print(
      "    trip: {:7} / {:10} / {:5} / {:7} / {:10} / {} [ptr={}, debug={}]\n",
      sched.stations_.at(trp->id_.primary_.get_station_id())->eva_nr_.view(),
      format_time(trp->id_.primary_.get_time()), trp->id_.primary_.train_nr_,
      sched.stations_.at(trp->id_.secondary_.target_station_id_)
          ->eva_nr_.view(),
      format_time(trp->id_.secondary_.target_time_),
      trp->id_.secondary_.line_id_, static_cast<void const*>(trp), trp->dbg_);
}

void print_trip(schedule const& sched, trip_idx_t const idx) {
  return print_trip(sched, get_trip(sched, idx));
}

void print_leg(schedule const& sched, journey_leg const& leg) {
  auto const& enter_station = sched.stations_.at(leg.enter_station_id_);
  auto const& exit_station = sched.stations_.at(leg.exit_station_id_);
  fmt::print("  {:7} {:7} {:50} -> {:7} {:7} {:50}",
             format_time(leg.enter_time_), enter_station->eva_nr_.str(),
             enter_station->name_.str(), format_time(leg.exit_time_),
             exit_station->eva_nr_.str(), exit_station->name_.str());
  if (leg.enter_transfer_) {
    fmt::print("  enter_transfer={:2} {}\n", leg.enter_transfer_->duration_,
               leg.enter_transfer_->type_);
  } else {
    fmt::print("\n");
  }
  print_trip(sched, leg.trip_idx_);
}

void print_trip_section(schedule const& sched,
                        motis::access::trip_section const& ts) {
  auto const cur_dep_time = ts.ev_key_from().get_time();
  auto const sched_dep_time = get_schedule_time(sched, ts.ev_key_from());
  auto const& from_station = ts.from_station(sched);
  auto const cur_arr_time = ts.ev_key_to().get_time();
  auto const sched_arr_time = get_schedule_time(sched, ts.ev_key_to());
  auto const& to_station = ts.to_station(sched);
  auto const& merged_trips = sched.merged_trips_.at(ts.lcon().trips_);
  fmt::print(
      "  {:7} [{:7}] {:7} {:50} -> {:7} [{:7}] {:7} {:50} [lc={}, "
      "merged_trips={}]",
      format_time(cur_dep_time), format_time(sched_dep_time),
      from_station.eva_nr_.str(), from_station.name_.str(),
      format_time(cur_arr_time), format_time(sched_arr_time),
      to_station.eva_nr_.str(), to_station.name_.str(),
      static_cast<void const*>(&ts.lcon()), ts.lcon().trips_);
  if (merged_trips->size() > 1) {
    fmt::print(" [{} trips]\n", merged_trips->size());
  } else {
    fmt::print("\n");
  }
}

void print_trip_edge(schedule const& sched, universe const& uv, edge const* e) {
  auto const cur_dep_time = e->from(uv)->current_time();
  auto const cur_sched_time = e->from(uv)->schedule_time();
  auto const& from_station = e->from(uv)->get_station(sched);
  auto const cur_arr_time = e->to(uv)->current_time();
  auto const sched_arr_time = e->to(uv)->schedule_time();
  auto const& to_station = e->to(uv)->get_station(sched);
  auto const& merged_trips = sched.merged_trips_.at(e->get_merged_trips_idx());
  fmt::print("  {:7} [{:7}] {:7} {:50} -> {:7} [{:7}] {:7} {:50}",
             format_time(cur_dep_time), format_time(cur_sched_time),
             from_station.eva_nr_.str(), from_station.name_.str(),
             format_time(cur_arr_time), format_time(sched_arr_time),
             to_station.eva_nr_.str(), to_station.name_.str());
  if (merged_trips->size() > 1) {
    fmt::print(" [{} trips]\n", merged_trips->size());
  } else {
    fmt::print("\n");
  }
}

void print_trip_sections(universe const& uv, schedule const& sched,
                         trip const* trp, trip_data_index const tdi) {
  std::cout << "paxmon trip:\n";
  if (tdi != INVALID_TRIP_DATA_INDEX) {
    for (auto const e : uv.trip_data_.edges(tdi)) {
      print_trip_edge(sched, uv, e.get(uv));
    }
  } else {
    std::cout << "  not found\n";
  }
  std::cout << "motis trip:" << std::endl;
  for (auto const& sec : motis::access::sections(trp)) {
    print_trip_section(sched, sec);
  }
  std::cout << "trip debug:\n";
  std::cout << debug::trip_with_sections{sched, trp};
}

void print_trip_sections(universe const& uv, schedule const& sched,
                         trip_idx_t const idx, trip_data_index const tdi) {
  return print_trip_sections(uv, sched, get_trip(sched, idx), tdi);
}

}  // namespace motis::paxmon
