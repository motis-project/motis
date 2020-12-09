#include "motis/paxmon/output/mcfp_scenario.h"

#include <cstdint>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "utl/verify.h"

#include "motis/core/conv/trip_conv.h"
#include "motis/hash_map.h"

#include "motis/paxmon/messages.h"
#include "motis/paxmon/trip_section_load_iterator.h"

namespace fs = boost::filesystem;
using namespace motis::module;

namespace motis::paxmon::output {

void write_stations(fs::path const& dir, schedule const& sched) {
  std::ofstream out{(dir / "stations.csv").string()};
  out.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  out << "id,transfer,name\n";
  for (auto const& st : sched.stations_) {
    out << st->eva_nr_.view() << "," << st->transfer_time_ << ","
        << std::quoted(st->name_.view(), '"', '"') << "\n";
  }
}

void write_footpaths(fs::path const& dir, schedule const& sched) {
  std::ofstream out{(dir / "footpaths.csv").string()};
  out.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  out << "from_station,to_station,duration\n";
  for (auto const& st : sched.stations_) {
    for (auto const& fp : st->outgoing_footpaths_) {
      out << st->eva_nr_.view() << ","
          << sched.stations_.at(fp.to_station_)->eva_nr_.view() << ","
          << fp.duration_ << "\n";
    }
  }
}

void write_trip(std::ofstream& out, schedule const& sched,
                paxmon_data const& data, trip const* trp, std::uint64_t id,
                bool const include_trip_info) {
  for (auto const& ts : sections_with_load{sched, data, trp}) {
    auto const& lc = ts.section_.lcon();
    auto const remaining_capacity =
        ts.has_capacity_info() ? std::max(0, ts.capacity() - ts.base_load())
                               : 0;
    out << id << "," << ts.section_.from_station(sched).eva_nr_ << ","
        << lc.d_time_ << "," << ts.section_.to_station(sched).eva_nr_ << ","
        << lc.a_time_ << "," << remaining_capacity;
    if (include_trip_info) {
      out << ","
          << sched.categories_.at(lc.full_con_->con_info_->family_)
                 ->name_.view()
          << "," << lc.full_con_->con_info_->train_nr_;
    }
    out << "\n";
  }
}

void write_trips(fs::path const& dir, schedule const& sched,
                 paxmon_data const& data,
                 mcd::hash_map<trip const*, std::uint64_t>& trip_ids,
                 bool const include_trip_info) {
  std::ofstream trips_file{(dir / "trips.csv").string()};
  trips_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  trips_file << "id,from_station,departure,to_station,arrival,capacity";
  if (include_trip_info) {
    trips_file << ",category,train_nr";
  }
  trips_file << "\n";

  std::ofstream trip_ids_file{(dir / "trip_ids.csv").string()};
  trip_ids_file.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  trip_ids_file << "id,station,train_nr,time,target_station,target_time,line\n";

  auto id = 1ULL;
  for (auto const& trp : sched.trip_mem_) {
    if (trp->edges_->empty()) {
      continue;
    }
    write_trip(trips_file, sched, data, trp.get(), id, include_trip_info);
    trip_ids[trp.get()] = id;
    auto const ext_trp = to_extern_trip(sched, trp.get());
    trip_ids_file << id << "," << ext_trp.station_id_ << ","
                  << ext_trp.train_nr_ << "," << ext_trp.time_ << ","
                  << ext_trp.target_station_id_ << "," << ext_trp.target_time_
                  << "," << std::quoted(ext_trp.line_id_.view(), '"', '"')
                  << "\n";
    ++id;
  }
}

void write_groups(fs::path const& dir, schedule const& sched,
                  paxmon_data const& data, std::vector<msg_ptr> const& messages,
                  mcd::hash_map<trip const*, std::uint64_t> const& trip_ids) {
  std::ofstream out{(dir / "groups.csv").string()};
  out.exceptions(std::ios_base::failbit | std::ios_base::badbit);
  out << "id,start,departure,destination,arrival,passengers,in_trip\n";
  auto id = 1ULL;
  for (auto const& msg : messages) {
    auto const update = motis_content(MonitoringUpdate, msg);
    for (auto const& event : *update->events()) {
      if (event->type() == MonitoringEventType_NO_PROBLEM) {
        continue;
      }
      auto const loc =
          from_fbs(sched, event->localization_type(), event->localization());
      auto const pg = data.get_passenger_group(event->group()->id());
      utl::verify(pg != nullptr, "mcfp_scenario: invalid group");
      out << id << "," << loc.at_station_->eva_nr_.view() << ","
          << loc.current_arrival_time_ << ","
          << sched.stations_
                 .at(pg->compact_planned_journey_.destination_station_id())
                 ->eva_nr_.view()
          << "," << pg->planned_arrival_time_ << "," << pg->passengers_ << ",";
      if (loc.in_trip()) {
        out << trip_ids.at(loc.in_trip_);
      }
      out << "\n";
      ++id;
    }
  }
}

void write_scenario(fs::path const& dir, schedule const& sched,
                    paxmon_data const& data,
                    std::vector<msg_ptr> const& messages,
                    bool const include_trip_info) {
  mcd::hash_map<trip const*, std::uint64_t> trip_ids;
  write_stations(dir, sched);
  write_footpaths(dir, sched);
  write_trips(dir, sched, data, trip_ids, include_trip_info);
  write_groups(dir, sched, data, messages, trip_ids);
}

}  // namespace motis::paxmon::output
