#include "motis/paxforecast/output/output.h"

#include <cstdint>

#include "rapidjson/filewritestream.h"
#include "rapidjson/writer.h"

#include "motis/core/access/service_access.h"
#include "motis/core/access/trip_access.h"

using namespace rapidjson;
using namespace motis::paxmon;

namespace motis::paxforecast::output {

struct log_output::writer {
  explicit writer(char const* filename)
      : file_(std::fopen(filename, "wb")),  // NOLINT
        stream_(file_, buffer_, sizeof(buffer_)),
        writer_(stream_) {}

  std::FILE* file_;
  char buffer_[65536]{};
  FileWriteStream stream_;
  Writer<FileWriteStream> writer_;
};

log_output::log_output(std::string const& filename)
    : writer_(std::make_unique<log_output::writer>(filename.c_str())) {}

log_output::~log_output() = default;

template <typename Writer>
void write_str(Writer& w, mcd::string const& str) {
  w.String(str.data(), str.size());
}

template <typename Writer>
void write_str(Writer& w, std::string const& str) {
  w.String(str.data(), str.size());
}

template <typename Writer>
void write_time(Writer& w, std::time_t t) {
  w.Uint64(static_cast<std::uint64_t>(t));
}

template <typename Writer>
void write_time(Writer& w, schedule const& sched, time t) {
  auto const unix_ts = t != INVALID_TIME
                           ? static_cast<std::uint64_t>(
                                 motis_to_unixtime(sched.schedule_begin_, t))
                           : static_cast<std::uint64_t>(0ULL);
  w.Uint64(unix_ts);
}

template <typename Writer>
void write_station(Writer& w, station const* s) {
  w.StartObject();

  w.Key("index");
  w.Uint(s->index_);

  w.Key("eva");
  write_str(w, s->eva_nr_);

  w.Key("name");
  write_str(w, s->name_);

  w.EndObject();
}

template <typename Writer>
void write_station(Writer& w, schedule const& sched, unsigned station_id) {
  write_station(w, sched.stations_[station_id].get());
}

template <typename Writer>
void write_trip_id(Writer& w, schedule const& sched, full_trip_id const& id) {
  w.StartObject();

  w.Key("station");
  write_station(w, sched, static_cast<unsigned>(id.primary_.station_id_));

  w.Key("time");
  write_time(w, sched, id.primary_.time_);

  w.Key("train_nr");
  w.Uint(id.primary_.get_train_nr());

  w.Key("target_station");
  write_station(w, sched, id.secondary_.target_station_id_);

  w.Key("target_time");
  write_time(w, sched, id.secondary_.target_time_);

  w.Key("line");
  write_str(w, id.secondary_.line_id_);

  w.EndObject();
}

template <typename Writer>
void write_trip_debug(Writer& w, trip_debug const& debug) {
  w.StartObject();

  w.Key("file");
  write_str(w, *debug.file_);

  w.Key("line_from");
  w.Int(debug.line_from_);

  w.Key("line_to");
  w.Int(debug.line_to_);

  w.EndObject();
}

template <typename Writer>
void write_trip(Writer& w, schedule const& sched, trip const* trp) {
  w.StartObject();

  w.Key("id");
  write_trip_id(w, sched, trp->id_);

  if (trp->dbg_.file_ != nullptr) {
    w.Key("source");
    write_trip_debug(w, trp->dbg_);
  }

  if (trp->edges_ != nullptr && !trp->edges_->empty()) {
    auto const ci = trp->edges_->front()
                        ->m_.route_edge_.conns_.at(trp->lcon_idx_)
                        .full_con_->con_info_;
    w.Key("name");
    write_str(w, get_service_name(sched, ci));

    w.Key("category");
    write_str(w, sched.categories_[ci->family_]->name_);

    w.Key("train_nr");
    w.Uint(ci->train_nr_);
  }

  w.EndObject();
}

template <typename Writer>
void write_external_trip(Writer& w, schedule const& sched,
                         extern_trip const& et) {
  auto const trp = get_trip(sched, et);
  write_trip(w, sched, trp);
}

template <typename Writer>
void write_localization(Writer& w, schedule const& sched,
                        passenger_localization const& loc) {
  w.StartObject();

  w.Key("in_trip");
  w.Bool(loc.in_trip());

  w.Key("arrival_time");
  write_time(w, sched, loc.arrival_time_);

  if (loc.at_station_ != nullptr) {
    w.Key("station");
    write_station(w, loc.at_station_);
  }

  if (loc.in_trip_ != nullptr) {
    w.Key("trip");
    write_trip(w, sched, loc.in_trip_);
  }

  w.EndObject();
}

template <typename Writer>
void write_transfer_info(Writer& w, transfer_info const& ti) {
  w.StartObject();

  w.Key("duration");
  w.Uint(ti.duration_);

  w.Key("type");
  if (ti.type_ == transfer_info::type::FOOTPATH) {
    w.String("footpath");
  } else {
    w.String("same_station");
  }

  w.EndObject();
}

template <typename Writer>
void write_compact_journey(Writer& w, schedule const& sched,
                           compact_journey const& j) {
  w.StartObject();

  w.Key("legs");
  w.StartArray();
  for (auto const& leg : j.legs_) {
    w.StartObject();

    w.Key("trip");
    write_external_trip(w, sched, leg.trip_);

    w.Key("enter_station");
    write_station(w, sched, leg.enter_station_id_);

    w.Key("exit_station");
    write_station(w, sched, leg.exit_station_id_);

    w.Key("enter_time");
    write_time(w, sched, leg.enter_time_);

    w.Key("exit_time");
    write_time(w, sched, leg.exit_time_);

    if (leg.enter_transfer_) {
      w.Key("enter_transfer");
      write_transfer_info(w, leg.enter_transfer_.value());
    }

    w.EndObject();
  }
  w.EndArray();

  w.EndObject();
}

template <typename Writer>
void write_passenger_group(Writer& w, schedule const& sched,
                           passenger_group const& grp,
                           passenger_localization const& localization) {
  w.StartObject();

  w.Key("passengers");
  w.Uint(grp.passengers_);

  w.Key("planned_journey");
  write_compact_journey(w, sched, grp.compact_planned_journey_);

  w.Key("status");
  w.String(localization.arrival_time_ <=
                   grp.compact_planned_journey_.legs_.front().enter_time_
               ? "before_start"
               : "en_route");

  w.EndObject();
}

template <typename Writer>
void write_alternative(Writer& w, schedule const& sched, alternative const& alt,
                       passenger_localization const& /*localization*/) {
  w.StartObject();

  w.Key("arrival_time");
  write_time(w, sched, alt.arrival_time_);

  w.Key("transfers");
  w.Uint(alt.transfers_);

  w.Key("duration");
  w.Uint(alt.duration_);

  w.Key("journey");
  write_compact_journey(w, sched, alt.compact_journey_);

  w.EndObject();
}

template <typename Writer>
void write_cpg(Writer& w, schedule const& sched,
               combined_passenger_group const& cpg) {
  w.StartObject();

  w.Key("passengers");
  w.Uint(cpg.passengers_);

  w.Key("destination_station");
  write_station(w, sched, cpg.destination_station_id_);

  w.Key("localization");
  write_localization(w, sched, cpg.localization_);

  w.Key("groups");
  w.StartArray();
  for (auto const& grp : cpg.groups_) {
    write_passenger_group(w, sched, *grp, cpg.localization_);
  }
  w.EndArray();

  w.Key("alternatives");
  w.StartArray();
  for (auto const& alt : cpg.alternatives_) {
    write_alternative(w, sched, alt, cpg.localization_);
  }
  w.EndArray();

  w.EndObject();
}

template <typename Writer>
void write_simulation_result(Writer& w, schedule const& sched,
                             simulation_result const& sim_result,
                             graph const& g) {
  w.StartObject();

  w.Key("over_capacity");
  w.Bool(sim_result.is_over_capacity());

  w.Key("edge_count_over_capacity");
  w.Uint64(sim_result.edge_count_over_capacity());

  w.Key("total_passengers_over_capacity");
  w.Uint64(sim_result.total_passengers_over_capacity());

  w.Key("trips_over_capacity");
  w.StartArray();
  for (auto const& [trp, edges] : sim_result.trips_over_capacity_with_edges()) {
    w.StartObject();

    w.Key("trip");
    write_trip(w, sched, trp);

    w.Key("edges");
    w.StartArray();
    for (auto const e : edges) {
      if (e->type_ != edge_type::TRIP) {
        continue;
      }
      w.StartObject();

      w.Key("passengers");
      w.Uint(e->passengers());

      w.Key("capacity");
      w.Uint(e->capacity());

      w.Key("additional");
      w.Uint(sim_result.additional_passengers_.at(e));

      w.Key("from");
      write_station(w, sched, e->from(g)->station_);

      w.Key("to");
      write_station(w, sched, e->to(g)->station_);

      w.EndObject();
    }
    w.EndArray();

    w.EndObject();
  }
  w.EndArray();

  w.EndObject();
}

void log_output::write_broken_connection(
    schedule const& sched, paxmon_data const& data,
    std::map<unsigned, std::vector<combined_passenger_group>> const&
        combined_groups,
    simulation_result const& sim_result) {
  auto& w = writer_->writer_;

  w.StartObject();

  w.Key("event");
  w.String("broken_connections");

  w.Key("system_time");
  write_time(w, sched.system_time_);

  w.Key("combined_groups");
  w.StartArray();
  for (auto const& [destination_station_id, cpgs] : combined_groups) {
    (void)destination_station_id;
    for (auto const& cpg : cpgs) {
      write_cpg(w, sched, cpg);
    }
  }
  w.EndArray();

  w.Key("sim_result");
  write_simulation_result(w, sched, sim_result, data.graph_);

  w.EndObject();

  writer_->stream_.Put('\n');
}

void log_output::flush() {
  writer_->stream_.Flush();
  std::fflush(writer_->file_);
}

}  // namespace motis::paxforecast::output
