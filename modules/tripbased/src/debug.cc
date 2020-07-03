#include <algorithm>
#include <utility>

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/access/service_access.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_stop.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/tripbased/debug.h"
#include "motis/tripbased/error.h"

using namespace flatbuffers;

namespace motis::tripbased {

trip_id find_trip(tb_data const& data, schedule const& sched,
                  TripSelectorWrapper const* selector) {
  switch (selector->selector_type()) {
    case TripSelector_TripId: {
      auto const id = reinterpret_cast<TripId const*>(selector->selector());
      auto const motis_trip = from_fbs(sched, id);
      auto const expanded_trip = std::find_if(
          begin(sched.expanded_trips_.data_), end(sched.expanded_trips_.data_),
          [&](trip const* t) { return t->id_ == motis_trip->id_; });
      if (expanded_trip == end(sched.expanded_trips_.data_)) {
        throw std::system_error(error::trip_not_found);
      }
      return static_cast<trip_id>(
          std::distance(begin(sched.expanded_trips_.data_), expanded_trip));
    }
    case TripSelector_ExpandedTripId: {
      auto const index =
          reinterpret_cast<ExpandedTripId const*>(selector->selector())
              ->index();
      if (index < data.trip_count_) {
        return index;
      } else {
        throw std::system_error(error::trip_not_found);
      }
    }
    default: throw std::system_error(error::trip_not_found);
  }
}

Offset<TripBasedTripId> fbs_tb_trip_id(FlatBufferBuilder& fbb,
                                       schedule const& sched,
                                       trip_id const trp) {
  auto const t = sched.expanded_trips_.data_.at(trp);
  return CreateTripBasedTripId(fbb, trp, to_fbs(sched, fbb, t));
}

Offset<LineDebugInfo> get_line_debug_info(FlatBufferBuilder& fbb,
                                          tb_data const& data,
                                          line_id const line) {
  utl::verify(line < data.line_count_, "get_line_debug_info: invalid line id");
  return CreateLineDebugInfo(fbb, line, data.line_to_first_trip_[line],
                             data.line_to_last_trip_[line],
                             data.line_stop_count_[line]);
}

Offset<Station> fbs_station(FlatBufferBuilder& fbb, schedule const& sched,
                            station_id const id) {
  auto const station = sched.stations_[id].get();
  Position const station_pos{station->lat(), station->lng()};
  return CreateStation(fbb, fbb.CreateString(station->eva_nr_),
                       fbb.CreateString(station->name_), &station_pos);
}

std::pair<Offset<Vector<Offset<TransportDebugInfo>>>,
          Offset<Vector<Offset<TransportDebugInfo>>>>
get_transport_debug_infos(FlatBufferBuilder& fbb, schedule const& sched,
                          trip_id const trp, stop_idx_t const stop_idx) {
  std::vector<Offset<TransportDebugInfo>> arrival_transports,
      departure_transports;
  access::trip_stop stop{sched.expanded_trips_.data_[trp], stop_idx};
  auto const add_transports = [&](std::vector<Offset<TransportDebugInfo>>&
                                      transports,
                                  light_connection const& lcon) {
    auto con_info = lcon.full_con_->con_info_;
    while (con_info != nullptr) {
      auto const& cat_name = sched.categories_[con_info->family_]->name_;
      auto const clasz_it = sched.classes_.find(cat_name);
      auto const clasz = clasz_it == end(sched.classes_) ? service_class::OTHER
                                                         : clasz_it->second;
      transports.push_back(CreateTransportDebugInfo(
          fbb, fbb.CreateString(cat_name), con_info->family_,
          static_cast<service_class_t>(clasz),
          output_train_nr(con_info->train_nr_, con_info->original_train_nr_),
          con_info->original_train_nr_,
          fbb.CreateString(con_info->line_identifier_),
          fbb.CreateString(get_service_name(sched, con_info)),
          fbb.CreateString(con_info->provider_ != nullptr
                               ? con_info->provider_->full_name_
                               : ""),
          fbb.CreateString(con_info->dir_ != nullptr ? *con_info->dir_ : "")));
      con_info = con_info->merged_with_;
    }
  };
  if (stop.has_arrival()) {
    add_transports(arrival_transports, stop.arr_lcon());
  }
  if (stop.has_departure()) {
    add_transports(departure_transports, stop.dep_lcon());
  }
  return {fbb.CreateVector(arrival_transports),
          fbb.CreateVector(departure_transports)};
}

Offset<Vector<Offset<TransferDebugInfo>>> get_transfer_debug_info(
    FlatBufferBuilder& fbb, tb_data const& data, schedule const& sched,
    trip_id const trp, stop_idx_t const stop_idx) {
  auto const from_trip = fbs_tb_trip_id(fbb, sched, trp);
  auto const from_station = fbs_station(
      fbb, sched, data.stops_on_line_[data.trip_to_line_[trp]][stop_idx]);
  return fbb.CreateVector(utl::to_vec(
      data.transfers_.at(trp, stop_idx), [&](tb_transfer const& transfer) {
        return CreateTransferDebugInfo(
            fbb, from_trip, fbs_tb_trip_id(fbb, sched, transfer.to_trip_),
            stop_idx, transfer.to_stop_idx_,
            static_cast<uint64_t>(
                motis_to_unixtime(sched, data.arrival_times_[trp][stop_idx])),
            static_cast<uint64_t>(motis_to_unixtime(
                sched, data.departure_times_[transfer.to_trip_]
                                            [transfer.to_stop_idx_])),
            from_station,
            fbs_station(
                fbb, sched,
                data.stops_on_line_[data.trip_to_line_[transfer.to_trip_]]
                                   [transfer.to_stop_idx_]));
      }));
}

Offset<Vector<Offset<TransferDebugInfo>>> get_reverse_transfer_debug_info(
    FlatBufferBuilder& fbb, tb_data const& data, schedule const& sched,
    trip_id const trp, stop_idx_t const stop_idx) {
  auto const to_trip = fbs_tb_trip_id(fbb, sched, trp);
  auto const to_station = fbs_station(
      fbb, sched, data.stops_on_line_[data.trip_to_line_[trp]][stop_idx]);
  return fbb.CreateVector(utl::to_vec(
      data.reverse_transfers_.at(trp, stop_idx),
      [&](tb_reverse_transfer const& transfer) {
        return CreateTransferDebugInfo(
            fbb, fbs_tb_trip_id(fbb, sched, transfer.from_trip_), to_trip,
            transfer.from_stop_idx_, stop_idx,
            static_cast<uint64_t>(motis_to_unixtime(
                sched, data.arrival_times_[transfer.from_trip_]
                                          [transfer.from_stop_idx_])),
            static_cast<uint64_t>(
                motis_to_unixtime(sched, data.departure_times_[trp][stop_idx])),
            fbs_station(
                fbb, sched,
                data.stops_on_line_[data.trip_to_line_[transfer.from_trip_]]
                                   [transfer.from_stop_idx_]),
            to_station);
      }));
}

Offset<Vector<Offset<FootpathDebugInfo>>> get_footpath_debug_info(
    FlatBufferBuilder& fbb, tb_data const& data, schedule const& sched,
    station_id const station) {
  auto const from = fbs_station(fbb, sched, station);
  return fbb.CreateVector(
      utl::to_vec(data.footpaths_[station], [&](tb_footpath const& fp) {
        return CreateFootpathDebugInfo(
            fbb, from, fbs_station(fbb, sched, fp.to_stop_), fp.duration_);
      }));
}

Offset<Vector<Offset<FootpathDebugInfo>>> get_reverse_footpath_debug_info(
    FlatBufferBuilder& fbb, tb_data const& data, schedule const& sched,
    station_id const station) {
  auto const to = fbs_station(fbb, sched, station);
  return fbb.CreateVector(
      utl::to_vec(data.reverse_footpaths_[station], [&](tb_footpath const& fp) {
        return CreateFootpathDebugInfo(
            fbb, fbs_station(fbb, sched, fp.from_stop_), to, fp.duration_);
      }));
}

Offset<Vector<Offset<StopDebugInfo>>> get_trip_stop_debug_info(
    FlatBufferBuilder& fbb, tb_data const& data, schedule const& sched,
    trip_id const trp, line_id const line) {
  auto const ts = [&](time const t) {
    return static_cast<uint64_t>(
        t == INVALID_TIME ? 0 : motis_to_unixtime(sched, t));
  };
  auto const stop_count = data.line_stop_count_[line];
  std::vector<Offset<StopDebugInfo>> sdis;
  sdis.reserve(stop_count);
  auto const& stops = data.stops_on_line_[line];
  auto const& arrivals = data.arrival_times_[trp];
  auto const& departures = data.departure_times_[trp];
  auto const& in_allowed = data.in_allowed_[line];
  auto const& out_allowed = data.out_allowed_[line];
  for (stop_idx_t stop_idx = 0U; stop_idx < stop_count; ++stop_idx) {
    auto const station = stops[stop_idx];
    auto const [arrival_transports, departure_transports] =  // NOLINT
        get_transport_debug_infos(fbb, sched, trp, stop_idx);
    sdis.push_back(CreateStopDebugInfo(
        fbb, fbs_station(fbb, sched, station), ts(arrivals[stop_idx]),
        ts(departures[stop_idx]), in_allowed[stop_idx] != 0,
        out_allowed[stop_idx] != 0,
        get_transfer_debug_info(fbb, data, sched, trp, stop_idx),
        get_reverse_transfer_debug_info(fbb, data, sched, trp, stop_idx),
        fbb.CreateVector(utl::to_vec(
            data.lines_at_stop_[station],
            [&](line_stop const& ls) {
              return CreateLineAtStopDebugInfo(
                  fbb, get_line_debug_info(fbb, data, ls.line_), ls.stop_idx_);
            })),
        get_footpath_debug_info(fbb, data, sched, station),
        get_reverse_footpath_debug_info(fbb, data, sched, station),
        sched.stations_[station]->transfer_time_, arrival_transports,
        departure_transports));
  }
  return fbb.CreateVector(sdis);
}

Offset<TripDebugInfo> get_trip_debug_info(FlatBufferBuilder& fbb,
                                          tb_data const& data,
                                          schedule const& sched,
                                          TripSelectorWrapper const* selector) {
  auto const trp = find_trip(data, sched, selector);
  utl::verify(trp < data.trip_count_, "get_trip_debug_info: invalid trip id");
  auto const line = data.trip_to_line_[trp];
  return CreateTripDebugInfo(
      fbb, fbs_tb_trip_id(fbb, sched, trp),
      get_line_debug_info(fbb, data, line),
      get_trip_stop_debug_info(fbb, data, sched, trp, line));
}

Offset<Vector<Offset<TripAtStopDebugInfo>>> get_trips_at_stop_debug_info(
    FlatBufferBuilder& fbb, tb_data const& data, schedule const& sched,
    station_id const station) {
  auto const ts = [&](time const t) {
    return static_cast<uint64_t>(
        t == INVALID_TIME ? 0 : motis_to_unixtime(sched, t));
  };
  std::vector<Offset<TripAtStopDebugInfo>> trips;
  for (auto const& ls : data.lines_at_stop_[station]) {
    for (auto trp = data.line_to_first_trip_[ls.line_];
         trp <= data.line_to_last_trip_[ls.line_]; ++trp) {
      auto const [arrival_transports, departure_transports] =  // NOLINT
          get_transport_debug_infos(fbb, sched, trp, ls.stop_idx_);
      trips.push_back(CreateTripAtStopDebugInfo(
          fbb, fbs_tb_trip_id(fbb, sched, trp), ls.stop_idx_,
          ts(data.arrival_times_[trp][ls.stop_idx_]),
          ts(data.departure_times_[trp][ls.stop_idx_]),
          data.in_allowed_[ls.line_][ls.stop_idx_] != 0,
          data.out_allowed_[ls.line_][ls.stop_idx_] != 0, arrival_transports,
          departure_transports));
    }
  }
  return fbb.CreateVector(trips);
}

Offset<StationDebugInfo> get_station_debug_info(FlatBufferBuilder& fbb,
                                                tb_data const& data,
                                                schedule const& sched,
                                                std::string const& eva_no) {
  auto const station = get_station(sched, eva_no);
  return CreateStationDebugInfo(
      fbb, fbs_station(fbb, sched, station->index_),
      fbb.CreateVector(utl::to_vec(
          data.lines_at_stop_[station->index_],
          [&](line_stop const& ls) {
            return CreateLineAtStopDebugInfo(
                fbb, get_line_debug_info(fbb, data, ls.line_), ls.stop_idx_);
          })),
      get_footpath_debug_info(fbb, data, sched, station->index_),
      get_reverse_footpath_debug_info(fbb, data, sched, station->index_),
      station->transfer_time_,
      get_trips_at_stop_debug_info(fbb, data, sched, station->index_));
}

}  // namespace motis::tripbased
