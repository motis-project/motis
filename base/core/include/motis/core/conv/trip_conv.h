#pragma once

#include "motis/core/access/time_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/conv/event_type_conv.h"
#include "motis/core/journey/extern_trip.h"
#include "motis/protocol/TripId_generated.h"

namespace motis {

inline trip const* from_fbs(schedule const& sched, TripId const* t,
                            bool const fuzzy = false) {
  return get_trip(sched, t->station_id()->str(), t->train_nr(), t->time(),
                  t->target_station_id()->str(), t->target_time(),
                  t->line_id()->str(), fuzzy);
}

inline flatbuffers::Offset<TripId> to_fbs(schedule const& sched,
                                          flatbuffers::FlatBufferBuilder& fbb,
                                          trip const* trp) {
  auto const& p = trp->id_.primary_;
  auto const& s = trp->id_.secondary_;
  return CreateTripId(
      fbb, fbb.CreateString(trp->gtfs_trip_id_.str()),
      fbb.CreateString(sched.stations_.at(p.station_id_)->eva_nr_), p.train_nr_,
      motis_to_unixtime(sched, p.time_),
      fbb.CreateString(sched.stations_.at(s.target_station_id_)->eva_nr_),
      motis_to_unixtime(sched, s.target_time_), fbb.CreateString(s.line_id_));
}

inline flatbuffers::Offset<TripId> to_fbs(flatbuffers::FlatBufferBuilder& fbb,
                                          extern_trip const& t) {
  return CreateTripId(fbb, fbb.CreateString(t.id_),
                      fbb.CreateString(t.station_id_), t.train_nr_, t.time_,
                      fbb.CreateString(t.target_station_id_), t.target_time_,
                      fbb.CreateString(t.line_id_));
}

inline trip const* from_extern_trip(schedule const& sched,
                                    extern_trip const* t) {
  return get_trip(sched, t->station_id_, t->train_nr_, t->time_,
                  t->target_station_id_, t->target_time_, t->line_id_);
}

inline extern_trip to_extern_trip(schedule const& sched, trip const* t) {
  return extern_trip{
      t->gtfs_trip_id_,
      sched.stations_.at(t->id_.primary_.station_id_)->eva_nr_,
      t->id_.primary_.get_train_nr(),
      motis_to_unixtime(sched, t->id_.primary_.time_),
      sched.stations_.at(t->id_.secondary_.target_station_id_)->eva_nr_,
      motis_to_unixtime(sched, t->id_.secondary_.target_time_),
      t->id_.secondary_.line_id_};
}

inline extern_trip to_extern_trip(TripId const* trp) {
  return extern_trip{trp->id() == nullptr ? "" : trp->id()->str(),
                     trp->station_id()->str(),
                     trp->train_nr(),
                     trp->time(),
                     trp->target_station_id()->str(),
                     trp->target_time(),
                     trp->line_id()->str()};
}

}  // namespace motis
