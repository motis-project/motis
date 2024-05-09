#pragma once

#include "motis/core/conv/event_type_conv.h"
#include "motis/core/journey/extern_trip.h"
#include "motis/protocol/TripId_generated.h"

namespace motis {

inline flatbuffers::Offset<TripId> to_fbs(flatbuffers::FlatBufferBuilder& fbb,
                                          extern_trip const& t) {
  return CreateTripId(fbb, fbb.CreateString(t.id_),
                      fbb.CreateString(t.station_id_), t.train_nr_, t.time_,
                      fbb.CreateString(t.target_station_id_), t.target_time_,
                      fbb.CreateString(t.line_id_));
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
