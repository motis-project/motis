#pragma once

#include "motis/core/schedule/station.h"
#include "motis/protocol/Station_generated.h"

namespace motis {

inline flatbuffers::Offset<Station> to_fbs(flatbuffers::FlatBufferBuilder& fbb,
                                           station const& s) {
  auto const pos = Position{s.lat(), s.lng()};
  return CreateStation(fbb, fbb.CreateString(s.eva_nr_),
                       fbb.CreateString(s.name_), &pos);
}

}  // namespace motis
