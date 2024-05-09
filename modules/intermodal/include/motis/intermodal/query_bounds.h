#pragma once

#include <ctime>

#include "flatbuffers/flatbuffers.h"

#include "geo/latlng.h"

#include "motis/core/common/constants.h"
#include "motis/core/common/unixtime.h"

#include "motis/protocol/Message_generated.h"

namespace motis::intermodal {

struct query_start {
  query_start(routing::Start const start_type,
              flatbuffers::Offset<void> const start, geo::latlng pos,
              unixtime time, bool intermodal)
      : start_type_(start_type),
        start_(start),
        is_intermodal_(intermodal),
        pos_(pos),
        time_(time) {}

  routing::Start start_type_;
  flatbuffers::Offset<void> start_;

  bool is_intermodal_;
  geo::latlng pos_;
  unixtime time_;
};

query_start parse_query_start(flatbuffers::FlatBufferBuilder&,
                              IntermodalRoutingRequest const*);

struct query_dest {
  query_dest(flatbuffers::Offset<routing::InputStation> const station,
             geo::latlng pos, bool intermodal)
      : station_(station), is_intermodal_(intermodal), pos_(pos) {}

  flatbuffers::Offset<routing::InputStation> station_;

  bool is_intermodal_;
  geo::latlng pos_;
};

query_dest parse_query_dest(flatbuffers::FlatBufferBuilder&,
                            IntermodalRoutingRequest const*);

}  // namespace motis::intermodal
