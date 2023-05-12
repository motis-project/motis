#pragma once

#include <memory>
#include <set>

#include "flatbuffers/flatbuffers.h"

#include "motis/core/common/typed_flatbuffer.h"
#include "motis/core/common/unixtime.h"

#include "motis/protocol/RISMessage_generated.h"

namespace motis::ris {

struct ris_message : typed_flatbuffer<RISMessage> {
  ris_message(unixtime earliest, unixtime latest, unixtime timestamp,
              flatbuffers::FlatBufferBuilder&& fbb)
      : typed_flatbuffer(std::move(fbb)),
        earliest_(earliest),
        latest_(latest),
        timestamp_(timestamp) {}

  // testing w/o flatbuffers
  ris_message(unixtime earliest, unixtime latest, unixtime timestamp,
              std::string const& msg)
      : typed_flatbuffer(msg),
        earliest_(earliest),
        latest_(latest),
        timestamp_(timestamp) {}

  unixtime earliest_;
  unixtime latest_;
  unixtime timestamp_;
};

}  // namespace motis::ris
