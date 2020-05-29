#pragma once

#include <memory>
#include <set>

#include "flatbuffers/flatbuffers.h"

#ifdef GetMessage
#undef GetMessage
#endif

#include "motis/protocol/RISMessage_generated.h"

#include "motis/core/common/typed_flatbuffer.h"

namespace motis::ris {

struct ris_message : typed_flatbuffer<Message> {
  ris_message(time_t earliest, time_t latest, time_t timestamp,
              flatbuffers::FlatBufferBuilder&& fbb)
      : typed_flatbuffer(std::move(fbb)),
        earliest_(earliest),
        latest_(latest),
        timestamp_(timestamp) {}

  // testing w/o flatbuffers
  ris_message(time_t earliest, time_t latest, time_t timestamp,
              std::string const& msg)
      : typed_flatbuffer(msg),
        earliest_(earliest),
        latest_(latest),
        timestamp_(timestamp) {}

  time_t earliest_;
  time_t latest_;
  time_t timestamp_;
};

}  // namespace motis::ris
