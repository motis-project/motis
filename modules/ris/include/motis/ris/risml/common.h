#pragma once

#include <ctime>
#include <limits>

#include "pugixml.hpp"

#include "flatbuffers/flatbuffers.h"

#include "motis/core/common/typed_flatbuffer.h"
#include "motis/core/common/unixtime.h"

namespace motis::ris::risml {

struct context {
  explicit context(unixtime timestamp)
      : timestamp_{timestamp},
        earliest_{std::numeric_limits<unixtime>::max()},
        latest_{std::numeric_limits<unixtime>::min()} {}

  flatbuffers::FlatBufferBuilder b_;
  unixtime timestamp_, earliest_, latest_;
};

pugi::xml_attribute inline child_attr(pugi::xml_node const& n, char const* e,
                                      char const* a) {
  return n.child(e).attribute(a);
}

}  // namespace motis::ris::risml
