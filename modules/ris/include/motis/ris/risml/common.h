#pragma once

#include <ctime>
#include <limits>

#include "pugixml.hpp"

#include "flatbuffers/flatbuffers.h"

namespace motis::ris::risml {

struct context {
  explicit context(time_t timestamp)
      : timestamp_{timestamp},
        earliest_{std::numeric_limits<time_t>::max()},
        latest_{std::numeric_limits<time_t>::min()} {}

  flatbuffers::FlatBufferBuilder b_;
  time_t timestamp_, earliest_, latest_;
};

pugi::xml_attribute inline child_attr(pugi::xml_node const& n, char const* e,
                                      char const* a) {
  return n.child(e).attribute(a);
}

}  // namespace motis::ris::risml