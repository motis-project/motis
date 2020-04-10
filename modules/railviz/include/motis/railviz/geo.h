#pragma once

#include "motis/protocol/PathBoxesResponse_generated.h"
#include "motis/protocol/Position_generated.h"

namespace motis::railviz::geo {

struct coord {
  double lat_, lng_;
};

using box = std::pair<coord, coord>;

inline coord from_fbs(Position const* p) { return {p->lat(), p->lng()}; }

inline box from_fbs(path::Box const* b) {
  return {from_fbs(b->north_east()), from_fbs(b->south_west())};
}

}  // namespace motis::railviz::geo
