#pragma once

#include "geo/latlng.h"

#include "motis/protocol/Position_generated.h"

namespace motis {

inline geo::latlng from_fbs(Position const* pos) {
  return geo::latlng{pos->lat(), pos->lng()};
}

inline Position to_fbs(geo::latlng const& p) {
  return Position{p.lat_, p.lng_};
}

}  // namespace motis