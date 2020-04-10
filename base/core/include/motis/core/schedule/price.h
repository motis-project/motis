#pragma once

#include "motis/core/schedule/connection.h"
#include "motis/core/schedule/station.h"

#ifndef M_PI
#define M_PI (3.14159265359)
#endif

#define d2r (M_PI / 180.0)

namespace motis {

inline int get_price_per_km(int clasz) {
  switch (clasz) {
    case MOTIS_ICE: return 22;

    case MOTIS_N:
    case MOTIS_IC:
    case MOTIS_X: return 18;

    case MOTIS_RE:
    case MOTIS_RB:
    case MOTIS_S:
    case MOTIS_U:
    case MOTIS_STR:
    case MOTIS_BUS: return 15;

    default: return 0;
  }
}

inline double get_distance(const station& s1, const station& s2) {
  double lat1 = s1.width_, long1 = s1.length_;
  double lat2 = s2.width_, long2 = s2.length_;
  double dlong = (long2 - long1) * d2r;
  double dlat = (lat2 - lat1) * d2r;
  double a = pow(sin(dlat / 2.0), 2) +
             cos(lat1 * d2r) * cos(lat2 * d2r) * pow(sin(dlong / 2.0), 2);
  double c = 2 * atan2(sqrt(a), sqrt(1 - a));
  double d = 6367 * c;
  return d;
}

}  // namespace motis
