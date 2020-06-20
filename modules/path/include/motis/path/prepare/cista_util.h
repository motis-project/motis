#pragma once

#include "cista/type_hash/type_hash.h"

#include "geo/latlng.h"

namespace geo {

inline cista::hash_t type_hash(latlng const& el, cista::hash_t h,
                               std::map<cista::hash_t, unsigned>& done) {
  return cista::hash_combine(cista::type_hash(el.lat_, h, done),
                             cista::type_hash(el.lng_, h, done));
}

template <typename Ctx>
inline void serialize(Ctx&, latlng const*, cista::offset_t) {}

template <typename Ctx>
inline void deserialize(Ctx const&, latlng*) {}

}  // namespace geo
