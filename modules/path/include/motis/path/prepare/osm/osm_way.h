#pragma once

#include <string>

#include "cista/reflection/comparable.h"

#include "motis/path/prepare/osm_path.h"
#include "motis/path/prepare/source_spec.h"

namespace motis::path {

using source_bits_t = std::uint64_t;

enum class source_bits : source_bits_t {
  NO_SOURCE = 0ULL,
  RELATION = 1ULL << 0ULL,  // quite likely
  UNLIKELY = 1ULL << 1ULL,  // technically yes, but usually not
  VERY_UNLIKELY = 1ULL << 2ULL,  // technically yes, but quite certainly not

  BUS = 1ULL << 10ULL,
  RAIL = 1ULL << 11ULL,
  LIGHT_RAIL = 1ULL << 12ULL,
  SUBWAY = 1ULL << 13ULL,
  TRAM = 1ULL << 14ULL,
  SHIP = 1ULL << 15ULL,

  ONEWAY = 1ULL << 63ULL
};

inline source_bits operator|(source_bits const lhs, source_bits const rhs) {
  return static_cast<source_bits>(static_cast<source_bits_t>(lhs) |
                                  static_cast<source_bits_t>(rhs));
}

inline source_bits operator&(source_bits const lhs, source_bits const rhs) {
  return static_cast<source_bits>(static_cast<source_bits_t>(lhs) &
                                  static_cast<source_bits_t>(rhs));
}

struct osm_way {
  CISTA_COMPARABLE();

  bool is_valid() const { return !ids_.empty(); }
  void invalidate() { ids_.clear(); }

  int64_t from() const { return path_.osm_node_ids_.front(); }
  int64_t to() const { return path_.osm_node_ids_.back(); }

  mcd::vector<int64_t> ids_;
  source_bits source_bits_{source_bits::NO_SOURCE};
  osm_path path_;
};

mcd::vector<osm_way> aggregate_osm_ways(mcd::vector<osm_way> osm_ways);

}  // namespace motis::path
