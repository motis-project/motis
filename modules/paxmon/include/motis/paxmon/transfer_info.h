#pragma once

#include <iosfwd>

#include "cista/hashing.h"
#include "cista/reflection/comparable.h"

#include "motis/core/schedule/time.h"

namespace motis::paxmon {

struct transfer_info {
  CISTA_COMPARABLE()

  cista::hash_t hash() const { return cista::build_hash(duration_, type_); }

  bool requires_transfer() const {
    return type_ == type::SAME_STATION || type_ == type::FOOTPATH;
  }

  enum class type { SAME_STATION, FOOTPATH, MERGE, THROUGH };
  duration duration_{};
  type type_{type::SAME_STATION};
};

inline std::ostream& operator<<(std::ostream& out,
                                transfer_info::type const t) {
  switch (t) {
    case transfer_info::type::SAME_STATION: return out << "SAME_STATION";
    case transfer_info::type::FOOTPATH: return out << "FOOTPATH";
    case transfer_info::type::MERGE: return out << "MERGE";
    case transfer_info::type::THROUGH: return out << "THROUGH";
  }
  return out;
}

inline duration get_transfer_duration(std::optional<transfer_info> const& ti) {
  return ti.has_value() ? ti.value().duration_ : 0;
}

}  // namespace motis::paxmon
