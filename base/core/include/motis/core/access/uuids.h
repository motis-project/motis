#pragma once

#include <optional>

#include "boost/uuid/uuid.hpp"

#include "motis/core/schedule/schedule.h"

namespace motis::access {

inline std::optional<boost::uuids::uuid> get_event_uuid(schedule const& sched,
                                                        trip const* trp,
                                                        ev_key const evk) {
  if (auto const it =
          sched.event_to_uuid_.find(mcd::pair{ptr<trip const>{trp}, evk});
      it != end(sched.event_to_uuid_)) {
    return {it->second};
  } else {
    return {};
  }
}

}  // namespace motis::access
