#include "motis/rt/trip_formation_handler.h"

namespace motis::rt {

void handle_trip_formation_msg(statistics& stats,
                               update_msg_builder& update_builder,
                               ris::TripFormationMessage const* msg) {
  ++stats.trip_formation_msgs_;
  update_builder.trip_formation_message(msg);
}

}  // namespace motis::rt
