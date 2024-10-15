#include "motis/core/access/trip_stop.h"

#include "motis/core/access/connection_access.h"
#include "motis/core/access/edge_access.h"

namespace motis::access {

trip_stop::trip_stop(trip const* t, int const index) : trip_(t), index_(index) {
  assert(!trip_->edges_->empty());
  if (index == static_cast<int>(trip_->edges_->size())) {
    node_ = trip_->edges_->back().get_edge()->to_;
  } else {
    node_ = trip_->edges_->at(index).get_edge()->from_;
  }
  assert(node_->is_route_node());
}

int trip_stop::index() const { return index_; }

bool trip_stop::has_arrival() const { return index_ > 0; }
bool trip_stop::has_departure() const {
  return index_ < static_cast<int>(trip_->edges_->size());
}

light_connection const& trip_stop::arr_lcon() const {
  return get_lcon(trip_->edges_->at(index_ - 1).get_edge(), trip_->lcon_idx_);
}

light_connection const& trip_stop::dep_lcon() const {
  return get_lcon(trip_->edges_->at(index_).get_edge(), trip_->lcon_idx_);
}

connection_info const& trip_stop::arr_info(schedule const& sched) const {
  return get_connection_info(sched, arr_lcon(), trip_);
}

connection_info const& trip_stop::dep_info(schedule const& sched) const {
  return get_connection_info(sched, dep_lcon(), trip_);
}

station const& trip_stop::get_station(schedule const& sched) const {
  return *sched.stations_[node_->get_station()->id_];
}

uint32_t trip_stop::get_station_id() const { return node_->get_station()->id_; }

node const* trip_stop::get_route_node() const { return node_; }

bool trip_stop::is_first() const { return index_ == 0; }
bool trip_stop::is_last() const { return index_ == trip_->edges_->size(); }

}  // namespace motis::access
