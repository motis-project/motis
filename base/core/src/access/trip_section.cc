#include "motis/core/access/trip_section.h"

#include "motis/core/access/connection_access.h"
#include "motis/core/access/edge_access.h"

namespace motis::access {

station const& get_station(schedule const& sched, node const* n) {
  return *sched.stations_[n->get_station()->id_];
}

trip_section::trip_section(concrete_trip const t, int const index)
    : ctrp_(t), index_(index), edge_(t.trp_->edges_->at(index).get_edge()) {}

int trip_section::index() const { return index_; }

light_connection const& trip_section::lcon() const {
  return get_lcon(edge_, ctrp_.trp_->lcon_idx_);
}

time trip_section::arr_time() const {
  return get_lcon(edge_, ctrp_.trp_->lcon_idx_)
      .event_time(event_type::ARR, ctrp_.day_idx_);
}

time trip_section::dep_time() const {
  return get_lcon(edge_, ctrp_.trp_->lcon_idx_)
      .event_time(event_type::ARR, ctrp_.day_idx_);
}

connection const& trip_section::fcon() const { return *lcon().full_con_; }

connection_info const& trip_section::info(schedule const& sched) const {
  return get_connection_info(sched, lcon(), ctrp_.trp_);
}

class edge const* trip_section::edge() const {
  return edge_;
}

station const& trip_section::from_station(schedule const& sched) const {
  return get_station(sched, edge_->from_);
}

station const& trip_section::to_station(schedule const& sched) const {
  return get_station(sched, edge_->to_);
}

uint32_t trip_section::from_station_id() const {
  return edge_->from_->get_station()->id_;
}

uint32_t trip_section::to_station_id() const {
  return edge_->to_->get_station()->id_;
}

ev_key trip_section::ev_key_from() const {
  return ev_key{ctrp_.trp_->edges_->at(index_), ctrp_.trp_->lcon_idx_,
                event_type::DEP};
}

ev_key trip_section::ev_key_to() const {
  return ev_key{ctrp_.trp_->edges_->at(index_), ctrp_.trp_->lcon_idx_,
                event_type::ARR};
}

edge const* trip_section::get_route_edge() const { return edge_; }

node* trip_section::from_node() const { return edge_->from_; }

node* trip_section::to_node() const { return edge_->to_; }

}  // namespace motis::access
