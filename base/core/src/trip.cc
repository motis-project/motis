#include "motis/core/schedule/trip.h"

namespace motis {

time concrete_trip::get_first_dep_time() const {
  return {day_idx_, trp_->id_.primary_.first_departure_mam()};
}

time concrete_trip::get_last_arr_time() const {
  return {static_cast<day_idx_t>(day_idx_ + trp_->day_offsets_.back()),
          trp_->id_.secondary_.last_arrival_mam_};
}

generic_light_connection concrete_trip::lcon(size_t const index) const {
  auto const i = index - 1;
  auto const& e = *trp_->edges_->at(i).get_edge();
  auto const lcon_idx = trp_->lcon_idx_;

  switch (e.type()) {
    case edge_type::RT_ROUTE_EDGE: return &e.rt_lcons().at(lcon_idx);
    case edge_type::STATIC_ROUTE_EDGE:
      return std::pair{
          &e.static_lcons().at(lcon_idx),
          static_cast<day_idx_t>(day_idx_ + trp_->day_offsets_.at(i))};
    default: throw utl::fail("get_lcon on {}", e.type_str());
  }
}

}  // namespace motis
