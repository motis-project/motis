#include "motis/core/access/station_access.h"

#include "motis/core/access/error.h"

namespace motis {

station* find_station(schedule const& sched, std::string_view eva_nr) {
  auto it = sched.eva_to_station_.find(eva_nr);
  if (it == end(sched.eva_to_station_)) {
    return nullptr;
  }
  return it->second;
}

station* get_station(schedule const& sched, std::string_view eva_nr) {
  auto it = sched.eva_to_station_.find(eva_nr);
  if (it == end(sched.eva_to_station_)) {
    throw std::system_error(access::error::station_not_found);
  }
  return it->second;
}

station_node* get_station_node(schedule const& sched, std::string_view eva_nr) {
  auto index = get_station(sched, eva_nr)->index_;
  return sched.station_nodes_[index].get();
}

station_node* find_station_node(schedule const& sched,
                                std::string_view eva_nr) {
  auto const s = find_station(sched, eva_nr);
  return s == nullptr ? nullptr : sched.station_nodes_.at(s->index_).get();
}

}  // namespace motis
