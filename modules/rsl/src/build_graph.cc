#include "motis/rsl/build_graph.h"

#include <iostream>

#include "utl/verify.h"

#include "motis/core/common/logging.h"

#include "motis/rsl/graph_access.h"

using namespace motis::logging;

namespace motis::rsl {

std::uint64_t initial_over_capacity{0};

namespace {

void add_interchange(event_node* from, event_node* to, passenger_group* grp,
                     duration transfer_time) {
  for (auto& e : from->out_edges_) {  // TODO(pablo): performance
    if (e->type_ == edge_type::INTERCHANGE && e->to_ == to &&
        e->transfer_time_ == transfer_time) {
      e->rsl_connection_info_.section_infos_.emplace_back(grp);
      e->passengers_ += grp->passengers_;
      grp->edges_.emplace_back(e.get());
      return;
    }
  }
  auto const new_edge = add_edge(edge{from,
                                      to,
                                      edge_type::INTERCHANGE,
                                      nullptr,
                                      transfer_time,
                                      0,
                                      grp->passengers_,
                                      false,
                                      {{pax_section_info{grp}}}});
  grp->edges_.emplace_back(new_edge);
}

inline duration get_transfer_duration(std::optional<transfer_info> const& ti) {
  return ti.has_value() ? ti.value().duration_ : 0;
}

void add_passenger_group_to_graph(schedule const& sched, rsl_data& data,
                                  passenger_group& grp) {
  event_node* exit_node = nullptr;
  trip_data* last_trip = nullptr;

  for (auto const& leg : grp.compact_planned_journey_.legs_) {
    utl::verify(leg.enter_time_ != INVALID_TIME, "invalid enter time");
    utl::verify(leg.exit_time_ != INVALID_TIME, "invalid exit time");

    trip_data* te = nullptr;
    try {
      te = get_or_add_trip(sched, data, leg.trip_);
    } catch (std::system_error const& e) {
      std::cerr << "  passenger group " << grp.id_ << "." << grp.sub_id_
                << std::endl;
      throw e;
    }
    auto in_trip = false;
    last_trip = nullptr;
    for (auto e : te->edges_) {
      if (!in_trip) {
        if (e->from_->station_ == leg.enter_station_id_ &&
            e->from_->schedule_time_ == leg.enter_time_) {
          in_trip = true;
          if (exit_node == nullptr) {
            exit_node = &te->enter_exit_node_;
          }
          auto const transfer_time = get_transfer_duration(leg.enter_transfer_);
          add_interchange(exit_node, e->from_, &grp, transfer_time);
        }
      }
      if (in_trip) {
        e->passengers_ += grp.passengers_;
        e->rsl_connection_info_.section_infos_.emplace_back(&grp);
        if (e->passengers_ > e->capacity_) {
          ++initial_over_capacity;
          /*
          std::cout << "edge over capacity: " << e->passengers_ << "/"
                    << e->capacity_ << ": "
                    << e->rsl_connection_info_.section_infos_.size()
                    << " passenger groups, trip: " << e->trip_->dbg_.str()
                    << std::endl;
                    */
        }
        grp.edges_.emplace_back(e);
        if (e->to_->station_ == leg.exit_station_id_ &&
            e->to_->schedule_time_ == leg.exit_time_) {
          exit_node = e->to_;
          last_trip = te;
          break;
        }
      }
    }
  }

  if (exit_node != nullptr && last_trip != nullptr) {
    add_interchange(exit_node, &last_trip->enter_exit_node_, &grp, 0);
  }
}

};  // namespace

void build_graph_from_journeys(schedule const& sched, rsl_data& data) {
  auto errors = 0ULL;
  for (auto& pg : data.graph_.passenger_groups_) {
    try {
      add_passenger_group_to_graph(sched, data, *pg);
    } catch (std::system_error const& e) {
      LOG(motis::logging::error)
          << "could not add passenger group: " << e.what();
      ++errors;
      // throw e;
    }
  }
  if (errors != 0) {
    LOG(motis::logging::error)
        << "could not add " << errors << " passenger groups";
  }
}

}  // namespace motis::rsl
