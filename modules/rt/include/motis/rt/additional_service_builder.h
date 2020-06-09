#pragma once

#include <algorithm>
#include <iterator>
#include <set>
#include <string>
#include <vector>

#include "utl/get_or_create.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/conv/event_type_conv.h"
#include "motis/loader/classes.h"

#include "motis/rt/build_route_node.h"
#include "motis/rt/connection_builder.h"
#include "motis/rt/incoming_edges.h"
#include "motis/rt/update_constant_graph.h"
#include "motis/rt/update_msg_builder.h"

#include "motis/protocol/RISMessage_generated.h"

namespace motis::rt {

struct additional_service_builder {
  using section = std::tuple<light_connection, station_node*, station_node*>;

  enum class status {
    OK,
    TRIP_ID_MISMATCH,
    EVENT_COUNT_MISMATCH,
    EVENT_ORDER_MISMATCH,
    STATION_NOT_FOUND,
    EVENT_TIME_OUT_OF_RANGE,
    STATION_MISMATCH,
    DECREASING_TIME
  };

  additional_service_builder(schedule& sched,
                             update_msg_builder& update_builder)
      : sched_{sched}, update_builder_{update_builder} {}

  status check_events(
      flatbuffers::Vector<flatbuffers::Offset<ris::AdditionalEvent>> const*
          events) const {
    if (events->size() == 0 || events->size() % 2 != 0) {
      return status::EVENT_COUNT_MISMATCH;
    }

    station const* arr_station = nullptr;
    uint64_t prev_time = 0;
    event_type next = event_type::DEP;
    for (auto const& ev : *events) {
      auto ev_type = from_fbs(ev->base()->type());
      if (ev_type != next) {
        return status::EVENT_ORDER_MISMATCH;
      }

      if (unix_to_motistime(sched_, ev->base()->schedule_time()) ==
          INVALID_TIME) {
        return status::EVENT_TIME_OUT_OF_RANGE;
      }

      auto station = find_station(sched_, ev->base()->station_id()->str());
      if (station == nullptr) {
        return status::STATION_NOT_FOUND;
      }

      if (prev_time > ev->base()->schedule_time()) {
        return status::DECREASING_TIME;
      }

      if (ev_type == event_type::DEP && arr_station != nullptr &&
          arr_station != station) {
        return status::STATION_MISMATCH;
      }

      prev_time = ev->base()->schedule_time();
      next = (next == event_type::DEP) ? event_type::ARR : event_type::DEP;
      arr_station = station;
    }

    return status::OK;
  }

  std::vector<section> build_sections(
      flatbuffers::Vector<flatbuffers::Offset<ris::AdditionalEvent>> const*
          events) {
    std::vector<section> sections;
    for (auto it = std::begin(*events); it != std::end(*events);) {
      light_connection lcon{};
      lcon.valid_ = 1U;

      // DEP
      auto dep_station =
          get_station_node(sched_, it->base()->station_id()->str());
      auto dep_track = it->track()->str();
      lcon.d_time_ = unix_to_motistime(sched_, it->base()->schedule_time());
      ++it;

      // ARR
      auto arr_station =
          get_station_node(sched_, it->base()->station_id()->str());
      lcon.a_time_ = unix_to_motistime(sched_, it->base()->schedule_time());
      lcon.full_con_ =
          get_full_con(sched_, con_infos_, dep_track, it->track()->str(),
                       it->category()->str(), it->base()->line_id()->str(),
                       it->base()->service_num());
      ++it;

      sections.emplace_back(lcon, dep_station, arr_station);
    }
    return sections;
  }

  static std::set<station_node*> get_station_nodes(
      std::vector<section> const& sections) {
    std::set<station_node*> station_nodes;
    for (auto& c : sections) {
      station_nodes.insert(std::get<1>(c));
      station_nodes.insert(std::get<2>(c));
    }
    return station_nodes;
  }

  mcd::vector<trip::route_edge> build_route(
      std::vector<section> const& sections,
      std::vector<incoming_edge_patch>& incoming) {
    auto const route_id = sched_.route_count_++;

    mcd::vector<trip::route_edge> trip_edges;
    node* prev_route_node = nullptr;
    for (auto const& s : sections) {
      light_connection l{};
      station_node *from_station = nullptr, *to_station = nullptr;
      std::tie(l, from_station, to_station) = s;

      auto const from_station_transfer_time =
          sched_.stations_.at(from_station->id_)->transfer_time_;
      auto const to_station_transfer_time =
          sched_.stations_.at(to_station->id_)->transfer_time_;

      auto const from_route_node =
          prev_route_node != nullptr
              ? prev_route_node
              : build_route_node(sched_, route_id, sched_.node_count_++,
                                 from_station, from_station_transfer_time, true,
                                 true, incoming);
      auto const to_route_node =
          build_route_node(sched_, route_id, sched_.node_count_++, to_station,
                           to_station_transfer_time, true, true, incoming);

      from_route_node->edges_.push_back(
          make_route_edge(from_route_node, to_route_node, {l}));

      auto const route_edge = &from_route_node->edges_.back();
      add_outgoing_edge(route_edge, incoming);
      trip_edges.emplace_back(route_edge);
      constant_graph_add_route_edge(sched_, route_edge);

      prev_route_node = to_route_node;
    }

    return trip_edges;
  }

  trip const* update_trips(mcd::vector<trip::route_edge> const& trip_edges) {
    auto const first_edge = trip_edges.front().get_edge();
    auto const first_station = first_edge->from_->get_station();
    auto const first_lcon = first_edge->m_.route_edge_.conns_[0];

    auto const last_edge = trip_edges.back().get_edge();
    auto const last_station = last_edge->to_->get_station();
    auto const last_lcon = last_edge->m_.route_edge_.conns_[0];

    sched_.trip_edges_.emplace_back(
        mcd::make_unique<mcd::vector<trip::route_edge>>(trip_edges));
    sched_.trip_mem_.emplace_back(mcd::make_unique<trip>(
        full_trip_id{primary_trip_id{first_station->id_,
                                     first_lcon.full_con_->con_info_->train_nr_,
                                     first_lcon.d_time_},
                     secondary_trip_id{
                         last_station->id_, last_lcon.a_time_,
                         first_lcon.full_con_->con_info_->line_identifier_}},
        sched_.trip_edges_.back().get(), 0U, trip_debug{}));

    auto const trp = sched_.trip_mem_.back().get();
    auto const trp_entry = mcd::pair{trp->id_.primary_, ptr<trip>(trp)};
    sched_.trips_.insert(
        std::lower_bound(begin(sched_.trips_), end(sched_.trips_), trp_entry),
        trp_entry);

    auto const new_trps_id = sched_.merged_trips_.size();
    sched_.merged_trips_.emplace_back(
        mcd::make_unique<mcd::vector<ptr<trip>>,
                         std::initializer_list<ptr<trip>>>({trp}));

    for (auto const& trp_edge : trip_edges) {
      trp_edge.get_edge()->m_.route_edge_.conns_[0].trips_ = new_trps_id;
    }

    return trp;
  }

  status verify_trip_id(trip const* trp, ris::IdEvent const* id_ev) const {
    auto const id_station = find_station(sched_, id_ev->station_id()->str());
    auto const id_event_time =
        unix_to_motistime(sched_, id_ev->schedule_time());
    auto const trp_id = trp->id_.primary_;
    if (id_station != nullptr && id_event_time != INVALID_TIME &&
        id_station->index_ == trp_id.station_id_ &&
        id_ev->service_num() == trp_id.train_nr_ &&
        id_event_time == trp_id.time_) {
      return status::OK;
    } else {
      return status::TRIP_ID_MISMATCH;
    }
  }

  status build_additional_train(ris::AdditionMessage const* msg) {
    auto const result = check_events(msg->events());
    if (result != status::OK) {
      return result;
    }

    auto const sections = build_sections(msg->events());
    auto const station_nodes = get_station_nodes(sections);
    std::vector<incoming_edge_patch> incoming;
    save_outgoing_edges(station_nodes, incoming);
    auto const route = build_route(sections, incoming);
    patch_incoming_edges(incoming);
    auto const trp = update_trips(route);

    update_builder_.add_reroute(trp, {}, 0);

    return verify_trip_id(trp, msg->trip_id());
  }

  schedule& sched_;
  update_msg_builder& update_builder_;
  std::map<connection_info, connection_info const*> con_infos_;
};

}  // namespace motis::rt
