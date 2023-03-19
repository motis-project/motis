#include "motis/cc/cc.h"

#include "utl/verify.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/station_access.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/message_to_journeys.h"
#include "utl/enumerate.h"

using namespace motis::module;

namespace motis::cc {

struct interchange {
  void reset() {
    enter_ = ev_key();
    exit_ = ev_key();
  }
  size_t exit_stop_idx_{0}, enter_stop_idx_{0};
  ev_key exit_, enter_;
};

void cc::init(motis::module::registry& reg) {
  reg.register_op("/cc", [&](msg_ptr const& m) { return cc::check_journey(m); },
                  {kScheduleReadAccess});
}

ev_key get_event_at(schedule const& sched, Connection const* con,
                    std::size_t stop_idx, event_type const ev_type) {
  utl::verify(stop_idx < con->stops()->Length(), "stop not in range");
  auto const stop = con->stops()->Get(stop_idx);
  auto const station_idx =
      get_station(sched, stop->station()->id()->str())->index_;
  auto const ev_time = unix_to_motistime(
      sched, ev_type == event_type::DEP ? stop->departure()->schedule_time()
                                        : stop->arrival()->schedule_time());
  utl::verify(ev_time != INVALID_TIME, "interchange event time not valid");

  auto const trp_it = std::find_if(
      std::begin(*con->trips()), std::end(*con->trips()),
      [&ev_type, &stop_idx](Trip const* trp) {
        return (ev_type == event_type::ARR &&
                trp->range()->to() == static_cast<uint16_t>(stop_idx)) ||
               (ev_type == event_type::DEP &&
                trp->range()->from() == static_cast<uint16_t>(stop_idx));
      });
  utl::verify(trp_it != std::end(*con->trips()),
              "no trip end/start at interchange");

  trip const* trp = nullptr;

  try {
    trp = from_fbs(sched, trp_it->id());
  } catch (std::exception const& e) {
    fmt::print("trip not found: {} (dbg={})\n",
               to_extern_trip(trp_it->id()).to_str(), trp_it->debug()->c_str());
    throw;
  }

  auto const edge_it = std::find_if(
      begin(*trp->edges_), end(*trp->edges_), [&](trip::route_edge const& e) {
        auto const k = ev_key{e, trp->lcon_idx_, ev_type};
        auto const schedule_time = get_schedule_time(sched, k);
        return (k.lcon()->valid_ != 0U) &&  //
               ((ev_type == event_type::ARR &&
                 e->to_->get_station()->id_ == station_idx &&
                 schedule_time == ev_time) ||
                (ev_type == event_type::DEP &&
                 e->from_->get_station()->id_ == station_idx &&
                 schedule_time == ev_time));
      });
  utl::verify(edge_it != end(*trp->edges_),
              "important event (type={}, schedule_time={}, station=(id={}, "
              "name={})) not in trip {} (dbg={})",
              ev_type == event_type::DEP ? "DEP" : "ARR", format_time(ev_time),
              sched.stations_.at(station_idx)->name_,
              sched.stations_.at(station_idx)->eva_nr_,
              to_extern_trip(sched, trp).to_str(), trp->dbg_);

  return ev_key{*edge_it, trp->lcon_idx_, ev_type};
}

std::vector<interchange> get_interchanges(schedule const& sched,
                                          Connection const* con) {
  std::vector<interchange> interchanges;

  interchange ic;
  auto stop_idx = 0;
  for (auto const& s : *con->stops()) {
    if (s->exit()) {
      ic.exit_ = get_event_at(sched, con, stop_idx, event_type::ARR);
      ic.exit_stop_idx_ = stop_idx;
    }

    if (s->enter()) {
      ic.enter_ = get_event_at(sched, con, stop_idx, event_type::DEP);
      ic.enter_stop_idx_ = stop_idx;
      if (ic.exit_.is_not_null()) {
        interchanges.emplace_back(ic);
      }
      ic.reset();
    }

    ++stop_idx;
  }

  return interchanges;
}

using dist_t = duration;
struct label {
  label(node const* node, uint32_t dist) : node_(node), dist_(dist) {}

  friend bool operator>(label const& a, label const& b) {
    return a.dist_ > b.dist_;
  }

  node const* node_;
  dist_t dist_;
};
struct get_bucket {
  std::size_t operator()(label const& l) const { return l.dist_; }
};

duration get_shortest_footpath(schedule const& sched, node const* from_node,
                               node const* to_node) {
  dial<label, MAX_TRAVEL_TIME_MINUTES, get_bucket> pq;
  mcd::vector<dist_t> dists;
  mcd::vector<edge const*> pred_edge;
  dists.resize(sched.next_node_id_, std::numeric_limits<dist_t>::max());
  pred_edge.resize(sched.next_node_id_, nullptr);
  pq.push(label{from_node, 0});
  while (!pq.empty()) {
    auto l = pq.top();
    pq.pop();

    if (l.node_ == to_node) {
      break;
    }

    for (auto const& e : l.node_->edges_) {
      if (e.type() != edge::ROUTE_EDGE && e.type() != edge::HOTEL_EDGE) {
        auto const new_dist = l.dist_ + e.get_foot_edge_cost().time_;
        if (new_dist < dists[e.to_->id_] &&
            new_dist <= MAX_TRAVEL_TIME_MINUTES) {
          dists[e.to_->id_] = new_dist;
          pred_edge[e.to_->id_] = &e;
          pq.push(label(e.to_, new_dist));
        }
      }
    }
  }

  auto const print_edge = [&](edge const* e) {
    if (e == nullptr) {
      return;
    }
    auto const& from = *sched.stations_.at(e->from_->get_station()->id_);
    auto const& to = *sched.stations_.at(e->to_->get_station()->id_);
    std::cout << from.name_ << " (" << from.eva_nr_ << ")"
              << " --" << e->type_str() << "--" << e->get_foot_edge_cost().time_
              << "--> " << to.name_ << "(" << to.eva_nr_ << ")\n";
  };
  edge const* pred = pred_edge[to_node->id_];
  while (pred != nullptr && pred->from_ != from_node) {
    print_edge(pred);
    pred = pred_edge[pred->from_->id_];
  }
  print_edge(pred);

  std::cout << "distance: " << dists[to_node->id_] << "\n";

  return dists[to_node->id_];
}

motis::time get_foot_edge_duration(schedule const& sched, Connection const* con,
                                   std::size_t src_stop_idx) {
  utl::verify(src_stop_idx + 1 < con->stops()->Length(),
              "walk target index out of range");

  auto const from = get_station(
      sched, con->stops()->Get(src_stop_idx)->station()->id()->str());
  auto const to = get_station(
      sched, con->stops()->Get(src_stop_idx + 1)->station()->id()->str());

  auto const from_node = sched.station_nodes_.at(from->index_).get();
  auto const to_node = sched.station_nodes_.at(to->index_).get();

  utl::verify(from_node->foot_node_ != nullptr,
              "walk src node has no foot node");

  auto const shorted_footpath =
      get_shortest_footpath(sched, from_node, to_node);
  utl::verify(shorted_footpath != std::numeric_limits<dist_t>::max(),
              "foot edge {} ({}) --> {} ({}) not found, shorted_footpath={}",
              from->name_, from->eva_nr_, to->name_, to->eva_nr_,
              shorted_footpath);

  return shorted_footpath;
}

void check_interchange(schedule const& sched, Connection const* con,
                       interchange const& ic) {
  auto const transfer_time = ic.enter_.get_time() - ic.exit_.get_time();
  if (ic.exit_stop_idx_ == ic.enter_stop_idx_) {
    auto const track_transfer_time =
        sched.stations_.at(ic.enter_.get_station_idx())
            ->get_transfer_time_between_tracks(ic.exit_.get_track(),
                                               ic.enter_.get_track());
    utl::verify(transfer_time >= track_transfer_time,
                "transfer time {} below station transfer time {}");
  } else {
    auto min_transfer_time = 0;
    for (auto i = ic.exit_stop_idx_; i < ic.enter_stop_idx_; ++i) {
      min_transfer_time += get_foot_edge_duration(sched, con, i);
    }
    utl::verify(transfer_time >= min_transfer_time,
                "transfer_time={} < min_transfer_time={} (exit_stop_idx={}, "
                "enter_stop_idx={})",
                transfer_time, min_transfer_time, ic.exit_stop_idx_,
                ic.enter_stop_idx_);
  }
}

msg_ptr cc::check_journey(msg_ptr const& msg) {
  switch (msg->get()->content_type()) {
    case MsgContent_Connection: {
      auto const con = motis_content(Connection, msg);
      auto const& sched = get_sched();
      for (auto const& ic : get_interchanges(sched, con)) {
        check_interchange(sched, con, ic);
      }
      return make_success_msg();
    }

    case MsgContent_RoutingResponse: {
      using motis::routing::RoutingResponse;
      auto const res = motis_content(RoutingResponse, msg);
      auto const& sched = get_sched();
      for (auto const& [i, con] : utl::enumerate(*res->connections())) {
        try {
          for (auto const& ic : get_interchanges(sched, con)) {
            check_interchange(sched, con, ic);
          }
        } catch (std::exception const& e) {
          throw utl::fail("connection {}: {}", i, e.what());
        }
      }
      return make_success_msg();
    }

    default: throw std::system_error(error::unexpected_message_type);
  }
}

}  // namespace motis::cc
