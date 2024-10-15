#include "motis/cc/cc.h"

#include "utl/verify.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/station_access.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/message_to_journeys.h"

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
  reg.register_op("/cc",
                  [&](msg_ptr const& m) { return cc::check_journey(m); });
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
  auto const trp = from_fbs(sched, trp_it->id());

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
  utl::verify(edge_it != end(*trp->edges_), "important event not in trip");

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
  auto const& foot_edges = from_node->foot_node_->edges_;
  auto const fe_it = std::find_if(
      begin(foot_edges), end(foot_edges), [&to_node](edge const& e) {
        return e.type() == edge::FWD_EDGE && e.to_ == to_node;
      });
  utl::verify(fe_it != end(foot_edges), "foot edge not found");

  return fe_it->m_.foot_edge_.time_cost_;
}

void check_interchange(schedule const& sched, Connection const* con,
                       interchange const& ic) {
  auto const transfer_time = ic.enter_.get_time() - ic.exit_.get_time();
  if (ic.exit_stop_idx_ == ic.enter_stop_idx_) {
    utl::verify(
        transfer_time >= sched.stations_.at(ic.enter_.get_station_idx())
                             ->get_transfer_time_between_tracks(
                                 ic.exit_.get_track(), ic.enter_.get_track()),
        "transfer time below station transfer time");
  } else {
    auto min_transfer_time = 0;
    for (auto i = ic.exit_stop_idx_; i < ic.enter_stop_idx_; ++i) {
      min_transfer_time += get_foot_edge_duration(sched, con, i);
    }
    utl::verify(transfer_time >= min_transfer_time,
                "transfer time below walk duration");
  }
}

msg_ptr cc::check_journey(msg_ptr const& msg) {
  auto const con = motis_content(Connection, msg);
  auto const& sched = get_sched();
  for (auto const& ic : get_interchanges(sched, con)) {
    check_interchange(sched, con, ic);
  }
  return make_success_msg();
}

}  // namespace motis::cc
