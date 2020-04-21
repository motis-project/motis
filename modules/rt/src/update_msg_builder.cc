#include "motis/rt/update_msg_builder.h"

#include <algorithm>
#include <tuple>

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/access/edge_access.h"
#include "motis/core/access/realtime_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/conv/event_type_conv.h"
#include "motis/core/conv/timestamp_reason_conv.h"
#include "motis/core/conv/trip_conv.h"

using namespace motis::module;
using namespace flatbuffers;

namespace motis::rt {

update_msg_builder::update_msg_builder(schedule const& sched) : sched_{sched} {}

void update_msg_builder::add_delay(delay_info const* di) {
  ++delay_count_;
  auto const& k = di->get_ev_key();

  if (!k.lcon_is_valid()) {
    return;
  }

  auto const trp =
      sched_.merged_trips_[get_lcon(k.route_edge_, k.lcon_idx_).trips_]->at(0);
  delays_[trp].emplace_back(di);
}

void update_msg_builder::add_reroute(
    trip const* trp, mcd::vector<trip::route_edge> const& old_edges,
    lcon_idx_t const old_lcon_idx) {
  ++reroute_count_;
  updates_.emplace_back(CreateRtUpdate(
      fbb_, Content_RtRerouteUpdate,
      CreateRtRerouteUpdate(fbb_, to_fbs(sched_, fbb_, trp),
                            to_fbs_event_infos(old_edges, old_lcon_idx),
                            to_fbs_event_infos(*trp->edges_, trp->lcon_idx_))
          .Union()));
}

void update_msg_builder::add_free_text_nodes(
    trip const* trp, free_text const& ft, std::vector<ev_key> const& events) {
  auto const trip = to_fbs(sched_, fbb_, trp);
  auto const r = Range{0, 0};
  auto const free_text =
      CreateFreeText(fbb_, &r, ft.code_, fbb_.CreateString(ft.text_),
                     fbb_.CreateString(ft.type_));
  for (auto const& k : events) {
    updates_.emplace_back(CreateRtUpdate(
        fbb_, Content_RtFreeTextUpdate,
        CreateRtFreeTextUpdate(
            fbb_, trip,
            CreateRtEventInfo(
                fbb_,
                fbb_.CreateString(
                    sched_.stations_.at(k.get_station_idx())->eva_nr_),
                motis_to_unixtime(
                    sched_, k ? get_schedule_time(sched_, k) : INVALID_TIME),
                to_fbs(k.ev_type_)),
            free_text)
            .Union()));
  }
}

void update_msg_builder::add_track_nodes(ev_key const& k,
                                         std::string const& track,
                                         motis::time const schedule_time) {
  auto const trip =
      to_fbs(sched_, fbb_, sched_.merged_trips_[k.lcon()->trips_]->at(0));
  updates_.emplace_back(CreateRtUpdate(
      fbb_, Content_RtTrackUpdate,
      CreateRtTrackUpdate(
          fbb_, trip,
          CreateRtEventInfo(
              fbb_,
              fbb_.CreateString(
                  sched_.stations_.at(k.get_station_idx())->eva_nr_),
              motis_to_unixtime(sched_, schedule_time), to_fbs(k.ev_type_)),
          fbb_.CreateString(track))
          .Union()));
}

void update_msg_builder::reset() {
  delays_.clear();
  updates_.clear();
  fbb_.Clear();
  delay_count_ = 0;
  reroute_count_ = 0;
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<RtEventInfo>>>
update_msg_builder::to_fbs_event_infos(
    mcd::vector<trip::route_edge> const& edges, lcon_idx_t const lcon_idx) {
  std::vector<flatbuffers::Offset<RtEventInfo>> events;
  for (auto const& e : edges) {
    utl::verify(e->type() == edge::ROUTE_EDGE, "invalid trip edge");
    events.emplace_back(CreateRtEventInfo(
        fbb_,
        fbb_.CreateString(
            sched_.stations_[e->from_->get_station()->id_]->eva_nr_),
        motis_to_unixtime(
            sched_, get_schedule_time(sched_, e, lcon_idx, event_type::DEP)),
        EventType_DEP));
    events.emplace_back(CreateRtEventInfo(
        fbb_,
        fbb_.CreateString(
            sched_.stations_[e->to_->get_station()->id_]->eva_nr_),
        motis_to_unixtime(
            sched_, get_schedule_time(sched_, e, lcon_idx, event_type::ARR)),
        EventType_ARR));
  }
  return fbb_.CreateVector(events);
}

void update_msg_builder::build_delay_updates() {
  for (auto& [trp, delays] : delays_) {
    auto const fbs_trip = to_fbs(sched_, fbb_, trp);
    std::stable_sort(
        begin(delays), end(delays),
        [&](delay_info const* a, delay_info const* b) {
          auto const a_station_idx = a->get_ev_key().get_station_idx();
          auto const b_station_idx = b->get_ev_key().get_station_idx();
          auto const a_is_dep = a->get_ev_key().is_departure();
          auto const b_is_dep = b->get_ev_key().is_departure();
          return std::tie(a->schedule_time_, a_station_idx, a_is_dep) <
                 std::tie(b->schedule_time_, b_station_idx, b_is_dep);
        });
    updates_.emplace_back(CreateRtUpdate(
        fbb_, Content_RtDelayUpdate,
        CreateRtDelayUpdate(
            fbb_, fbs_trip,
            fbb_.CreateVector(utl::to_vec(
                delays,
                [&](delay_info const* di) {
                  auto const& k = di->get_ev_key();
                  return CreateUpdatedRtEventInfo(
                      fbb_,
                      CreateRtEventInfo(
                          fbb_,
                          fbb_.CreateString(
                              sched_.stations_.at(k.get_station_idx())
                                  ->eva_nr_),
                          motis_to_unixtime(sched_, di->get_schedule_time()),
                          to_fbs(k.ev_type_)),
                      motis_to_unixtime(sched_, di->get_current_time()),
                      to_fbs(di->get_reason()));
                })))
            .Union()));
  }
  delays_.clear();
}

msg_ptr update_msg_builder::finish() {
  build_delay_updates();
  fbb_.create_and_finish(
      MsgContent_RtUpdates,
      CreateRtUpdates(fbb_, fbb_.CreateVector(updates_)).Union(), "/rt/update",
      DestinationType_Topic);
  return make_msg(fbb_);
}

}  // namespace motis::rt
