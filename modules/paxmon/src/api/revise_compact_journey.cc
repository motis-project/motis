#include "motis/paxmon/api/revise_compact_journey.h"

#include "utl/to_vec.h"

#include "motis/core/access/time_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/journeys_to_message.h"

#include "motis/module/context/motis_call.h"

#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace motis::paxmon;
using namespace motis::revise;
using namespace flatbuffers;

namespace motis::paxmon::api {

journey compact_journey_to_basic_journey(schedule const& sched,
                                         compact_journey const& cj) {
  auto j = journey{};
  auto last_time = unixtime{};

  auto const push_station = [&](station const& st) -> journey::stop& {
    if (j.stops_.empty() || j.stops_.back().eva_no_ != st.eva_nr_) {
      return j.stops_.emplace_back(journey::stop{.exit_ = false,
                                                 .enter_ = false,
                                                 .name_ = st.name_.str(),
                                                 .eva_no_ = st.eva_nr_.str(),
                                                 .lat_ = st.lat(),
                                                 .lng_ = st.lng()});
    } else {
      return j.stops_.back();
    }
  };

  for (auto const& leg : cj.legs()) {
    auto const& enter_station = *sched.stations_.at(leg.enter_station_id_);
    auto const& exit_station = *sched.stations_.at(leg.exit_station_id_);

    if (!j.stops_.empty() && j.stops_.back().eva_no_ != enter_station.eva_nr_) {
      if (leg.enter_transfer_ &&
          leg.enter_transfer_->type_ == transfer_info::type::FOOTPATH) {
        auto const prev_idx = static_cast<unsigned>(j.stops_.size() - 1);
        j.transports_.emplace_back(
            journey::transport{.from_ = prev_idx,
                               .to_ = prev_idx + 1,  // will be added below
                               .is_walk_ = true,
                               .duration_ = leg.enter_transfer_->duration_,
                               .mumo_id_ = -1});
      }
    }

    auto& enter_stop = push_station(enter_station);
    auto const enter_idx = static_cast<unsigned>(j.stops_.size() - 1);
    enter_stop.enter_ = true;
    auto& dep = enter_stop.departure_;
    dep.valid_ = true;
    dep.schedule_timestamp_ = motis_to_unixtime(sched, leg.enter_time_);
    dep.timestamp_ = dep.schedule_timestamp_;

    auto& exit_stop = push_station(exit_station);
    auto const exit_idx = static_cast<unsigned>(j.stops_.size() - 1);
    exit_stop.exit_ = true;
    auto& arr = exit_stop.arrival_;
    arr.valid_ = true;
    arr.schedule_timestamp_ = motis_to_unixtime(sched, leg.exit_time_);
    arr.timestamp_ = arr.schedule_timestamp_;
    last_time = arr.schedule_timestamp_;

    auto const* trp = get_trip(sched, leg.trip_idx_);

    j.transports_.emplace_back(journey::transport{
        .from_ = enter_idx,
        .to_ = exit_idx,
        .is_walk_ = false,
    });

    j.trips_.emplace_back(
        journey::trip{.from_ = enter_idx,
                      .to_ = exit_idx,
                      .extern_trip_ = to_extern_trip(sched, trp),
                      .debug_ = trp->dbg_.str()});
  }

  if (cj.final_footpath().is_footpath()) {
    auto const& fp = cj.final_footpath();
    auto const& from_station = *sched.stations_.at(fp.from_station_id_);
    auto const& to_station = *sched.stations_.at(fp.to_station_id_);

    auto& from_stop = push_station(from_station);
    auto const from_idx = static_cast<unsigned>(j.stops_.size() - 1);
    auto& dep = from_stop.departure_;
    dep.valid_ = true;
    dep.schedule_timestamp_ = last_time;
    dep.timestamp_ = dep.schedule_timestamp_;

    auto& to_stop = push_station(to_station);
    auto const to_idx = static_cast<unsigned>(j.stops_.size() - 1);
    auto& arr = to_stop.arrival_;
    arr.valid_ = true;
    arr.schedule_timestamp_ = last_time + fp.duration_ * 60;
    arr.timestamp_ = arr.schedule_timestamp_;

    j.transports_.emplace_back(journey::transport{.from_ = from_idx,
                                                  .to_ = to_idx,
                                                  .is_walk_ = true,
                                                  .duration_ = fp.duration_,
                                                  .mumo_id_ = -1});
  }

  return j;
}

Offset<Connection> compact_journey_to_basic_connection(
    FlatBufferBuilder& fbb, schedule const& sched, compact_journey const& cj) {
  return to_connection(fbb, compact_journey_to_basic_journey(sched, cj));
}

msg_ptr revise_journeys(std::vector<journey> const& journeys,
                        ctx::res_id_t const schedule_res_id) {
  message_creator mc;
  mc.create_and_finish(
      MsgContent_ReviseRequest,
      CreateReviseRequest(
          mc,
          mc.CreateVector(utl::to_vec(
              journeys, [&](auto const& j) { return to_connection(mc, j); })),
          schedule_res_id)
          .Union(),
      "/revise");
  auto const req = make_msg(mc);
  return motis_call(req)->val();
}

msg_ptr revise_compact_journey(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonReviseCompactJourneyRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto const& uv = uv_access.uv_;

  auto const cjs = utl::to_vec(*req->journeys(), [&](auto const& fbs_cj) {
    return from_fbs(sched, fbs_cj);
  });

  auto const revised = revise_journeys(
      utl::to_vec(cjs,
                  [&](auto const& cj) {
                    return compact_journey_to_basic_journey(sched, cj);
                  }),
      uv.schedule_res_id_);

  auto const revise_res = motis_content(ReviseResponse, revised);

  message_creator mc;
  mc.create_and_finish(
      MsgContent_PaxMonReviseCompactJourneyResponse,
      CreatePaxMonReviseCompactJourneyResponse(
          mc, mc.CreateVector(utl::to_vec(*revise_res->connections(),
                                          [&](auto const& con) {
                                            return motis_copy_table(Connection,
                                                                    mc, con);
                                          })))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
