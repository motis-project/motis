#include "motis/paxmon/delayed_init.h"

#include <cstdint>
#include <algorithm>
#include <limits>
#include <memory>

#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/access/time_access.h"
#include "motis/core/journey/message_to_journeys.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/message.h"

#include "motis/paxmon/loader/motis_journeys/motis_journeys.h"

using namespace motis::module;
using namespace motis::logging;

namespace motis::paxmon {

msg_ptr initial_reroute_query(schedule const& sched,
                              loader::unmatched_journey const& uj,
                              std::string const& router,
                              unixtime const min_time,
                              unixtime const max_time) {
  using namespace motis::routing;

  message_creator fbb;

  auto const planned_departure = static_cast<unixtime>(
      motis_to_unixtime(sched.schedule_begin_, uj.departure_time_));

  // should be ensured while loading the journey
  utl::verify(planned_departure >= min_time && planned_departure <= max_time,
              "initial_reroute_query: departure time out of range");

  auto const interval =
      Interval{std::max(min_time, planned_departure - 2 * 60 * 60),
               std::min(max_time, planned_departure + 2 * 60 * 60)};
  auto const& start_station = sched.stations_.at(uj.start_station_idx_);
  auto const& destination_station =
      sched.stations_.at(uj.destination_station_idx_);
  fbb.create_and_finish(
      MsgContent_RoutingRequest,
      CreateRoutingRequest(
          fbb, Start_PretripStart,
          CreatePretripStart(
              fbb,
              CreateInputStation(fbb, fbb.CreateString(start_station->eva_nr_),
                                 fbb.CreateString(start_station->name_)),
              &interval)
              .Union(),
          CreateInputStation(fbb,
                             fbb.CreateString(destination_station->eva_nr_),
                             fbb.CreateString(destination_station->name_)),
          SearchType_Default, SearchDir_Forward,
          fbb.CreateVector(std::vector<flatbuffers::Offset<Via>>{}),
          fbb.CreateVector(
              std::vector<flatbuffers::Offset<AdditionalEdgeWrapper>>{}))
          .Union(),
      router);
  return make_msg(fbb);
}

journey const* get_closest_journey(schedule const& sched,
                                   loader::unmatched_journey const& uj,
                                   std::vector<journey> const& journeys) {
  journey const* best_journey = nullptr;
  auto best_score = std::numeric_limits<std::int64_t>::max();
  auto const planned_dep = static_cast<std::int64_t>(
      motis_to_unixtime(sched.schedule_begin_, uj.departure_time_));
  auto const planned_arr = static_cast<std::int64_t>(
      motis_to_unixtime(sched.schedule_begin_, uj.arrival_time_));

  for (auto const& j : journeys) {
    if (j.stops_.empty()) {
      continue;
    }
    auto const dep_diff = static_cast<std::int64_t>(
                              j.stops_.front().departure_.schedule_timestamp_) -
                          planned_dep;
    auto const arr_diff = static_cast<std::int64_t>(
                              j.stops_.back().arrival_.schedule_timestamp_) -
                          planned_arr;
    auto const score = dep_diff * dep_diff + arr_diff * arr_diff;
    if (score < best_score) {
      best_score = score;
      best_journey = &j;
    }
  }

  return best_journey;
}

void delayed_init(paxmon_data& data, universe& uv, schedule const& sched,
                  delay_init_options const& opt) {
  using namespace motis::ris;
  using namespace motis::routing;

  auto const ris_status_msg = motis_call(make_no_msg("/ris/status"))->val();
  auto const ris_status = motis_content(RISStatusResponse, ris_status_msg);

  if (!ris_status->delayed_init()) {
    LOG(warn) << "required option ris.delayed_init=1 not set, rerouting "
                 "unmatched journeys not possible";
    return;
  }

  if (opt.reroute_unmatched_) {
    auto const min_time = external_schedule_begin(sched);
    auto const max_time = external_schedule_end(sched);

    for (auto& ljf : data.loaded_journey_files_) {
      if (ljf.unmatched_journeys_.empty()) {
        continue;
      }

      scoped_timer const timer{"reroute unmatched journeys"};
      LOG(info) << "routing " << ljf.unmatched_journeys_.size()
                << " unmatched journeys from " << ljf.path_.filename()
                << " using " << opt.initial_reroute_router_ << "...";
      auto const futures =
          utl::to_vec(ljf.unmatched_journeys_, [&](auto const& uj) {
            return motis_call(initial_reroute_query(
                sched, uj, opt.initial_reroute_router_, min_time, max_time));
          });
      ctx::await_all(futures);
      LOG(info) << "adding replacement journeys...";
      for (auto const& [uj, fut] : utl::zip(ljf.unmatched_journeys_, futures)) {
        auto const rr_msg = fut->val();
        auto const rr = motis_content(RoutingResponse, rr_msg);
        auto const journeys = message_to_journeys(rr);
        auto const* journey = get_closest_journey(sched, uj, journeys);
        if (journey == nullptr) {
          continue;
        }
        ++ljf.unmatched_journey_rerouted_count_;
        ljf.unmatched_pax_rerouted_count_ += uj.passengers_;

        if (uj.group_sizes_.empty()) {
          loader::motis_journeys::load_journey(
              sched, uv, *journey, uj.source_, uj.passengers_,
              route_source_flags::MATCH_REROUTED);
          ++ljf.unmatched_group_rerouted_count_;
        } else {
          auto source = uj.source_;
          for (auto const& group_size : uj.group_sizes_) {
            loader::motis_journeys::load_journey(
                sched, uv, *journey, source, group_size,
                route_source_flags::MATCH_REROUTED);
            ++source.secondary_ref_;
            ++ljf.unmatched_group_rerouted_count_;
          }
        }
      }
      ljf.unmatched_journeys_.clear();
    }
  } else {
    for (auto& ljf : data.loaded_journey_files_) {
      if (!ljf.unmatched_journeys_.empty()) {
        LOG(warn) << "ignoring " << ljf.unmatched_journeys_.size()
                  << " unmatched journeys from " << ljf.path_.filename()
                  << ", set paxmon.reroute_unmatched=1 to enable rerouting";
        ljf.unmatched_journeys_.clear();
      }
    }
  }

  motis_call(make_no_msg("/ris/delayed_init"))->val();
}

}  // namespace motis::paxmon
