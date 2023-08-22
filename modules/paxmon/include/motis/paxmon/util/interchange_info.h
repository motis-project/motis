#pragma once

#include <cstdint>
#include <limits>

#include "flatbuffers/flatbuffers.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/conv/station_conv.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/universe.h"
#include "motis/paxmon/util/group_delay.h"

namespace motis::paxmon::util {

struct interchange_info {
  flatbuffers::Offset<PaxMonInterchangeInfo> to_fbs_interchange_info(
      flatbuffers::FlatBufferBuilder& fbb, universe const& uv,
      schedule const& sched, bool const include_delay_info) const {
    auto const id = PaxMonInterchangeId{ei_.node_, ei_.out_edge_idx_};
    auto const* e = ei_.get(uv);

    auto const make_fbs_event = [&](event_node const* ev, bool const arrival) {
      std::vector<flatbuffers::Offset<PaxMonTripStopInfo>> res;
      if (!ev->is_enter_exit_node()) {
        std::vector<flatbuffers::Offset<TripServiceInfo>> fbs_trips;
        // TODO(pablo): service infos only for arriving trip section
        if (arrival) {
          for (auto const& trp_edge : ev->incoming_edges(uv)) {
            if (trp_edge.is_trip() || trp_edge.is_disabled()) {
              for (auto const& trp : trp_edge.get_trips(sched)) {
                fbs_trips.emplace_back(
                    to_fbs_trip_service_info(fbb, sched, trp));
              }
            }
          }
        } else {
          for (auto const& trp_edge : ev->outgoing_edges(uv)) {
            if (trp_edge.is_trip() || trp_edge.is_disabled()) {
              for (auto const& trp : trp_edge.get_trips(sched)) {
                fbs_trips.emplace_back(
                    to_fbs_trip_service_info(fbb, sched, trp));
              }
            }
          }
        }
        res.emplace_back(CreatePaxMonTripStopInfo(
            fbb, motis_to_unixtime(sched, ev->schedule_time()),
            motis_to_unixtime(sched, ev->current_time()), ev->is_canceled(),
            fbb.CreateVector(fbs_trips),
            to_fbs(fbb, *sched.stations_.at(ev->station_idx()))));
      }
      return fbb.CreateVector(res);
    };

    return CreatePaxMonInterchangeInfo(
        fbb, &id, make_fbs_event(from_, true), make_fbs_event(to_, false),
        groups_, group_count_, pax_count_, e->transfer_time(), e->is_valid(uv),
        e->is_disabled(), e->is_broken(), e->is_canceled(uv),
        include_delay_info
            ? CreatePaxMonInterchangeDelayInfo(
                  fbb, min_delay_increase_, max_delay_increase_,
                  total_delay_increase_, squared_total_delay_increase_,
                  unreachable_pax_)
            : 0);
  }

  edge_index ei_;

  event_node const* from_{};
  event_node const* to_{};

  std::int32_t group_count_{};
  std::int32_t pax_count_{};

  std::int16_t min_delay_increase_{std::numeric_limits<std::int16_t>::max()};
  std::int16_t max_delay_increase_{};
  std::int64_t total_delay_increase_{};
  std::uint64_t squared_total_delay_increase_{};
  std::int32_t unreachable_pax_{};

  std::uint32_t normal_routes_{};
  std::uint32_t broken_routes_{};

  flatbuffers::Offset<PaxMonCombinedGroupRoutes> groups_{};
};

struct get_interchange_info_options {
  bool include_group_infos_{};
  bool include_disabled_group_routes_{};
  bool include_delay_info_{};
  bool only_planned_routes_{};
};

inline interchange_info get_interchange_info(
    universe const& uv, schedule const& sched, edge_index const ei,
    flatbuffers::FlatBufferBuilder& fbb,
    get_interchange_info_options const& options) {
  auto const* ic_edge = ei.get(uv);
  auto info = interchange_info{
      .ei_ = ei, .from_ = ic_edge->from(uv), .to_ = ic_edge->to(uv)};

  auto group_route_infos = std::vector<PaxMonGroupRouteBaseInfo>{};
  auto last_pg = std::numeric_limits<passenger_group_index>::max();

  auto const handle_pgwr = [&](auto const& pgwr) {
    if (options.only_planned_routes_ && pgwr.route_ != 0) {
      return;
    }
    auto const& pg = uv.passenger_groups_.at(pgwr.pg_);
    auto const& gr = uv.passenger_groups_.route(pgwr);

    auto const is_new_group = last_pg != pgwr.pg_;

    if (is_new_group) {
      last_pg = pgwr.pg_;
      ++info.group_count_;
      info.pax_count_ += pg->passengers_;
    }

    if (!options.include_disabled_group_routes_ && gr.probability_ == 0.0F) {
      return;
    }

    if (options.include_delay_info_) {
      auto const current_group_delay_info =
          get_current_estimated_delay(uv, pgwr.pg_);
      auto const current_group_est_delay =
          static_cast<std::int16_t>(current_group_delay_info.estimated_delay_);

      auto const sched_delay = get_scheduled_delay(uv, pgwr);

      auto const delay_increase =
          static_cast<std::int16_t>(current_group_est_delay - sched_delay);

      if (!current_group_delay_info.possibly_unreachable_) {
        info.min_delay_increase_ =
            std::min(info.min_delay_increase_, delay_increase);
        info.max_delay_increase_ =
            std::max(info.max_delay_increase_, delay_increase);
      }

      if (delay_increase > 0) {
        info.total_delay_increase_ += delay_increase;
        info.squared_total_delay_increase_ +=
            static_cast<std::uint64_t>(delay_increase) *
            static_cast<std::uint64_t>(delay_increase);
      }

      if (is_new_group && current_group_delay_info.possibly_unreachable_) {
        info.unreachable_pax_ += pg->passengers_;
      }
    }

    if (options.include_group_infos_) {
      group_route_infos.emplace_back(to_fbs_base_info(fbb, *pg, gr));
    }
  };

  auto const normal_routes =
      uv.pax_connection_info_.group_routes(ic_edge->pci_);
  auto const broken_routes =
      uv.pax_connection_info_.broken_group_routes(ic_edge->pci_);

  info.normal_routes_ = normal_routes.size();
  info.broken_routes_ = broken_routes.size();

  for (auto const& pgwr : normal_routes) {
    handle_pgwr(pgwr);
  }

  for (auto const& pgwr : broken_routes) {
    handle_pgwr(pgwr);
  }

  if (options.include_group_infos_) {
    std::sort(begin(group_route_infos), end(group_route_infos),
              [](PaxMonGroupRouteBaseInfo const& a,
                 PaxMonGroupRouteBaseInfo const& b) {
                return std::make_pair(a.g(), a.r()) <
                       std::make_pair(b.g(), b.r());
              });
    auto const pdf =
        get_load_pdf(uv.passenger_groups_,
                     uv.pax_connection_info_.group_routes(ic_edge->pci_));
    auto const cdf = get_cdf(pdf);
    info.groups_ = CreatePaxMonCombinedGroupRoutes(
        fbb, fbb.CreateVectorOfStructs(group_route_infos),
        to_fbs_distribution(fbb, pdf, cdf));
  }

  return info;
}

}  // namespace motis::paxmon::util
