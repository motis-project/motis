#include "motis/paxmon/api/broken_transfers.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <utility>

#include "utl/to_vec.h"

#include "motis/core/access/time_access.h"

#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/util/detailed_transfer_info.h"

using namespace motis::module;
using namespace motis::paxmon;
using namespace motis::paxmon::util;
using namespace flatbuffers;

namespace motis::paxmon::api {

msg_ptr broken_transfers(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonBrokenTransfersRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto const& uv = uv_access.uv_;

  auto const max_results = req->max_results();
  auto const skip_first = req->skip_first();

  auto const filter_interval_begin =
      req->filter_interval()->begin() != 0
          ? unix_to_motistime(sched.schedule_begin_,
                              req->filter_interval()->begin())
          : 0;
  auto const filter_interval_end =
      req->filter_interval()->end() != 0
          ? unix_to_motistime(sched.schedule_begin_,
                              req->filter_interval()->end())
          : std::numeric_limits<time>::max();

  auto const include_insufficient_transfer_time =
      req->include_insufficient_transfer_time();
  auto const include_missed_initial_departure =
      req->include_missed_initial_departure();
  auto const include_canceled_transfer = req->include_canceled_transfer();
  auto const include_canceled_initial_departure =
      req->include_canceled_initial_departure();
  auto const include_canceled_final_arrival =
      req->include_canceled_final_arrival();
  auto const only_planned_routes = req->only_planned_routes();

  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  auto const ignore_past_transfers =
      req->ignore_past_transfers() && current_time != INVALID_TIME;

  auto const include_event = [&](event_node const* ev) {
    if (ev->is_enter_exit_node()) {
      return false;
    }
    if (ignore_past_transfers && ev->current_time() < current_time) {
      return false;
    }
    return (ev->schedule_time() >= filter_interval_begin &&
            ev->schedule_time() <= filter_interval_end) ||
           (ev->current_time() >= filter_interval_begin &&
            ev->current_time() <= filter_interval_end);
  };

  message_creator mc;

  auto transfers = std::vector<detailed_transfer_info>{};
  auto broken_transfers = 0U;

  auto const gdti_options = get_detailed_transfer_info_options{
      .include_disabled_group_routes_ = true,
      .include_delay_info_ = true,
      .only_planned_routes_ = only_planned_routes};

  for (auto const& bucket : uv.interchanges_at_station_) {
    auto const station_idx = bucket.index();
    for (auto const& ei : bucket) {
      auto const* ic_edge = ei.get(uv);

      if (!ic_edge->is_broken()) {
        continue;
      }

      auto const* from = ic_edge->from(uv);
      if (from->station_idx() != station_idx) {
        // transfers are included in from and to station buckets,
        // only process them once (as outgoing edges)
        continue;
      }
      auto const* to = ic_edge->to(uv);

      if (!include_event(from) && !include_event(to)) {
        continue;
      }

      if (from->station_idx() == 0) {  // initial departure
        if (to->is_canceled() ? !include_canceled_initial_departure
                              : !include_missed_initial_departure) {
          continue;
        }
      } else if (to->station_idx() == 0) {  // final arrival
        if (!include_canceled_final_arrival && from->is_canceled()) {
          continue;
        }
      } else if (from->is_canceled() || to->is_canceled()) {
        if (!include_canceled_transfer) {
          continue;
        }
      } else if (!include_insufficient_transfer_time) {
        continue;
      }

      ++broken_transfers;

      auto& info = transfers.emplace_back(
          get_detailed_transfer_info(uv, sched, ei, mc, gdti_options));

      if (info.group_count_ == 0) {
        transfers.pop_back();
      }
    }
  }

  switch (req->sort_by()) {
    case PaxMonBrokenTransfersSortOrder_AffectedPax:
      std::stable_sort(
          begin(transfers), end(transfers),
          [](detailed_transfer_info const& lhs,
             detailed_transfer_info const& rhs) {
            return std::tie(lhs.pax_count_, lhs.total_delay_increase_) >
                   std::tie(rhs.pax_count_, rhs.total_delay_increase_);
          });
      break;
    case PaxMonBrokenTransfersSortOrder_TotalDelayIncrease:
      std::stable_sort(begin(transfers), end(transfers),
                       [](detailed_transfer_info const& lhs,
                          detailed_transfer_info const& rhs) {
                         return lhs.total_delay_increase_ >
                                rhs.total_delay_increase_;
                       });
      break;
    case PaxMonBrokenTransfersSortOrder_SquaredTotalDelayIncrease:
      std::stable_sort(begin(transfers), end(transfers),
                       [](detailed_transfer_info const& lhs,
                          detailed_transfer_info const& rhs) {
                         return lhs.squared_total_delay_increase_ >
                                rhs.squared_total_delay_increase_;
                       });
      break;
    case PaxMonBrokenTransfersSortOrder_MinDelayIncrease:
      std::stable_sort(begin(transfers), end(transfers),
                       [](detailed_transfer_info const& lhs,
                          detailed_transfer_info const& rhs) {
                         return std::tie(lhs.min_delay_increase_,
                                         lhs.total_delay_increase_) >
                                std::tie(rhs.min_delay_increase_,
                                         rhs.total_delay_increase_);
                       });
      break;
    case PaxMonBrokenTransfersSortOrder_MaxDelayIncrease:
      std::stable_sort(begin(transfers), end(transfers),
                       [](detailed_transfer_info const& lhs,
                          detailed_transfer_info const& rhs) {
                         return std::tie(lhs.max_delay_increase_,
                                         lhs.total_delay_increase_) >
                                std::tie(rhs.max_delay_increase_,
                                         rhs.total_delay_increase_);
                       });
      break;
    case PaxMonBrokenTransfersSortOrder_UnreachablePax:
      std::stable_sort(
          begin(transfers), end(transfers),
          [](detailed_transfer_info const& lhs,
             detailed_transfer_info const& rhs) {
            return std::tie(lhs.unreachable_pax_, lhs.total_delay_increase_) >
                   std::tie(rhs.unreachable_pax_, rhs.total_delay_increase_);
          });
      break;
  }

  auto const total_matching_transfers = transfers.size();

  if (skip_first > 0) {
    transfers.erase(begin(transfers),
                    std::next(begin(transfers),
                              std::min(static_cast<std::size_t>(skip_first),
                                       transfers.size())));
  }

  auto remaining = 0ULL;
  if (max_results != 0 && transfers.size() > max_results) {
    remaining = transfers.size() - max_results;
    transfers.resize(max_results);
  }

  mc.create_and_finish(
      MsgContent_PaxMonBrokenTransfersResponse,
      CreatePaxMonBrokenTransfersResponse(
          mc, total_matching_transfers, transfers.size(), remaining,
          skip_first + transfers.size(),
          mc.CreateVector(utl::to_vec(transfers,
                                      [&](auto const& info) {
                                        return info.to_fbs_transfer_info(
                                            mc, uv, sched, true);
                                      })))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
