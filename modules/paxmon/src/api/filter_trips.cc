#include "motis/paxmon/api/filter_trips.h"

#include <algorithm>
#include <iterator>
#include <tuple>

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/access/trip_access.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

namespace {

struct trip_info {
  trip_idx_t trip_idx_{};
  time first_departure_{};

  unsigned section_count_{};
  unsigned critical_sections_{};
  unsigned crowded_sections_{};

  unsigned max_excess_pax_{};
  unsigned cumulative_excess_pax_{};

  std::uint16_t max_expected_pax_{};

  std::vector<edge_load_info> edge_load_infos_{};
};

}  // namespace

msg_ptr filter_trips(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonFilterTripsRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;
  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);

  auto const ignore_past_sections =
      req->ignore_past_sections() && current_time != INVALID_TIME;
  auto const include_load_threshold = req->include_load_threshold();
  auto const max_results = req->max_results();
  auto const skip_first = req->skip_first();
  auto const critical_load_threshold = req->critical_load_threshold();
  auto const crowded_load_threshold = req->crowded_load_threshold();
  auto const include_edges = req->include_edges();
  auto const filter_by_time = req->filter_by_time();
  auto const filter_interval_begin =
      unix_to_motistime(sched.schedule_begin_, req->filter_interval()->begin());
  auto const filter_interval_end =
      unix_to_motistime(sched.schedule_begin_, req->filter_interval()->end());

  auto total_critical_sections = 0ULL;
  std::vector<trip_info> selected_trips;

  for (auto const& [trp_idx, tdi] : uv.trip_data_.mapping_) {
    auto ti = trip_info{trp_idx};
    auto include = false;

    if (filter_by_time != PaxMonFilterTripsTimeFilter_NoFilter) {
      auto const* trp = get_trip(sched, trp_idx);
      auto const dep = trp->id_.primary_.get_time();
      if (filter_by_time == PaxMonFilterTripsTimeFilter_DepartureTime) {
        if (dep < filter_interval_begin || dep >= filter_interval_end) {
          continue;
        }
      } else /* if (filter_by_time ==
                PaxMonFilterTripsTimeFilter_DepartureOrArrivalTime) */
      {
        auto const arr = trp->id_.secondary_.target_time_;
        if ((dep < filter_interval_begin || dep >= filter_interval_end) &&
            (arr < filter_interval_begin || arr >= filter_interval_end)) {
          continue;
        }
      }
    }

    for (auto const& ei : uv.trip_data_.edges(tdi)) {
      auto const* e = ei.get(uv);
      if (!e->is_trip()) {
        continue;
      }
      if (ti.first_departure_ == 0) {
        ti.first_departure_ = e->from(uv)->schedule_time();
      }
      auto const ignore_section =
          ignore_past_sections && e->to(uv)->current_time() < current_time;
      if (!include_edges && ignore_section) {
        continue;
      }
      auto const groups = uv.pax_connection_info_.groups_[e->pci_];
      auto const pdf = get_load_pdf(uv.passenger_groups_, groups);
      auto const cdf = get_cdf(pdf);
      auto const capacity = e->capacity();
      auto const pax_limits = get_pax_limits(uv.passenger_groups_, groups);
      auto const expected_pax = get_expected_load(uv, e->pci_);
      if (include_edges) {
        ti.edge_load_infos_.emplace_back(edge_load_info{
            e, cdf, false, load_factor_possibly_ge(cdf, capacity, 1.0F),
            expected_pax});
        if (ignore_section) {
          continue;
        }
      }
      ++ti.section_count_;
      ti.max_expected_pax_ = std::max(ti.max_expected_pax_, expected_pax);
      if (!e->has_capacity()) {
        continue;
      }
      if (!include &&
          load_factor_possibly_ge(cdf, capacity, include_load_threshold)) {
        include = true;
      }
      if (load_factor_possibly_ge(cdf, capacity, critical_load_threshold)) {
        ++ti.critical_sections_;
        ++total_critical_sections;
      } else if (load_factor_possibly_ge(cdf, capacity,
                                         crowded_load_threshold)) {
        ++ti.crowded_sections_;
      }
      if (pax_limits.max_ > capacity) {
        auto const excess_pax =
            static_cast<unsigned>(pax_limits.max_ - capacity);
        ti.max_excess_pax_ = std::max(ti.max_excess_pax_, excess_pax);
        ti.cumulative_excess_pax_ += excess_pax;
      }
    }
    if (include) {
      selected_trips.emplace_back(ti);
    }
  }

  switch (req->sort_by()) {
    case PaxMonFilterTripsSortOrder_MostCritical:
      std::stable_sort(
          begin(selected_trips), end(selected_trips),
          [](trip_info const& lhs, trip_info const& rhs) {
            return std::tie(lhs.max_excess_pax_, lhs.cumulative_excess_pax_,
                            lhs.critical_sections_, lhs.crowded_sections_) >
                   std::tie(rhs.max_excess_pax_, rhs.cumulative_excess_pax_,
                            rhs.critical_sections_, rhs.crowded_sections_);
          });
      break;
    case PaxMonFilterTripsSortOrder_FirstDeparture:
      std::stable_sort(begin(selected_trips), end(selected_trips),
                       [](trip_info const& lhs, trip_info const& rhs) {
                         return lhs.first_departure_ < rhs.first_departure_;
                       });
      break;
    case PaxMonFilterTripsSortOrder_ExpectedPax:
      std::stable_sort(begin(selected_trips), end(selected_trips),
                       [](trip_info const& lhs, trip_info const& rhs) {
                         return lhs.max_expected_pax_ > rhs.max_expected_pax_;
                       });
      break;
    case PaxMonFilterTripsSortOrder_TrainNr:
      std::stable_sort(begin(selected_trips), end(selected_trips),
                       [&](trip_info const& lhs, trip_info const& rhs) {
                         auto const* lhs_trp = get_trip(sched, lhs.trip_idx_);
                         auto const* rhs_trp = get_trip(sched, rhs.trip_idx_);
                         return lhs_trp->id_.primary_.train_nr_ <
                                rhs_trp->id_.primary_.train_nr_;
                       });
      break;
    default: break;
  }

  auto const total_matching_trips = selected_trips.size();
  if (skip_first > 0) {
    selected_trips.erase(
        begin(selected_trips),
        std::next(begin(selected_trips),
                  std::min(static_cast<std::size_t>(skip_first),
                           selected_trips.size())));
  }

  auto remaining_trips = 0ULL;
  if (max_results != 0 && selected_trips.size() > max_results) {
    remaining_trips = selected_trips.size() - max_results;
    selected_trips.resize(max_results);
  }

  message_creator mc;
  mc.create_and_finish(
      MsgContent_PaxMonFilterTripsResponse,
      CreatePaxMonFilterTripsResponse(
          mc, total_matching_trips, selected_trips.size(), remaining_trips,
          total_critical_sections,
          mc.CreateVector(utl::to_vec(
              selected_trips,
              [&](trip_info const& ti) {
                return CreatePaxMonFilteredTripInfo(
                    mc,
                    to_fbs_trip_service_info(mc, sched,
                                             get_trip(sched, ti.trip_idx_)),
                    ti.section_count_, ti.critical_sections_,
                    ti.crowded_sections_, ti.max_excess_pax_,
                    ti.cumulative_excess_pax_, ti.max_expected_pax_,
                    mc.CreateVector(utl::to_vec(
                        ti.edge_load_infos_, [&](edge_load_info const& eli) {
                          return to_fbs(mc, sched, uv, eli);
                        })));
              })))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
