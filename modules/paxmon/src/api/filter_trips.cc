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

#include "motis/paxmon/api/util/trip_time_filter.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

namespace {

struct trip_info {
  trip_idx_t trip_idx_{};
  trip_data_index tdi_{};
  time first_departure_{};

  unsigned section_count_{};
  unsigned critical_sections_{};
  unsigned crowded_sections_{};

  unsigned max_excess_pax_{};
  unsigned cumulative_excess_pax_{};

  float max_load_{};
  float first_critical_load_{};
  time first_critical_time_{INVALID_TIME};

  std::uint16_t max_expected_pax_{};
  std::uint16_t max_pax_range_{};
  std::uint16_t max_pax_{};
  std::uint16_t max_capacity_{};

  std::vector<edge_load_info> edge_load_infos_{};
};

}  // namespace

msg_ptr filter_trips(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonFilterTripsRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto const& uv = uv_access.uv_;
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
  auto const filter_by_train_nr = req->filter_by_train_nr();
  auto const filter_train_nrs = utl::to_vec(*req->filter_train_nrs());
  auto const filter_by_service_class = req->filter_by_service_class();
  auto const filter_service_classes = utl::to_vec(
      *req->filter_service_classes(),
      [](auto const& sc) { return static_cast<service_class>(sc); });
  auto const filter_by_capacity_status = req->filter_by_capacity_status();
  auto const filter_has_trip_formation = req->filter_has_trip_formation();
  auto const filter_has_capacity_for_all_sections =
      req->filter_has_capacity_for_all_sections();

  auto const trip_filters_active =
      (filter_by_time != PaxMonFilterTripsTimeFilter_NoFilter) ||
      filter_by_train_nr || filter_by_service_class;

  auto total_critical_sections = 0ULL;
  std::vector<trip_info> selected_trips;

  for (auto const& [trp_idx, tdi] : uv.trip_data_.mapping_) {
    auto ti = trip_info{.trip_idx_ = trp_idx, .tdi_ = tdi};
    auto const trip_edges = uv.trip_data_.edges(tdi);
    auto include = false;

    if (trip_edges.empty()) {
      continue;
    }

    if (trip_filters_active) {
      auto const* trp = get_trip(sched, trp_idx);

      if (filter_by_train_nr) {
        auto const train_nr = trp->id_.primary_.get_train_nr();
        if (std::find(begin(filter_train_nrs), end(filter_train_nrs),
                      train_nr) == end(filter_train_nrs)) {
          continue;
        }
      }

      if (!include_trip_based_on_time_filter(trp, filter_by_time,
                                             filter_interval_begin,
                                             filter_interval_end)) {
        continue;
      }

      if (filter_by_service_class) {
        auto const trip_class = trip_edges[0].get(uv)->clasz_;
        if (std::find(begin(filter_service_classes),
                      end(filter_service_classes),
                      trip_class) == end(filter_service_classes)) {
          continue;
        }
      }
    }

    if (filter_by_capacity_status) {
      auto const& tcs = uv.trip_data_.capacity_status(tdi);
      if (tcs.has_trip_formation_ != filter_has_trip_formation) {
        continue;
      }
      if (tcs.has_capacity_for_all_sections_ !=
          filter_has_capacity_for_all_sections) {
        continue;
      }
    }

    for (auto const& ei : trip_edges) {
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
      auto const group_routes = uv.pax_connection_info_.group_routes(e->pci_);
      auto const pdf = get_load_pdf(uv.passenger_groups_, group_routes);
      auto const cdf = get_cdf(pdf);
      auto const capacity = e->capacity();
      auto const pax_limits =
          get_pax_limits(uv.passenger_groups_, group_routes);
      auto const expected_pax = get_expected_load(uv, e->pci_);
      ti.max_pax_range_ = std::max(
          ti.max_pax_range_,
          static_cast<std::uint16_t>(pax_limits.max_ - pax_limits.min_));
      ti.max_pax_ = std::max(ti.max_pax_, pax_limits.max_);
      ti.max_capacity_ = std::max(ti.max_capacity_, capacity);
      if (include_edges) {
        ti.edge_load_infos_.emplace_back(
            make_edge_load_info(uv, e, pdf, cdf, false));
        if (ignore_section) {
          continue;
        }
      }
      ++ti.section_count_;
      ti.max_expected_pax_ = std::max(ti.max_expected_pax_, expected_pax);
      if (!include && include_load_threshold == 0.0F) {
        include = true;
      }
      if (!e->has_capacity()) {
        continue;
      }
      if (!include &&
          load_factor_possibly_ge(cdf, capacity, include_load_threshold)) {
        include = true;
      }
      auto const load =
          static_cast<float>(pax_limits.max_) / static_cast<float>(capacity);
      ti.max_load_ = std::max(ti.max_load_, load);
      if (load_factor_possibly_ge(cdf, capacity, critical_load_threshold)) {
        ++ti.critical_sections_;
        ++total_critical_sections;
        if (ti.first_critical_time_ == INVALID_TIME) {
          ti.first_critical_load_ = load;
          ti.first_critical_time_ = e->from(uv)->current_time();
        }
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
                         return std::tie(lhs_trp->id_.primary_.train_nr_,
                                         lhs.first_departure_) <
                                std::tie(rhs_trp->id_.primary_.train_nr_,
                                         rhs.first_departure_);
                       });
      break;
    case PaxMonFilterTripsSortOrder_MaxLoad:
      std::stable_sort(begin(selected_trips), end(selected_trips),
                       [](trip_info const& lhs, trip_info const& rhs) {
                         return std::tie(lhs.max_load_, lhs.max_excess_pax_,
                                         lhs.cumulative_excess_pax_) >
                                std::tie(rhs.max_load_, rhs.max_excess_pax_,
                                         rhs.cumulative_excess_pax_);
                       });
      break;
    case PaxMonFilterTripsSortOrder_EarliestCritical:
      std::stable_sort(
          begin(selected_trips), end(selected_trips),
          [](trip_info const& lhs, trip_info const& rhs) {
            if (lhs.first_critical_time_ < rhs.first_critical_time_) {
              return true;
            } else {
              return std::tie(lhs.first_critical_load_, lhs.max_load_,
                              lhs.max_excess_pax_, lhs.cumulative_excess_pax_) >
                     std::tie(rhs.first_critical_load_, rhs.max_load_,
                              rhs.max_excess_pax_, rhs.cumulative_excess_pax_);
            }
          });
      break;
    case PaxMonFilterTripsSortOrder_MaxPaxRange:
      std::stable_sort(begin(selected_trips), end(selected_trips),
                       [](trip_info const& lhs, trip_info const& rhs) {
                         return std::tie(lhs.max_pax_range_, lhs.max_load_,
                                         lhs.max_excess_pax_) >
                                std::tie(rhs.max_pax_range_, rhs.max_load_,
                                         rhs.max_excess_pax_);
                       });
      break;
    case PaxMonFilterTripsSortOrder_MaxPax:
      std::stable_sort(begin(selected_trips), end(selected_trips),
                       [](trip_info const& lhs, trip_info const& rhs) {
                         return std::tie(lhs.max_pax_, lhs.max_load_) >
                                std::tie(rhs.max_pax_, rhs.max_load_);
                       });
      break;
    case PaxMonFilterTripsSortOrder_MaxCapacity:
      std::stable_sort(begin(selected_trips), end(selected_trips),
                       [](trip_info const& lhs, trip_info const& rhs) {
                         return std::tie(lhs.max_capacity_, lhs.max_load_) >
                                std::tie(rhs.max_capacity_, rhs.max_load_);
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
          skip_first + selected_trips.size(), total_critical_sections,
          mc.CreateVector(utl::to_vec(
              selected_trips,
              [&](trip_info const& ti) {
                return CreatePaxMonFilteredTripInfo(
                    mc,
                    to_fbs_trip_service_info(mc, sched,
                                             get_trip(sched, ti.trip_idx_)),
                    ti.section_count_, ti.critical_sections_,
                    ti.crowded_sections_, ti.max_excess_pax_,
                    ti.cumulative_excess_pax_, ti.max_load_,
                    ti.max_expected_pax_,
                    to_fbs(mc, uv.trip_data_.capacity_status(ti.tdi_)),
                    mc.CreateVector(utl::to_vec(
                        ti.edge_load_infos_, [&](edge_load_info const& eli) {
                          return to_fbs(mc, sched, uv, eli);
                        })));
              })))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
