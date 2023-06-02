#include "motis/paxmon/api/capacity_status.h"

#include <cstdint>
#include <algorithm>
#include <optional>
#include <utility>

#include "utl/to_vec.h"

#include "motis/hash_map.h"
#include "motis/hash_set.h"
#include "motis/pair.h"
#include "motis/string.h"

#include "motis/core/common/logging.h"

#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"

#include "motis/paxmon/capacity_internal.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace motis::logging;
using namespace motis::paxmon;
using namespace flatbuffers;

namespace motis::paxmon::api {

namespace {

struct capacity_stats {
  Offset<PaxMonTripCapacityStats> to_fbs(FlatBufferBuilder& fbb) const {
    return CreatePaxMonTripCapacityStats(
        fbb, tracked_, ok_, no_formation_data_at_all_,
        no_formation_data_some_sections_some_merged_,
        no_formation_data_some_sections_all_merged_, no_vehicles_found_at_all_,
        no_vehicles_found_some_sections_,
        some_vehicles_not_found_some_sections_);
  }

  std::uint32_t tracked_{};

  std::uint32_t ok_{};

  std::uint32_t no_formation_data_at_all_{};
  std::uint32_t no_formation_data_some_sections_some_merged_{};
  std::uint32_t no_formation_data_some_sections_all_merged_{};

  std::uint32_t no_vehicles_found_at_all_{};
  std::uint32_t no_vehicles_found_some_sections_{};
  std::uint32_t some_vehicles_not_found_some_sections_{};
};

struct mini_vehicle_info {
  std::uint64_t uic_{};
  mcd::string baureihe_;
  mcd::string type_code_;
};

struct missing_vehicle_info {
  std::string_view baureihe_;
  std::string_view type_code_;
  std::uint32_t count_{};
};

}  // namespace

msg_ptr capacity_status(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonCapacityStatusRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto const& uv = uv_access.uv_;
  auto const& caps = uv.capacity_maps_;

  auto const include_trips_without_capacity =
      req->include_trips_without_capacity();
  auto const include_other_trips_without_capacity =
      req->include_other_trips_without_capacity();
  auto const include_missing_vehicle_infos =
      req->include_missing_vehicle_infos();
  auto const include_uics_not_found = req->include_uics_not_found();

  message_creator mc;

  auto all_trips = capacity_stats{};
  auto high_speed_trips = capacity_stats{};
  auto long_distance_trips = capacity_stats{};
  auto other_trips = capacity_stats{};

  auto trips_without_capacity =
      std::vector<Offset<PaxMonTripCapacityShortInfo>>{};
  auto uics_not_found = mcd::hash_set<std::uint64_t>{};
  auto missing_vehicle_counts =
      mcd::hash_map<mcd::pair<mcd::string, mcd::string>, std::uint32_t>{};

  for (auto const& [trp_idx, tdi] : uv.trip_data_.mapping_) {
    auto const* trp = get_trip(sched, trp_idx);
    auto sections = access::sections{trp};
    auto const sec_count = sections.size();
    auto clasz = std::optional<service_class>{};

    auto trip_ok = true;
    auto secs_with_all_missing_tfs = 0U;
    auto secs_with_some_missing_tfs = 0U;
    auto secs_with_all_missing_uics = 0U;
    auto secs_with_some_missing_uics = 0U;

    for (auto const& sec : sections) {
      auto const& lc = sec.lcon();

      if (!clasz) {
        clasz = lc.full_con_->clasz_;
      }

      auto vehicles = mcd::hash_set<mini_vehicle_info>{};

      auto mt_count = 0U;
      auto mt_tf_found = 0U;
      auto uics_found = 0U;

      for (auto const& merged_trp : *sched.merged_trips_.at(lc.trips_)) {
        ++mt_count;
        auto const* tf_sec = get_trip_formation_section(sched, caps, merged_trp,
                                                        sec.ev_key_from());
        if (tf_sec != nullptr) {
          ++mt_tf_found;
          for (auto const& vi : tf_sec->vehicles_) {
            vehicles.insert(
                mini_vehicle_info{vi.uic_, vi.baureihe_, vi.type_code_});
          }
        } else {
          trip_ok = false;
        }
      }

      for (auto const& mvi : vehicles) {
        if (auto const it = caps.vehicle_capacity_map_.find(mvi.uic_);
            it != end(caps.vehicle_capacity_map_)) {
          ++uics_found;
        } else {
          trip_ok = false;
          if (include_uics_not_found) {
            uics_not_found.insert(mvi.uic_);
          }
          if (include_missing_vehicle_infos) {
            ++missing_vehicle_counts[mcd::pair{mvi.baureihe_, mvi.type_code_}];
          }
        }
      }

      if (mt_tf_found == 0) {
        ++secs_with_all_missing_tfs;
      } else if (mt_tf_found < mt_count) {
        ++secs_with_some_missing_tfs;
      }

      if (!vehicles.empty() && uics_found == 0) {
        ++secs_with_all_missing_uics;
      } else if (uics_found < vehicles.size()) {
        ++secs_with_some_missing_uics;
      }
    }

    auto const count = [&](auto const& fn) {
      fn(all_trips);
      if (clasz) {
        if (*clasz == service_class::ICE) {
          fn(high_speed_trips);
        } else if (*clasz == service_class::IC) {
          fn(long_distance_trips);
        } else {
          fn(other_trips);
        }
      }
    };

    count([](capacity_stats& stats) { ++stats.tracked_; });

    if (trip_ok) {
      count([](capacity_stats& stats) { ++stats.ok_; });
    }

    if (secs_with_all_missing_tfs == sec_count) {
      count([](capacity_stats& stats) { ++stats.no_formation_data_at_all_; });
    } else if (secs_with_all_missing_tfs > 0) {
      count([](capacity_stats& stats) {
        ++stats.no_formation_data_some_sections_all_merged_;
      });
    }
    if (secs_with_some_missing_tfs > 0) {
      count([](capacity_stats& stats) {
        ++stats.no_formation_data_some_sections_some_merged_;
      });
    }

    if (secs_with_all_missing_uics == sec_count) {
      count([](capacity_stats& stats) { ++stats.no_vehicles_found_at_all_; });
    } else if (secs_with_all_missing_uics > 0) {
      count([](capacity_stats& stats) {
        ++stats.no_vehicles_found_some_sections_;
      });
    }
    if (secs_with_some_missing_uics > 0) {
      count([](capacity_stats& stats) {
        ++stats.some_vehicles_not_found_some_sections_;
      });
    }

    if (include_trips_without_capacity &&
        (secs_with_all_missing_tfs > 0 || secs_with_some_missing_tfs > 0 ||
         secs_with_all_missing_uics > 0 || secs_with_some_missing_uics > 0)) {
      if (include_other_trips_without_capacity ||
          (clasz &&
           (*clasz == service_class::ICE || *clasz == service_class::IC))) {
        trips_without_capacity.emplace_back(CreatePaxMonTripCapacityShortInfo(
            mc, to_fbs_trip_service_info(mc, sched, trp), sec_count,
            secs_with_all_missing_tfs, secs_with_some_missing_tfs,
            secs_with_all_missing_uics, secs_with_some_missing_uics));
      }
    }
  }

  auto missing_vehicle_infos =
      utl::to_vec(missing_vehicle_counts, [&](auto const& entry) {
        return missing_vehicle_info{entry.first.first.view(),
                                    entry.first.second.view(), entry.second};
      });

  std::sort(missing_vehicle_infos.begin(), missing_vehicle_infos.end(),
            [](auto const& a, auto const& b) { return a.count_ > b.count_; });

  auto const fbs_missing_vehicle_infos =
      utl::to_vec(missing_vehicle_infos, [&](missing_vehicle_info& mvi) {
        return CreatePaxMonMissingVehicleInfo(
            mc, mc.CreateString(mvi.baureihe_), mc.CreateString(mvi.type_code_),
            mvi.count_);
      });

  auto uics_not_found_vec =
      utl::to_vec(uics_not_found, [](auto const uic) { return uic; });

  std::sort(uics_not_found_vec.begin(), uics_not_found_vec.end());

  mc.create_and_finish(
      MsgContent_PaxMonCapacityStatusResponse,
      CreatePaxMonCapacityStatusResponse(
          mc, all_trips.to_fbs(mc), high_speed_trips.to_fbs(mc),
          long_distance_trips.to_fbs(mc), other_trips.to_fbs(mc),
          mc.CreateVector(trips_without_capacity),
          mc.CreateVector(fbs_missing_vehicle_infos),
          mc.CreateVector(uics_not_found_vec))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
