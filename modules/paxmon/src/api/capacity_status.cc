#include "motis/paxmon/api/capacity_status.h"

#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <optional>
#include <sstream>
#include <utility>

#include "boost/uuid/uuid_io.hpp"

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
        fbb, tracked_, ok_, trip_formation_data_found_,
        no_formation_data_at_all_, no_formation_data_some_sections_some_merged_,
        no_formation_data_some_sections_all_merged_, no_vehicles_found_at_all_,
        no_vehicles_found_some_sections_,
        some_vehicles_not_found_some_sections_);
  }

  std::uint32_t tracked_{};

  std::uint32_t ok_{};

  std::uint32_t trip_formation_data_found_{};

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

enum class output_type { DEFAULT, CSV_TRIPS, CSV_FORMATIONS };

}  // namespace

msg_ptr capacity_status(paxmon_data& data, msg_ptr const& msg) {
  auto out_type = output_type::DEFAULT;
  auto include_trips_without_capacity = false;
  auto include_other_trips_without_capacity = false;
  auto include_missing_vehicle_infos = false;
  auto include_uics_not_found = false;
  auto uv_id = universe_id{};

  switch (msg->get()->content_type()) {
    case MsgContent_PaxMonCapacityStatusRequest: {
      auto const req = motis_content(PaxMonCapacityStatusRequest, msg);
      uv_id = req->universe();
      include_trips_without_capacity = req->include_trips_without_capacity();
      include_other_trips_without_capacity =
          req->include_other_trips_without_capacity();
      include_missing_vehicle_infos = req->include_missing_vehicle_infos();
      include_uics_not_found = req->include_uics_not_found();
      break;
    }
    case MsgContent_MotisNoMessage: {
      auto const target = msg->get()->destination()->target()->view();
      if (target.contains("trips.csv")) {
        out_type = output_type::CSV_TRIPS;
      } else if (target.contains("formations.csv")) {
        out_type = output_type::CSV_FORMATIONS;
      }
      break;
    }
    default:
      throw std::system_error(motis::module::error::unexpected_message_type);
  }

  auto const uv_access = get_universe_and_schedule(data, uv_id);
  auto const& sched = uv_access.sched_;
  auto const& uv = uv_access.uv_;
  auto const& caps = uv.capacity_maps_;

  message_creator mc;
  std::stringstream csv;

  auto all_trips = capacity_stats{};
  auto high_speed_trips = capacity_stats{};
  auto long_distance_trips = capacity_stats{};
  auto other_trips = capacity_stats{};

  auto trips_without_capacity =
      std::vector<Offset<PaxMonTripCapacityShortInfo>>{};
  auto uics_not_found = mcd::hash_set<std::uint64_t>{};
  auto missing_vehicle_counts =
      mcd::hash_map<mcd::pair<mcd::string, mcd::string>, std::uint32_t>{};

  auto used_tfs = mcd::hash_set<trip_formation const*>{};

  if (out_type == output_type::CSV_TRIPS) {
    csv << "category,train_nr,start_station_eva,start_station_name,start_time,"
           "end_station_eva,end_station_name,end_time,has_all_data,has_"
           "formation,trip_sections,formation_sections,sections_all_missing_"
           "tfs,sections_some_missing_tfs,sections_all_missing_uics,sections_"
           "some_missing_uics\n";
  }

  for (auto const& [trp_idx, tdi] : uv.trip_data_.mapping_) {
    auto const* trp = get_trip(sched, trp_idx);
    auto sections = access::sections{trp};
    auto const sec_count = sections.size();
    auto clasz = std::optional<service_class>{};
    connection_info const* con_info = nullptr;

    auto trip_ok = true;
    auto secs_with_all_missing_tfs = 0U;
    auto secs_with_some_missing_tfs = 0U;
    auto secs_with_all_missing_uics = 0U;
    auto secs_with_some_missing_uics = 0U;

    auto const* tf = get_trip_formation(uv.capacity_maps_, trp);
    if (tf != nullptr) {
      used_tfs.insert(tf);
    }

    for (auto const& sec : sections) {
      auto const& lc = sec.lcon();

      if (!clasz) {
        clasz = lc.full_con_->clasz_;
        con_info = lc.full_con_->con_info_;
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

    if (out_type == output_type::CSV_TRIPS) {
      auto const category =
          con_info ? sched.categories_.at(con_info->family_)->name_.view()
                   : std::string_view{};
      auto const train_nr = con_info ? con_info->train_nr_ : 0U;
      auto const& start_st =
          sched.stations_.at(trp->id_.primary_.get_station_id());
      auto const& end_st =
          sched.stations_.at(trp->id_.secondary_.target_station_id_);
      auto const start_time = motis_to_unixtime(sched.schedule_begin_,
                                                trp->id_.primary_.get_time());
      auto const end_time = motis_to_unixtime(sched.schedule_begin_,
                                              trp->id_.secondary_.target_time_);
      csv << std::quoted(category) << "," << train_nr << ","
          << std::quoted(start_st->eva_nr_.view()) << ","
          << std::quoted(start_st->name_.view()) << ","
          << format_unix_time(start_time) << ","
          << std::quoted(end_st->eva_nr_.view()) << ","
          << std::quoted(end_st->name_.view()) << ","
          << format_unix_time(end_time) << "," << trip_ok << ","
          << (tf != nullptr) << "," << sec_count << ","
          << (tf != nullptr ? tf->sections_.size() : 0U) << ","
          << secs_with_all_missing_tfs << "," << secs_with_some_missing_tfs
          << "," << secs_with_all_missing_uics << ","
          << secs_with_some_missing_uics << "\n";
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

    if (tf != nullptr) {
      count([](capacity_stats& stats) { ++stats.trip_formation_data_found_; });
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

    if (out_type == output_type::DEFAULT && include_trips_without_capacity &&
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

  if (out_type == output_type::DEFAULT) {
    auto const fbs_missing_vehicle_infos =
        utl::to_vec(missing_vehicle_infos, [&](missing_vehicle_info& mvi) {
          return CreatePaxMonMissingVehicleInfo(
              mc, mc.CreateString(mvi.baureihe_),
              mc.CreateString(mvi.type_code_), mvi.count_);
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
  } else {

    if (out_type == output_type::CSV_FORMATIONS) {
      csv << "rbf_uuid,used,category,train_nr,start_station_eva,start_station_"
             "name,start_time,sections,vehicles,vehicles_found,all_vehicles_"
             "found,vehicle_groups\n";
      for (auto const& [trip_uuid, tf] :
           uv.capacity_maps_.trip_formation_map_) {
        auto const is_used = used_tfs.find(&tf) != end(used_tfs);
        auto const& st = sched.stations_.at(tf.ptid_.get_station_id());
        auto const dep_time =
            tf.ptid_.get_time() != INVALID_TIME
                ? motis_to_unixtime(sched.schedule_begin_, tf.ptid_.get_time())
                : static_cast<std::time_t>(0);

        mcd::hash_set<std::uint64_t> uics{};
        mcd::hash_set<std::string_view> vehicle_groups{};
        std::string vehicle_groups_str;
        for (auto const& sec : tf.sections_) {
          for (auto const& vi : sec.vehicles_) {
            uics.insert(vi.uic_);
          }
          for (auto const& vg : sec.vehicle_groups_) {
            if (auto inserted = vehicle_groups.insert(vg.name_.view());
                inserted.second) {
              if (!vehicle_groups_str.empty()) {
                vehicle_groups_str += "; ";
              }
              vehicle_groups_str += vg.name_.view();
            }
          }
        }
        auto vehicles_found = 0U;
        for (auto const uic : uics) {
          if (auto const it = caps.vehicle_capacity_map_.find(uic);
              it != end(caps.vehicle_capacity_map_)) {
            ++vehicles_found;
          }
        }

        csv << trip_uuid << "," << is_used << ","
            << std::quoted(tf.category_.view()) << ","
            << tf.ptid_.get_train_nr() << "," << std::quoted(st->eva_nr_.view())
            << "," << std::quoted(st->name_.view()) << ","
            << format_unix_time(dep_time) << "," << tf.sections_.size() << ","
            << uics.size() << "," << vehicles_found << ","
            << (uics.size() == vehicles_found) << ","
            << std::quoted(vehicle_groups_str) << "\n";
      }
    }

    mc.create_and_finish(
        MsgContent_HTTPResponse,
        CreateHTTPResponse(
            mc, HTTPStatus_OK,
            mc.CreateVector(std::vector<Offset<HTTPHeader>>{
                CreateHTTPHeader(mc, mc.CreateString("Content-Type"),
                                 mc.CreateString("text/csv"))}),
            mc.CreateString(csv.view()))
            .Union());
  }

  return make_msg(mc);
}

}  // namespace motis::paxmon::api
