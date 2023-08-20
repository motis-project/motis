#include "motis/paxmon/api/detailed_capacity_status.h"

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

#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"

#include "motis/paxmon/capacity_internal.h"
#include "motis/paxmon/csv_writer.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"

#include "motis/paxmon/api/util/trip_time_filter.h"

using namespace motis::module;
using namespace motis::logging;
using namespace motis::paxmon;
using namespace flatbuffers;

namespace motis::paxmon::api {

namespace {

struct capacity_stats {
  Offset<PaxMonDetailedTripCapacityStats> to_fbs(FlatBufferBuilder& fbb) const {
    return CreatePaxMonDetailedTripCapacityStats(
        fbb, fbb.CreateString(category_), static_cast<std::uint8_t>(clasz_),
        tracked_, full_data_, partial_data_, capacity_for_all_sections_,
        trip_formation_data_found_, no_formation_data_at_all_,
        no_formation_data_some_sections_some_merged_,
        no_formation_data_some_sections_all_merged_, no_vehicles_found_at_all_,
        no_vehicles_found_some_sections_,
        some_vehicles_not_found_some_sections_, trips_using_vehicle_uics_,
        trips_using_only_vehicle_uics_, trips_using_vehicle_groups_,
        trips_using_baureihe_, trips_using_type_code_);
  }

  std::string category_;
  service_class clasz_{service_class::OTHER};

  std::uint32_t tracked_{};

  std::uint32_t full_data_{};
  std::uint32_t partial_data_{};
  std::uint32_t capacity_for_all_sections_{};

  std::uint32_t trip_formation_data_found_{};

  std::uint32_t no_formation_data_at_all_{};
  std::uint32_t no_formation_data_some_sections_some_merged_{};
  std::uint32_t no_formation_data_some_sections_all_merged_{};

  std::uint32_t no_vehicles_found_at_all_{};
  std::uint32_t no_vehicles_found_some_sections_{};
  std::uint32_t some_vehicles_not_found_some_sections_{};

  std::uint32_t trips_using_vehicle_uics_{};
  std::uint32_t trips_using_only_vehicle_uics_{};
  std::uint32_t trips_using_vehicle_groups_{};
  std::uint32_t trips_using_baureihe_{};
  std::uint32_t trips_using_type_code_{};
};

struct missing_vehicle_info {
  std::string_view baureihe_;
  std::string_view type_code_;
  std::uint32_t count_{};
};

enum class output_type { DEFAULT, CSV_TRIPS, CSV_FORMATIONS };

}  // namespace

msg_ptr detailed_capacity_status(paxmon_data& data, msg_ptr const& msg) {
  auto out_type = output_type::DEFAULT;
  auto uv_id = universe_id{};
  auto filter_by_time = PaxMonFilterTripsTimeFilter_NoFilter;
  auto filter_interval_begin = INVALID_TIME;
  auto filter_interval_end = INVALID_TIME;
  auto include_missing_vehicle_infos = false;
  auto include_uics_not_found = false;

  switch (msg->get()->content_type()) {
    case MsgContent_PaxMonDetailedCapacityStatusRequest: {
      auto const req = motis_content(PaxMonDetailedCapacityStatusRequest, msg);
      uv_id = req->universe();
      filter_by_time = req->filter_by_time();
      include_missing_vehicle_infos = req->include_missing_vehicle_infos();
      include_uics_not_found = req->include_uics_not_found();
      break;
    }
    case MsgContent_MotisNoMessage: break;
    default:
      throw std::system_error(motis::module::error::unexpected_message_type);
  }

  auto const target = msg->get()->destination()->target()->view();
  if (target.contains("trips.csv")) {
    out_type = output_type::CSV_TRIPS;
  } else if (target.contains("formations.csv")) {
    out_type = output_type::CSV_FORMATIONS;
  }

  auto const uv_access = get_universe_and_schedule(data, uv_id);
  auto const& sched = uv_access.sched_;
  auto const& uv = uv_access.uv_;
  auto const& caps = uv.capacity_maps_;

  if (msg->get()->content_type() ==
      MsgContent_PaxMonDetailedCapacityStatusRequest) {
    auto const req = motis_content(PaxMonDetailedCapacityStatusRequest, msg);
    filter_interval_begin = unix_to_motistime(sched.schedule_begin_,
                                              req->filter_interval()->begin());
    filter_interval_end =
        unix_to_motistime(sched.schedule_begin_, req->filter_interval()->end());
  }

  message_creator mc;
  string_csv_writer csv;

  auto all_trips = capacity_stats{};
  auto by_category = mcd::hash_map<mcd::string, capacity_stats>{};

  auto uics_not_found = mcd::hash_set<std::uint64_t>{};
  auto missing_vehicle_counts =
      mcd::hash_map<mcd::pair<mcd::string, mcd::string>, std::uint32_t>{};

  auto used_tfs = mcd::hash_set<trip_formation const*>{};

  if (out_type == output_type::CSV_TRIPS) {
    csv << "category"
        << "train_nr"
        << "start_station_eva"
        << "start_station_name"
        << "start_time"
        << "end_station_eva"
        << "end_station_name"
        << "end_time"
        << "provider"
        << "full_data"
        << "partial_data"
        << "has_formation"
        << "cap_all_sections"
        << "cap_some_sections"
        << "trip_sections"
        << "formation_sections"
        << "sections_all_missing_tfs"
        << "sections_some_missing_tfs"
        << "sections_all_missing_uics"
        << "sections_some_missing_uics" << end_row;
  }

  auto const time_filter_active =
      filter_by_time != PaxMonFilterTripsTimeFilter_NoFilter;

  for (auto const& [trp_idx, tdi] : uv.trip_data_.mapping_) {
    auto const* trp = get_trip(sched, trp_idx);

    if (time_filter_active &&
        !include_trip_based_on_time_filter(
            trp, filter_by_time, filter_interval_begin, filter_interval_end)) {
      continue;
    }

    auto const& tcs = uv.trip_data_.capacity_status(tdi);
    auto sections = access::sections{trp};
    auto const sec_count = sections.size();
    auto categories = mcd::hash_set<std::string_view>{};
    connection_info const* con_info = nullptr;

    auto full_data = true;
    auto partial_data = false;
    auto cap_for_all_sections = true;
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

      categories.insert(
          sched.categories_.at(lc.full_con_->con_info_->family_)->name_.view());
      if (con_info == nullptr) {
        con_info = lc.full_con_->con_info_;
      }

      auto const cap = get_capacity(sched, lc, sec.ev_key_from(),
                                    sec.ev_key_to(), caps, true);

      if (!cap.has_capacity()) {
        cap_for_all_sections = false;
      }

      auto mt_tf_found_count = 0U;
      auto uics_found_count = 0U;
      auto uics_not_found_count = 0U;

      auto const mt_count = cap.trips_.size();
      for (auto const& trp_cap : cap.trips_) {
        if (trp_cap.formation_ != nullptr) {
          used_tfs.insert(trp_cap.formation_);
        }
        if (trp_cap.has_formation()) {
          ++mt_tf_found_count;
        } else {
          full_data = false;
        }
      }

      for (auto const& vg : cap.vehicle_groups_) {
        if (vg.duplicate_group_) {
          continue;
        }

        if (vg.source_ == capacity_source::UNKNOWN) {
          full_data = false;
        } else {
          partial_data = true;
        }

        if (vg.source_ != capacity_source::FORMATION_VEHICLE_GROUPS) {
          for (auto const& v : vg.vehicles_) {
            if (v.duplicate_vehicle_) {
              continue;
            }
            if (v.source_ == capacity_source::UNKNOWN) {
              full_data = false;
              if (include_missing_vehicle_infos) {
                ++missing_vehicle_counts[mcd::pair{v.vehicle_->baureihe_,
                                                   v.vehicle_->type_code_}];
              }
            }
            auto const uic_found =
                v.source_ == capacity_source::FORMATION_VEHICLES;
            if (uic_found) {
              ++uics_found_count;
            } else if (v.vehicle_->has_uic()) {
              ++uics_not_found_count;
              if (include_uics_not_found) {
                uics_not_found.insert(v.vehicle_->uic_);
              }
            }
          }
        }
      }

      if (mt_tf_found_count == 0) {
        ++secs_with_all_missing_tfs;
      } else if (mt_tf_found_count < mt_count) {
        ++secs_with_some_missing_tfs;
      }

      if (uics_found_count == 0) {
        ++secs_with_all_missing_uics;
      } else if (uics_not_found_count != 0) {
        ++secs_with_some_missing_uics;
      }
    }  // sections

    if (out_type == output_type::CSV_TRIPS) {
      auto const category =
          con_info != nullptr
              ? sched.categories_.at(con_info->family_)->name_.view()
              : std::string_view{};
      auto const train_nr = con_info != nullptr ? con_info->train_nr_ : 0U;
      auto const& start_st =
          sched.stations_.at(trp->id_.primary_.get_station_id());
      auto const& end_st =
          sched.stations_.at(trp->id_.secondary_.target_station_id_);
      auto const start_time = motis_to_unixtime(sched.schedule_begin_,
                                                trp->id_.primary_.get_time());
      auto const end_time = motis_to_unixtime(sched.schedule_begin_,
                                              trp->id_.secondary_.target_time_);
      auto const provider =
          con_info != nullptr && con_info->provider_ != nullptr
              ? con_info->provider_->full_name_.view()
              : std::string_view{};

      csv << category << train_nr << start_st->eva_nr_.view()
          << start_st->name_.view() << format_unix_time(start_time)
          << end_st->eva_nr_.view() << end_st->name_.view()
          << format_unix_time(end_time) << provider << full_data << partial_data
          << (tf != nullptr) << tcs.has_capacity_for_all_sections_
          << tcs.has_capacity_for_some_sections_ << sec_count
          << (tf != nullptr ? tf->sections_.size() : 0U)
          << secs_with_all_missing_tfs << secs_with_some_missing_tfs
          << secs_with_all_missing_uics << secs_with_some_missing_uics
          << end_row;
    }

    auto const count = [&](auto const& fn) {
      fn(all_trips);
      for (auto const& cat : categories) {
        fn(by_category[cat]);
      }
    };

    count([](capacity_stats& stats) { ++stats.tracked_; });

    if (full_data) {
      count([](capacity_stats& stats) { ++stats.full_data_; });
    }

    if (partial_data) {
      count([](capacity_stats& stats) { ++stats.partial_data_; });
    }

    if (cap_for_all_sections) {
      count([](capacity_stats& stats) { ++stats.capacity_for_all_sections_; });
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

    auto by_category_fbs = utl::to_vec(by_category, [&](auto& entry) {
      auto const& name = entry.first;
      capacity_stats& stats = entry.second;
      stats.category_ = name.view();
      if (auto const it = sched.classes_.find(name);
          it != end(sched.classes_)) {
        stats.clasz_ = it->second;
      }
      return stats.to_fbs(mc);
    });

    mc.create_and_finish(MsgContent_PaxMonDetailedCapacityStatusResponse,
                         CreatePaxMonDetailedCapacityStatusResponse(
                             mc, all_trips.to_fbs(mc),
                             mc.CreateVectorOfSortedTables(&by_category_fbs),
                             mc.CreateVector(fbs_missing_vehicle_infos),
                             mc.CreateVector(uics_not_found_vec))
                             .Union());
  } else {

    if (out_type == output_type::CSV_FORMATIONS) {
      csv << "rbf_uuid"
          << "used"
          << "category"
          << "train_nr"
          << "start_station_eva"
          << "start_station_name"
          << "start_time"
          << "sections"
          << "vehicles"
          << "vehicles_found"
          << "all_vehicles_found"
          << "vehicle_groups" << end_row;

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
          for (auto const& vg : sec.vehicle_groups_) {
            for (auto const& vi : vg.vehicles_) {
              uics.insert(vi.uic_);
            }
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

        csv << trip_uuid << is_used << tf.category_.view()
            << tf.ptid_.get_train_nr() << st->eva_nr_.view() << st->name_.view()
            << format_unix_time(dep_time) << tf.sections_.size() << uics.size()
            << vehicles_found << (uics.size() == vehicles_found)
            << vehicle_groups_str;
      }
    }

    mc.create_and_finish(
        MsgContent_HTTPResponse,
        CreateHTTPResponse(
            mc, HTTPStatus_OK,
            mc.CreateVector(std::vector<Offset<HTTPHeader>>{
                CreateHTTPHeader(mc, mc.CreateString("Content-Type"),
                                 mc.CreateString("text/csv"))}),
            mc.CreateString(csv.str()))
            .Union());
  }

  return make_msg(mc);
}

}  // namespace motis::paxmon::api
