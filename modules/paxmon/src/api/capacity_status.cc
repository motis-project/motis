#include "motis/paxmon/api/capacity_status.h"

#include <cstdint>
#include <algorithm>
#include <optional>
#include <utility>

#include "utl/to_vec.h"

#include "motis/hash_map.h"

#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"

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
  Offset<PaxMonCapacityStats> to_fbs(FlatBufferBuilder& fbb) const {
    return CreatePaxMonCapacityStats(fbb, tracked_, trip_formation_,
                                     capacity_for_all_sections_,
                                     capacity_for_some_sections_);
  }

  void add(trip_capacity_status const& status) {
    ++tracked_;
    if (status.has_trip_formation_) {
      ++trip_formation_;
    }
    if (status.has_capacity_for_all_sections_) {
      ++capacity_for_all_sections_;
    }
    if (status.has_capacity_for_some_sections_) {
      ++capacity_for_some_sections_;
    }
  }

  std::uint32_t tracked_{};
  std::uint32_t trip_formation_{};
  std::uint32_t capacity_for_all_sections_{};
  std::uint32_t capacity_for_some_sections_{};
};

struct provider_stats {
  Offset<PaxMonProviderCapacityStats> to_fbs(
      FlatBufferBuilder& fbb, schedule const& sched,
      std::string_view const provider_name) const {
    auto vec = utl::to_vec(by_category_, [&fbb, &sched](auto const& entry) {
      auto const& name = entry.first;
      auto const& stats = entry.second;
      auto clasz = service_class::OTHER;
      if (auto const it = sched.classes_.find(name);
          it != end(sched.classes_)) {
        clasz = it->second;
      }
      return CreatePaxMonCategoryCapacityStats(fbb, fbb.CreateString(name),
                                               static_cast<std::uint8_t>(clasz),
                                               stats.to_fbs(fbb));
    });
    return CreatePaxMonProviderCapacityStats(
        fbb, fbb.CreateString(provider_name),
        CreatePaxMonProviderInfo(
            fbb,
            fbb.CreateString(provider_ != nullptr
                                 ? provider_->short_name_.view()
                                 : std::string_view{}),
            fbb.CreateString(provider_ != nullptr ? provider_->long_name_.view()
                                                  : std::string_view{}),
            fbb.CreateString(provider_ != nullptr ? provider_->full_name_.view()
                                                  : std::string_view{})),
        stats_.to_fbs(fbb), fbb.CreateVectorOfSortedTables(&vec));
  }

  provider const* provider_{};
  capacity_stats stats_{};
  mcd::hash_map<std::string_view, capacity_stats> by_category_;
};

}  // namespace

msg_ptr capacity_status(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonCapacityStatusRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto const& uv = uv_access.uv_;

  auto const filter_by_time = req->filter_by_time();
  auto const filter_interval_begin =
      unix_to_motistime(sched.schedule_begin_, req->filter_interval()->begin());
  auto const filter_interval_end =
      unix_to_motistime(sched.schedule_begin_, req->filter_interval()->end());
  auto const time_filter_active =
      filter_by_time != PaxMonFilterTripsTimeFilter_NoFilter;

  message_creator mc;

  auto all_trips = capacity_stats{};
  auto by_provider = mcd::hash_map<std::string_view, provider_stats>{};

  for (auto const& [trp_idx, tdi] : uv.trip_data_.mapping_) {
    auto const* trp = get_trip(sched, trp_idx);

    if (time_filter_active &&
        !include_trip_based_on_time_filter(
            trp, filter_by_time, filter_interval_begin, filter_interval_end)) {
      continue;
    }

    auto const sections = access::sections{trp};
    if (sections.size() == 0) {
      continue;
    }
    auto const& tcs = uv.trip_data_.capacity_status(tdi);

    auto const& con_info = (*sections.begin()).lcon().full_con_->con_info_;
    auto const category_name =
        sched.categories_.at(con_info->family_)->name_.view();
    auto const provider_name = con_info->provider_ != nullptr
                                   ? con_info->provider_->full_name_.view()
                                   : std::string_view{};

    all_trips.add(tcs);

    auto& provider = by_provider[provider_name];
    if (provider.provider_ == nullptr && con_info->provider_ != nullptr) {
      provider.provider_ = con_info->provider_;
    }
    provider.stats_.add(tcs);
    provider.by_category_[category_name].add(tcs);
  }

  auto by_provider_fbs = utl::to_vec(by_provider, [&](auto const& entry) {
    auto const& name = entry.first;
    auto const& stats = entry.second;
    return stats.to_fbs(mc, sched, name);
  });

  mc.create_and_finish(MsgContent_PaxMonCapacityStatusResponse,
                       CreatePaxMonCapacityStatusResponse(
                           mc, all_trips.to_fbs(mc),
                           mc.CreateVectorOfSortedTables(&by_provider_fbs))
                           .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
