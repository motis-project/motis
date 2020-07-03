#pragma once

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/service_access.h"

#include "motis/protocol/Connection_generated.h"

namespace motis {

inline flatbuffers::Offset<Transport> to_fbs(
    schedule const& sched, flatbuffers::FlatBufferBuilder& fbb,
    connection_info const* ci, trip const* trp, int16_t from_stop_idx = 0,
    int16_t to_stop_idx = 0) {
  auto const range = Range{from_stop_idx, to_stop_idx};
  auto const& cat_name = sched.categories_.at(ci->family_)->name_;
  auto const clasz_it = sched.classes_.find(cat_name);
  auto const clasz =
      clasz_it == end(sched.classes_) ? service_class::OTHER : clasz_it->second;
  return CreateTransport(
      fbb, &range, fbb.CreateString(cat_name), ci->family_,
      static_cast<service_class_t>(clasz),
      output_train_nr(ci->train_nr_, ci->original_train_nr_),
      fbb.CreateString(ci->line_identifier_),
      fbb.CreateString(get_service_name(sched, ci)),
      fbb.CreateString(ci->provider_ != nullptr ? ci->provider_->full_name_
                                                : ""),
      fbb.CreateString(
          ci->dir_ != nullptr
              ? *ci->dir_
              : sched.stations_.at(trp->id_.secondary_.target_station_id_)
                    ->name_));
}

}  // namespace motis
