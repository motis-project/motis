#pragma once

#include <cmath>
#include <memory>
#include <optional>

#include "cista/indexed.h"

#include "motis/array.h"
#include "motis/hash_map.h"
#include "motis/memory.h"
#include "motis/string.h"
#include "motis/vector.h"

#include "motis/core/schedule/connection.h"
#include "motis/core/schedule/footpath.h"
#include "motis/core/schedule/timezone.h"

namespace motis {

constexpr auto const NO_SOURCE_SCHEDULE = std::numeric_limits<uint32_t>::max();

struct station {
  double lat() const { return width_; }
  double lng() const { return length_; }

  std::optional<uint16_t> get_platform(uint16_t track) const {
    auto const it = track_to_platform_.find(track);
    if (it == end(track_to_platform_)) {
      return std::nullopt;
    } else {
      return {it->second};
    }
  }

  int32_t get_transfer_time_between_tracks(uint16_t from_track,
                                           uint16_t to_track) const {
    return get_transfer_time_between_platforms(get_platform(from_track),
                                               get_platform(to_track));
  }

  int32_t get_transfer_time_between_platforms(
      std::optional<uint16_t> from_platform,
      std::optional<uint16_t> to_platform) const {
    return from_platform.has_value() && from_platform == to_platform
               ? platform_transfer_time_
               : transfer_time_;
  }

  uint32_t index_{0};
  double length_{0.0}, width_{0.0};
  int32_t transfer_time_{0};
  int32_t platform_transfer_time_{};
  mcd::array<uint64_t, static_cast<service_class_t>(service_class::NUM_CLASSES)>
      arr_class_events_{{0}}, dep_class_events_{{0}};
  mcd::string eva_nr_;
  cista::indexed<mcd::string> name_;
  ptr<timezone const> timez_{nullptr};
  mcd::vector<ptr<station>> equivalent_;
  mcd::vector<footpath> outgoing_footpaths_;
  mcd::vector<footpath> incoming_footpaths_;
  uint32_t source_schedule_{NO_SOURCE_SCHEDULE};
  mcd::hash_map<uint16_t, uint16_t> track_to_platform_;
  mcd::vector<mcd::string> external_ids_;
  bool dummy_{false};
};

using station_ptr = mcd::unique_ptr<station>;

}  // namespace motis
