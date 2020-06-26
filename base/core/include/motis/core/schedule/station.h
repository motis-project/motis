#pragma once

#include <cmath>
#include <memory>

#include "cista/indexed.h"

#include "motis/array.h"
#include "motis/memory.h"
#include "motis/string.h"
#include "motis/vector.h"

#include "motis/core/schedule/footpath.h"
#include "motis/core/schedule/timezone.h"

namespace motis {

struct station {
  double lat() const { return width_; }
  double lng() const { return length_; }

  uint32_t index_{0};
  double length_{0.0}, width_{0.0};
  int32_t transfer_time_{0};
  mcd::array<uint64_t, static_cast<service_class_t>(service_class::NUM_CLASSES)>
      arr_class_events_{{0}}, dep_class_events_{{0}};
  mcd::string eva_nr_;
  cista::indexed<mcd::string> name_;
  ptr<timezone const> timez_{nullptr};
  mcd::vector<ptr<station>> equivalent_;
  mcd::vector<footpath> outgoing_footpaths_;
  mcd::vector<footpath> incoming_footpaths_;
};

inline station make_station(unsigned index, double length, double width,
                            int transfer_time, mcd::string eva_nr,
                            mcd::string name, timezone const* timez) {
  station s{};
  s.index_ = index;
  s.length_ = length;
  s.width_ = width;
  s.transfer_time_ = transfer_time;
  s.eva_nr_ = std::move(eva_nr);
  s.name_ = std::move(name);
  s.timez_ = timez;
  return s;
}

using station_ptr = mcd::unique_ptr<station>;

}  // namespace motis
