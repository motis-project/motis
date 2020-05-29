#pragma once

#include <map>
#include <string>

#include "utl/parser/cstr.h"

#include "motis/loader/gtfs/stop.h"
#include "motis/loader/loaded_file.h"

namespace motis::loader::gtfs {

struct transfer {
  enum {
    RECOMMENDED_TRANSFER,
    TIMED_TRANSFER,
    MIN_TRANSFER_TIME,
    NOT_POSSIBLE,
    GENERATED
  };

  transfer() = default;
  transfer(int minutes, int type) : minutes_(minutes), type_(type) {}

  int minutes_;
  int type_;
};

using stop_pair = std::pair<stop const*, stop const*>;
std::map<stop_pair, transfer> read_transfers(loaded_file, stop_map const&);

}  // namespace motis::loader::gtfs
