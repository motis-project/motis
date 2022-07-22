#include "motis/paxmon/universe_info.h"

#include "motis/paxmon/multiverse.h"

namespace motis::paxmon {

universe_info::~universe_info() {
  if (auto mv = multiverse_.lock(); mv) {
    mv->release_universe(*this);
  }
}

}  // namespace motis::paxmon
