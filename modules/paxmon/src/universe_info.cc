#include "motis/paxmon/universe_info.h"

#include "motis/paxmon/multiverse.h"

namespace motis::paxmon {

universe_info::~universe_info() { multiverse_.release_universe(*this); }

}  // namespace motis::paxmon
