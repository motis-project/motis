#pragma once

namespace motis::revise {

enum class section_type { TRIP, WALK };
struct section {
  section(int from, int to, section_type type)
      : from_(from), to_(to), type_(type) {}
  int from_, to_;
  section_type type_;
};

}  // namespace motis::revise
