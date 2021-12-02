#include "motis/raptor/cpu/mark_store.h"

namespace motis::raptor {

cpu_mark_store::cpu_mark_store(mark_index const size) : marks_(size, false) {}

void cpu_mark_store::mark(mark_index const index) { marks_[index] = true; }

bool cpu_mark_store::marked(mark_index const index) const {
  return marks_[index];
}

void cpu_mark_store::reset() {
  std::fill(std::begin(marks_), std::end(marks_), false);
}

}  // namespace motis::raptor
