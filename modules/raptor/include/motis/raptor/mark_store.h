#pragma once

namespace motis::raptor {

using mark_index = uint32_t;

struct cpu_mark_store {
  explicit cpu_mark_store(mark_index const size) : marks_(size, false) {}

  void mark(mark_index const index) { marks_[index] = true; }
  bool marked(mark_index const index) const { return marks_[index]; }
  void reset() { std::fill(std::begin(marks_), std::end(marks_), false); }

private:
  std::vector<bool> marks_;
};

}  // namespace motis::raptor