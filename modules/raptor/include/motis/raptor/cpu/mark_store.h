#pragma once

#include <cinttypes>
#include <vector>

namespace motis::raptor {

using mark_index = uint32_t;

struct cpu_mark_store {
  explicit cpu_mark_store(mark_index size);

  void mark(mark_index index);
  bool marked(mark_index index) const;
  void reset();

private:
  std::vector<bool> marks_;
};

}  // namespace motis::raptor