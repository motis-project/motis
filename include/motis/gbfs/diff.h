#pragma once

#include <functional>

namespace motis::gbfs {

template <typename Collection1, typename Collection2>
void diff(
    const Collection1& old_collection,
    const Collection2& new_collection,
    std::function<void(const typename Collection1::value_type&)>&& removed,
    std::function<void(const typename Collection2::value_type&)>&& added = {},
    std::function<void(const typename Collection1::value_type&,
                       const typename Collection2::value_type&)>&& in_both =
        {}) {
  auto it1 = old_collection.begin();
  auto it2 = new_collection.begin();
  const auto end1 = old_collection.end();
  const auto end2 = new_collection.end();

  while (it1 != end1 && it2 != end2) {
    if (*it1 == *it2) {
      if (in_both) {
        in_both(*it1, *it2);
      }
      ++it1;
      ++it2;
    } else if (*it1 < *it2) {
      removed(*it1);
      ++it1;
    } else {
      if (added) {
        added(*it2);
      }
      ++it2;
    }
  }

  while (it1 != end1) {
    removed(*it1);
    ++it1;
  }

  if (added) {
    while (it2 != end2) {
      added(*it2);
      ++it2;
    }
  }
}

}  // namespace motis::gbfs
