#pragma once

#include <map>
#include <memory>
#include <vector>

#include "boost/icl/interval_set.hpp"

namespace motis {

template <typename T, typename Comparator = std::less<T>,
          typename IntType = unsigned>
class interval_map {
public:
  struct range {
    range() = default;
    range(IntType from, IntType to) : from_(from), to_(to) {}
    IntType from_, to_;
  };

  using interval = typename boost::icl::interval<IntType>::type;

  void add_entry(T entry, IntType index) {
    attributes_[entry] += interval(index, index + 1);
  }

  void add_entry(T entry, IntType from_index, IntType to_index) {
    attributes_[entry] += interval(from_index, to_index + 1);
  }

  std::map<T, std::vector<range>, Comparator> get_attribute_ranges() {
    std::map<T, std::vector<range>, Comparator> result;
    for (auto const& [key, ranges] : attributes_) {
      result[key].reserve(ranges.size());
      for (auto const& r : ranges) {
        result[key].emplace_back(r.lower(), r.upper() - 1);
      }
    }
    return result;
  }

private:
  std::map<T, boost::icl::interval_set<IntType>, Comparator> attributes_;
};

}  // namespace motis
