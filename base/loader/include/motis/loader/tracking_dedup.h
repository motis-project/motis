#pragma once

#include <vector>

#include "boost/sort/block_indirect_sort/block_indirect_sort.hpp"

#include "motis/core/common/logging.h"

namespace motis {

template <typename ForwardIterator, typename BinaryPredicate>
std::pair<std::vector<size_t>, ForwardIterator> tracking_unique(
    ForwardIterator begin_it, ForwardIterator end_it, BinaryPredicate p) {
  logging::scoped_timer timer("tracking_unique");
  std::vector<size_t> map(static_cast<size_t>(std::distance(begin_it, end_it)));
  auto map_it = begin(map);
  auto insert_it = begin_it;
  auto prev_it = begin_it;
  for (auto cur_it = std::next(begin_it); cur_it != end_it; ++cur_it) {
    if (!p(*cur_it, *insert_it)) {
      auto skipped = static_cast<size_t>(std::distance(prev_it, cur_it));
      std::fill(map_it, std::next(map_it, skipped),
                std::distance(begin_it, insert_it));
      std::advance(map_it, skipped);
      ++insert_it;
      *(insert_it) = std::move(*cur_it);
      prev_it = cur_it;
    }
  }

  std::fill(map_it, end(map), std::distance(begin_it, insert_it));

  return {map, ++insert_it};
}

template <typename T>
void apply_permutation(std::vector<T>& data, std::vector<size_t>& perm) {
  logging::scoped_timer timer("apply_permutation");
  std::vector<bool> swapped(data.size(), false);
  auto i = 0u;
  while (true) {
    auto const& j = perm[i];
    swapped[i] = true;
    if (!swapped[j]) {
      std::swap(data[i], data[j]);
      i = j;
    } else {
      auto const& next_i = std::find(begin(swapped), end(swapped), false);
      if (next_i == end(swapped)) {
        break;
      }
      i = std::distance(begin(swapped), next_i);
    }
  }
}

template <typename T, typename BinaryPredicate>
std::vector<size_t> tracking_dedupe(std::vector<T>& data, BinaryPredicate&& eq,
                                    BinaryPredicate&& lt) {
  // unique pass 1
  std::vector<size_t> unique_map1;
  auto new_end1 = data.begin();
  std::tie(unique_map1, new_end1) =
      tracking_unique(data.begin(), data.end(), eq);
  data.erase(new_end1, data.end());

  // sort permutation
  std::vector<size_t> perm(data.size());
  std::iota(begin(perm), end(perm), 0);
  boost::sort::block_indirect_sort(begin(perm), end(perm), lt);

  // apply permutation to data
  apply_permutation(data, perm);

  // invert permutation
  auto inverse_perm = std::vector<size_t>(data.size());
  for (auto i = 0u; i < inverse_perm.size(); ++i) {
    inverse_perm[perm[i]] = i;
  }

  // unique pass 2
  std::vector<size_t> unique_map2(data.size());
  auto new_end2 = data.begin();
  std::tie(unique_map2, new_end2) =
      tracking_unique(data.begin(), data.end(), eq);
  data.erase(new_end2, data.end());

  // make index
  auto idx_map = std::vector<size_t>(unique_map1.size());
  for (auto i = 0u; i < idx_map.size(); ++i) {
    idx_map[i] = unique_map2[inverse_perm[unique_map1[i]]];
  }

  return idx_map;
}
}  // namespace motis
