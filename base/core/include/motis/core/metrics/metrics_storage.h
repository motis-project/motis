#pragma once

#include <array>
#include <optional>

#include "motis/core/common/unixtime.h"

namespace motis {

template <typename T, std::size_t Entries = 60 * 24>
struct metrics_storage {
  T* at(unixtime const ts) {
    auto const mins = static_cast<std::uint64_t>(ts / 60);
    if (mins < start_mins_) {
      return nullptr;
    } else {
      if (start_mins_ == 0) {
        start_mins_ = mins;
      } else {
        while (mins >= start_mins_ + Entries) {
          data_[start_index_] = {};
          start_index_ = (start_index_ + 1) % Entries;
          ++start_mins_;
        }
      }
      return &data_[(start_index_ + mins - start_mins_) % Entries];
    }
  }

  T const* at(unixtime const ts) const {
    auto const mins = static_cast<std::uint64_t>(ts / 60);
    if (mins >= start_mins_ && mins < start_mins_ + Entries) {
      return &data_[(start_index_ + mins - start_mins_) % Entries];
    } else {
      return nullptr;
    }
  }

  unixtime start_time() const { return start_mins_ * 60; }
  unixtime end_time() const { return (start_mins_ + Entries - 1) * 60; }
  constexpr std::size_t size() const { return Entries; }

  std::array<T, Entries> data_{};
  std::size_t start_index_{};
  std::uint64_t start_mins_{};
};

}  // namespace motis
