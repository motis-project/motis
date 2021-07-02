#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <mutex>
#include <type_traits>

#include "motis/hash_set.h"

#include "motis/paxmon/passenger_group.h"

namespace motis::paxmon {

struct pax_connection_info {
  pax_connection_info() = default;
  ~pax_connection_info() = default;

  pax_connection_info(std::initializer_list<passenger_group_index> groups)
      : groups_{groups} {}

  template <typename InputIt>
  pax_connection_info(InputIt first, InputIt last) {
    if constexpr (std::is_same_v<
                      typename std::iterator_traits<InputIt>::value_type,
                      passenger_group>) {
      for (; first != last; ++first) {
        groups_.insert(first->id_);
      }
    } else {
      groups_.insert(first, last);
    }
  }

  pax_connection_info(pax_connection_info const& pci)
      : groups_{pci.groups_}, expected_load_{pci.expected_load_} {}

  pax_connection_info(pax_connection_info&& pci) noexcept
      : groups_{std::move(pci.groups_)}, expected_load_{pci.expected_load_} {}

  pax_connection_info& operator=(pax_connection_info const& pci) {
    if (this != &pci) {
      groups_ = pci.groups_;
      expected_load_ = pci.expected_load_;
    }
    return *this;
  }

  pax_connection_info& operator=(pax_connection_info&& pci) noexcept {
    groups_ = std::move(pci.groups_);
    expected_load_ = pci.expected_load_;
    return *this;
  }

  void init_expected_load(passenger_group_container const& pgc) {
    expected_load_ = 0;
    for (auto const grp_id : groups_) {
      auto const* grp = pgc[grp_id];
      if (is_planned_group(grp)) {
        expected_load_ += grp->passengers_;
      }
    }
  }

  mcd::hash_set<passenger_group_index> groups_;
  std::mutex mutex_;
  std::uint16_t expected_load_{};
};

}  // namespace motis::paxmon
