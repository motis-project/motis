#pragma once

#include <cstdint>
#include <initializer_list>
#include <mutex>

#include "motis/hash_set.h"

#include "motis/paxmon/passenger_group.h"

namespace motis::paxmon {

struct pax_connection_info {
  pax_connection_info() = default;
  ~pax_connection_info() = default;

  pax_connection_info(std::initializer_list<passenger_group*> groups)
      : groups_{groups} {
    init_expected_load();
  }

  template <typename InputIt>
  pax_connection_info(InputIt first, InputIt last) {
    groups_.insert(first, last);
    init_expected_load();
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

  void init_expected_load() {
    for (auto const grp : groups_) {
      if (is_planned_group(grp)) {
        expected_load_ += grp->passengers_;
      }
    }
  }

  mcd::hash_set<passenger_group*> groups_;
  std::mutex mutex_;
  std::uint16_t expected_load_{};
};

}  // namespace motis::paxmon
