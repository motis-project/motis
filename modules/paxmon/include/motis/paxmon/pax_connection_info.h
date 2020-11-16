#pragma once

#include <initializer_list>
#include <mutex>

#include "motis/hash_set.h"

#include "motis/paxmon/passenger_group.h"

namespace motis::paxmon {

struct pax_connection_info {
  pax_connection_info() = default;
  ~pax_connection_info() = default;

  pax_connection_info(std::initializer_list<passenger_group*> groups)
      : groups_{groups} {}

  template <typename InputIt>
  pax_connection_info(InputIt first, InputIt last) {
    groups_.insert(first, last);
  }

  pax_connection_info(pax_connection_info const& pci) : groups_{pci.groups_} {}

  pax_connection_info(pax_connection_info&& pci) noexcept
      : groups_{std::move(pci.groups_)} {}

  pax_connection_info& operator=(pax_connection_info const& pci) {
    if (this != &pci) {
      groups_ = pci.groups_;
    }
    return *this;
  }

  pax_connection_info& operator=(pax_connection_info&& pci) noexcept {
    groups_ = std::move(pci.groups_);
    return *this;
  }

  mcd::hash_set<passenger_group*> groups_;
  std::mutex mutex_;
};

}  // namespace motis::paxmon
