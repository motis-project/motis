#pragma once

#include <cassert>
#include <mutex>

#include "utl/verify.h"

#include "motis/data.h"
#include "motis/vector.h"

#include "motis/core/common/dynamic_fws_multimap.h"

#include "motis/paxmon/passenger_group.h"

namespace motis::paxmon {

using pci_index = std::uint32_t;

using pci_groups = dynamic_fws_multimap<passenger_group_index>::const_bucket;
using mutable_pci_groups =
    dynamic_fws_multimap<passenger_group_index>::mutable_bucket;

struct pci_container {
  pci_container() = default;
  ~pci_container() = default;

  pci_container(pci_container const& c)
      : groups_{c.groups_}, expected_load_{c.expected_load_} {}

  pci_container(pci_container&& c) noexcept
      : groups_{std::move(c.groups_)},
        expected_load_{std::move(c.expected_load_)} {}

  pci_container& operator=(pci_container const& c) {
    if (this != &c) {
      groups_ = c.groups_;
      expected_load_ = c.expected_load_;
    }
    return *this;
  }

  pci_container& operator=(pci_container&& c) noexcept {
    groups_ = std::move(c.groups_);
    expected_load_ = std::move(c.expected_load_);
    return *this;
  }

  pci_index insert() {
    auto const idx = static_cast<pci_index>(expected_load_.size());
    expected_load_.push_back(0U);
    groups_[idx];
    return idx;
  }

  void init_expected_load(passenger_group_container const& pgc,
                          pci_index const idx) {
    auto expected = std::uint16_t{};
    for (auto const grp_id : groups_[idx]) {
      auto const* grp = pgc[grp_id];
      if (is_planned_group(grp)) {
        expected += grp->passengers_;
      }
    }
    expected_load_[idx] = expected;
  }

  pci_groups groups(pci_index const idx) const { return groups_[idx]; }

  mutable_pci_groups groups(pci_index const idx) { return groups_[idx]; }

  std::mutex& mutex(pci_index const /*idx*/) { return mutex_; }

  std::size_t size() const { return expected_load_.size(); }

  [[nodiscard]] bool empty() const { return expected_load_.empty(); }

  std::size_t allocated_size() const {
    return groups_.allocated_size() +
           expected_load_.allocated_size_ * sizeof(std::uint16_t);
  }

  dynamic_fws_multimap<passenger_group_index> groups_;
  mcd::vector<std::uint16_t> expected_load_;
  std::mutex mutex_;
};

}  // namespace motis::paxmon
