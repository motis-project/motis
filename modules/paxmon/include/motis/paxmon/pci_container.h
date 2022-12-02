#pragma once

#include <cassert>
#include <cmath>
#include <mutex>

#include "utl/verify.h"

#include "motis/data.h"
#include "motis/vector.h"

#include "motis/core/common/dynamic_fws_multimap.h"

#include "motis/paxmon/passenger_group.h"

namespace motis::paxmon {

using pci_index = std::uint32_t;

using pci_group_routes =
    dynamic_fws_multimap<passenger_group_with_route>::const_bucket;
using mutable_pci_group_routes =
    dynamic_fws_multimap<passenger_group_with_route>::mutable_bucket;

struct pci_container {
  pci_container() = default;
  ~pci_container() = default;

  pci_container(pci_container const& c)
      : group_routes_{c.group_routes_},
        broken_group_routes_{c.broken_group_routes_},
        expected_load_{c.expected_load_} {}

  pci_container(pci_container&& c) noexcept
      : group_routes_{std::move(c.group_routes_)},
        broken_group_routes_{std::move(c.broken_group_routes_)},
        expected_load_{std::move(c.expected_load_)} {}

  pci_container& operator=(pci_container const& c) {
    if (this != &c) {
      group_routes_ = c.group_routes_;
      broken_group_routes_ = c.broken_group_routes_;
      expected_load_ = c.expected_load_;
    }
    return *this;
  }

  pci_container& operator=(pci_container&& c) noexcept {
    group_routes_ = std::move(c.group_routes_);
    broken_group_routes_ = std::move(c.broken_group_routes_);
    expected_load_ = std::move(c.expected_load_);
    return *this;
  }

  pci_index insert() {
    auto const idx = static_cast<pci_index>(expected_load_.size());
    expected_load_.push_back(0U);
    group_routes_[idx];
    broken_group_routes_[idx];
    return idx;
  }

  void init_expected_load(passenger_group_container const& pgc,
                          pci_index const idx) {
    auto expected = 0.F;
    for (auto const& pgwr : group_routes_[idx]) {
      auto const& gr = pgc.route(pgwr);
      if (gr.planned_) {
        expected += pgc[pgwr.pg_]->passengers_ * gr.probability_;
      }
    }
    expected_load_[idx] = static_cast<std::uint16_t>(std::round(expected));
  }

  void init_expected_load(passenger_group_container const& pgc) {
    for (auto idx = pci_index{0}; idx < size(); ++idx) {
      init_expected_load(pgc, idx);
    }
  }

  pci_group_routes group_routes(pci_index const idx) const {
    return group_routes_[idx];
  }

  mutable_pci_group_routes group_routes(pci_index const idx) {
    return group_routes_[idx];
  }

  pci_group_routes broken_group_routes(pci_index const idx) const {
    return broken_group_routes_[idx];
  }

  mutable_pci_group_routes broken_group_routes(pci_index const idx) {
    return broken_group_routes_[idx];
  }

  std::mutex& mutex(pci_index const /*idx*/) { return mutex_; }

  std::size_t size() const { return expected_load_.size(); }

  [[nodiscard]] bool empty() const { return expected_load_.empty(); }

  std::size_t allocated_size() const {
    return group_routes_.allocated_size() +
           broken_group_routes_.allocated_size() +
           expected_load_.allocated_size_ * sizeof(std::uint16_t);
  }

  dynamic_fws_multimap<passenger_group_with_route> group_routes_;
  dynamic_fws_multimap<passenger_group_with_route> broken_group_routes_;
  mcd::vector<std::uint16_t> expected_load_;
  std::mutex mutex_;
};

}  // namespace motis::paxmon
