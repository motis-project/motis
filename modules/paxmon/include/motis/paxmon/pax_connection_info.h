#pragma once

#include <mutex>
#include <vector>

#include "motis/paxmon/passenger_group.h"

namespace motis::paxmon {

struct pax_section_info {
  explicit pax_section_info(passenger_group* group) : group_{group} {}

  passenger_group* group_{};
};

inline bool operator==(pax_section_info const& lhs,
                       pax_section_info const& rhs) {
  return lhs.group_ == rhs.group_;
}

inline bool operator!=(pax_section_info const& lhs,
                       pax_section_info const& rhs) {
  return lhs.group_ != rhs.group_;
}

struct pax_connection_info {
  pax_connection_info() = default;

  explicit pax_connection_info(std::vector<pax_section_info>&& psi)
      : section_infos_{std::move(psi)} {}
  pax_connection_info(pax_connection_info const& pci)
      : section_infos_{pci.section_infos_} {}
  pax_connection_info(pax_connection_info&& pci) noexcept
      : section_infos_{std::move(pci.section_infos_)} {}

  pax_connection_info& operator=(pax_connection_info const& pci) {
    section_infos_ = pci.section_infos_;
    return *this;
  }

  pax_connection_info& operator=(pax_connection_info&& pci) noexcept {
    section_infos_ = std::move(pci.section_infos_);
    return *this;
  }

  std::vector<pax_section_info> section_infos_;
  std::mutex mutex_;
};

}  // namespace motis::paxmon
