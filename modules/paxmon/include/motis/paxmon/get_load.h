#pragma once

#include <map>

#include "motis/paxmon/graph.h"

namespace motis::paxmon {

using pdf_t = std::map<std::uint16_t, float>;
using cdf_t = std::map<std::uint16_t, float>;

std::uint16_t get_base_load(pax_connection_info const& pci);

pdf_t get_load_pdf(pax_connection_info const& pci);
cdf_t get_cdf(pdf_t const& pdf);
cdf_t get_load_cdf(pax_connection_info const& pci);

template <typename T>
std::map<float, float> to_load_factor(T const& df, std::uint16_t capacity) {
  auto lfdf = std::map<float, float>{};
  for (auto const& e : df) {
    lfdf[e.first / static_cast<float>(capacity)] = e.second;
  }
  return lfdf;
}

}  // namespace motis::paxmon
