#pragma once

#include <map>
#include <utility>
#include <vector>

#include "cista/reflection/comparable.h"

#include "motis/paxmon/graph.h"
#include "motis/paxmon/passenger_group.h"

namespace motis::paxmon {

struct pax_limits {
  CISTA_COMPARABLE()

  std::uint16_t min_{};
  std::uint16_t max_{};
};

struct pax_pdf {
  CISTA_COMPARABLE()

  std::vector<float> data_;
};

struct pax_cdf {
  CISTA_COMPARABLE()

  std::vector<float> data_;
};

using lf_df_t = std::map<float, float>;

pax_limits get_pax_limits(pax_connection_info const& pci);
std::uint16_t get_base_load(pax_connection_info const& pci);

pax_pdf get_load_pdf(pax_connection_info const& pci);
pax_cdf get_cdf(pax_pdf const& pdf);
pax_cdf get_load_cdf(pax_connection_info const& pci);

lf_df_t to_load_factor(pax_pdf const& pdf, std::uint16_t capacity);
lf_df_t to_load_factor(pax_cdf const& cdf, std::uint16_t capacity);

void add_additional_groups(
    pax_pdf& pdf, std::vector<std::pair<passenger_group const*, float>> const&
                      additional_groups);

bool load_factor_possibly_ge(pax_pdf const& pdf, std::uint16_t capacity,
                             float threshold);
bool load_factor_possibly_ge(pax_cdf const& cdf, std::uint16_t capacity,
                             float threshold);
bool load_factor_possibly_ge(lf_df_t const& lf_df, float threshold);

}  // namespace motis::paxmon
