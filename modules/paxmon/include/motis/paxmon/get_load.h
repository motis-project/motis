#pragma once

#include <map>
#include <utility>
#include <vector>

#include "cista/reflection/comparable.h"

#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/passenger_group_container.h"
#include "motis/paxmon/pci_container.h"
#include "motis/paxmon/universe.h"

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

pax_limits get_pax_limits(passenger_group_container const& pgc,
                          pci_groups groups);
std::uint16_t get_base_load(passenger_group_container const& pgc,
                            pci_groups groups);
std::uint16_t get_expected_load(pci_container const& pcis, pci_index idx);
std::uint16_t get_expected_load(universe const& uv, pci_index idx);

pax_pdf get_load_pdf(passenger_group_container const& pgc, pci_groups groups);
pax_cdf get_cdf(pax_pdf const& pdf);
pax_cdf get_load_cdf(passenger_group_container const& pgc, pci_groups groups);

std::uint16_t get_mean_load(passenger_group_container const& pgc,
                            pci_groups groups);
std::uint16_t get_pax_quantile(pax_cdf const& cdf, float q);
std::uint16_t get_median_load(pax_cdf const& cdf);

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
