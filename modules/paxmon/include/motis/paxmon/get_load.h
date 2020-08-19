#pragma once

#include <map>

#include "motis/paxmon/graph.h"

namespace motis::paxmon {

using df_t = std::map<std::uint16_t, float>;
using pdf_t = df_t;
using cdf_t = df_t;
using lf_df_t = std::map<float, float>;

std::uint16_t get_base_load(pax_connection_info const& pci);

pdf_t get_load_pdf(pax_connection_info const& pci);
cdf_t get_cdf(pdf_t const& pdf);
cdf_t get_load_cdf(pax_connection_info const& pci);
lf_df_t to_load_factor(df_t const& df, std::uint16_t capacity);

void add_additional_group(pdf_t& pdf, std::uint16_t passengers,
                          float probability);

bool load_factor_possibly_ge(df_t const& df, std::uint16_t capacity,
                             float threshold);
bool load_factor_possibly_ge(lf_df_t const& lf_df, float threshold);

}  // namespace motis::paxmon
