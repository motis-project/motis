#pragma once

#include "motis/paxmon/get_load.h"

namespace motis::paxmon {

pax_pdf get_load_pdf_base(pax_connection_info const& pci);

void add_additional_groups_base(
    pax_pdf& pdf, std::vector<std::pair<passenger_group const*, float>> const&
                      additional_groups);

#ifdef MOTIS_AVX2

pax_pdf get_load_pdf_avx(pax_connection_info const& pci);

void add_additional_groups_avx(
    pax_pdf& pdf, std::vector<std::pair<passenger_group const*, float>> const&
                      additional_groups);

#endif

}  // namespace motis::paxmon
