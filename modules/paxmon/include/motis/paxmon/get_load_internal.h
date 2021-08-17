#pragma once

#include "motis/paxmon/get_load.h"

namespace motis::paxmon {

pax_pdf get_load_pdf_base(passenger_group_container const& pgc,
                          pci_groups groups);

void add_additional_groups_base(
    pax_pdf& pdf, std::vector<std::pair<passenger_group const*, float>> const&
                      additional_groups);

#ifdef MOTIS_AVX2

pax_pdf get_load_pdf_avx(passenger_group_container const& pgc,
                         pci_groups groups);

void add_additional_groups_avx(
    pax_pdf& pdf, std::vector<std::pair<passenger_group const*, float>> const&
                      additional_groups);

#endif

}  // namespace motis::paxmon
