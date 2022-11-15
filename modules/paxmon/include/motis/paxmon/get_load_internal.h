#pragma once

#include "motis/paxmon/get_load.h"

namespace motis::paxmon {

void add_additional_groups_base(
    pax_pdf& pdf, std::vector<additional_group> const& additional_groups);

#ifdef MOTIS_AVX2

void add_additional_groups_avx(
    pax_pdf& pdf, std::vector<additional_group> const& additional_groups);

#endif

}  // namespace motis::paxmon
