#pragma once

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"

namespace motis::odm {

std::vector<nigiri::routing::journey> from_csv(std::string_view);

nigiri::pareto_set<nigiri::routing::journey> separate_pt(
    std::vector<nigiri::routing::journey>&);

std::string to_csv(nigiri::routing::journey const&);

std::string to_csv(std::vector<nigiri::routing::journey> const&);

nigiri::routing::journey make_odm_direct(nigiri::location_idx_t from,
                                         nigiri::location_idx_t to,
                                         nigiri::unixtime_t departure,
                                         nigiri::unixtime_t arrival);

}  // namespace motis::odm