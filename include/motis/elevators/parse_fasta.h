#pragma once

#include <filesystem>
#include <string_view>

#include "boost/json/object.hpp"

#include "motis/types.h"

namespace motis {

std::vector<nigiri::interval<nigiri::unixtime_t>> parse_out_of_service(
    boost::json::object const&);
vector_map<elevator_idx_t, elevator> parse_fasta(std::string_view);
vector_map<elevator_idx_t, elevator> parse_fasta(std::filesystem::path const&);

}  // namespace motis