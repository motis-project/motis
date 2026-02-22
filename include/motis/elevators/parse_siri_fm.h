#pragma once

#include <filesystem>
#include <string_view>

#include "motis/types.h"

namespace motis {

vector_map<elevator_idx_t, elevator> parse_siri_fm(std::string_view);
vector_map<elevator_idx_t, elevator> parse_siri_fm(
    std::filesystem::path const&);

}  // namespace motis
