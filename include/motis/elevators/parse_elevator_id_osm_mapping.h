#pragma once

#include <cinttypes>
#include <filesystem>
#include <string_view>

#include "motis/types.h"

namespace motis {

using elevator_id_osm_mapping_t = hash_map<std::uint64_t, std::string>;

elevator_id_osm_mapping_t parse_elevator_id_osm_mapping(std::string_view);
elevator_id_osm_mapping_t parse_elevator_id_osm_mapping(
    std::filesystem::path const&);

}  // namespace motis
