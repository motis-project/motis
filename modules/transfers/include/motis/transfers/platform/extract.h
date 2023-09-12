#pragma once

#include <filesystem>
#include <tuple>

#include "motis/transfers/platform/platform.h"

namespace motis::transfers {

using filter_rule_description = std::tuple<bool, std::string, std::string>;
auto const filter_rule_descriptions = std::vector<filter_rule_description>{
    {true, "public_transport", "platform"},
    {true, "public_transport", "stop_position"},
    {true, "railway", "platform"},
    {true, "railway", "tram_stop"},
};

auto const platform_name_keys = std::vector<std::string>{
    "name", "description", "ref_name", "local_ref", "ref",
};

platforms extract_platforms_from_osm_file(std::filesystem::path const&);

}  // namespace motis::transfers
