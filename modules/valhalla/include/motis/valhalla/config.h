#pragma once

#include <string>

#include "boost/property_tree/ptree.hpp"

namespace motis::valhalla {

boost::property_tree::ptree get_config(std::string const& tile_dir);

}  // namespace motis::valhalla