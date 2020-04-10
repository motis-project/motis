#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"

#include "motis/intermodal/eval/parse_bbox.h"

namespace motis::intermodal::eval {

std::unique_ptr<bbox> parse_bbox(std::string const& input) {
  std::vector<std::string> coords;
  boost::split(coords, input, boost::is_any_of(","));
  if (coords.size() != 4) {
    return nullptr;
  }

  try {
    auto const lon_min = boost::lexical_cast<double>(coords[0]);
    auto const lat_min = boost::lexical_cast<double>(coords[1]);
    auto const lon_max = boost::lexical_cast<double>(coords[2]);
    auto const lat_max = boost::lexical_cast<double>(coords[3]);
    auto const min_corner = geo::latlng(lat_min, lon_min);
    auto const max_corner = geo::latlng(lat_max, lon_max);
    return std::make_unique<bbox>(
        boost::geometry::model::box<geo::latlng>(min_corner, max_corner));
  } catch (boost::bad_lexical_cast const&) {
    return nullptr;
  }
}

}  // namespace motis::intermodal::eval
