#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "boost/range/irange.hpp"

#include "motis/footpaths/platforms.h"

#include "nigiri/location.h"
#include "nigiri/types.h"

namespace motis::footpaths {

struct matching_result {
  platform* pf_;
  nigiri::location_idx_t loc_idx_;
};

// -- matching helper --
typedef std::function<bool(platform const*, nigiri::location const&)>
    matches_func;
std::pair<bool, matching_result> matching(
    nigiri::location const&, motis::footpaths::platforms_index*,
    boost::strided_integer_range<int> const&,
    int const match_bus_stop_max_distance, matches_func const&);

// -- match functions --
std::pair<bool, matching_result> match_by_name(
    nigiri::location const&, motis::footpaths::platforms_index*,
    boost::strided_integer_range<int> const&,
    int const match_bus_stop_max_distance);

// -- helper functions --
std::string remove_special_characters(std::string const&);

/**
 * Searches the first consecutive sequence of numbers in a string.
 */
std::string get_first_number_sequence(std::string const&);

// -- platform/location name matcher --
bool name_match(platform const*, nigiri::location const&);
bool exact_str_match(std::string&, std::string&);

bool first_number_match(platform const*, nigiri::location const&);
bool exact_first_number_match(std::string&, std::string&);

}  // namespace motis::footpaths
