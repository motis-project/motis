#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "boost/range/irange.hpp"

#include "motis/footpaths/platforms.h"
#include "motis/footpaths/state.h"

#include "nigiri/location.h"
#include "nigiri/types.h"

namespace motis::footpaths {

struct matching_result {
  platform* pf_;
  nigiri::location_idx_t nloc_idx_;
  geo::latlng nloc_pos_;
};
using matching_results = std::vector<matching_result>;

// -- matching helper --
using matches_func =
    std::function<bool(const platform*, const nigiri::location&)>;
std::pair<bool, matching_result> matching(
    nigiri::location const&, state const& /* old_state */,
    state const& /* update_state */, boost::strided_integer_range<int> const&,
    int const match_bus_stop_max_distance, matches_func const&);

// -- match functions --
std::pair<bool, matching_result> match_by_name(
    nigiri::location const&, state const& /* old_state */,
    state const& /* update_state */, boost::strided_integer_range<int> const&,
    int const match_bus_stop_max_distance);

std::pair<bool, matching_result> match_by_distance(
    nigiri::location const&, state const& /* old_state */,
    state const& /* update_state */, int const r,
    int const match_bus_stop_max_distance);

// -- helper functions --
std::string remove_special_characters(std::string const&);

/**
 * Searches the first consecutive sequence of numbers in a string.
 */
std::string get_first_number_sequence(std::string const&);

std::string get_all_numbers(std::string const&);

// -- platform/location name matcher --
bool name_match(platform const*, nigiri::location const&);
bool exact_str_match(std::string&, std::string&);

bool first_number_match(platform const*, nigiri::location const&);
bool exact_first_number_match(std::string&, std::string&);

bool number_match(platform const*, nigiri::location const&);
bool exact_number_match(std::string&, std::string&);

}  // namespace motis::footpaths
