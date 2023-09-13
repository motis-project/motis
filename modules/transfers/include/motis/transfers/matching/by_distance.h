#pragma once

#include <utility>

#include "motis/transfers/matching/matcher.h"

#include "nigiri/location.h"

namespace motis::transfers {

struct distance_matcher : public matcher {
  explicit distance_matcher(matching_data const& data,
                            matching_options const& options)
      : matcher{data, options} {}

  ~distance_matcher() override = default;

  distance_matcher(distance_matcher const&) = delete;
  distance_matcher& operator=(distance_matcher const&) = delete;

  distance_matcher(distance_matcher&&) = delete;
  distance_matcher& operator=(distance_matcher&&) = delete;

  // Matches `nigiri::location`s with platforms extracted from OSM data and
  // returns a list of valid matches. Matching is based on the distance between
  // `nigiri::location` and `platform`. The platform with the smallest distance
  // is chosen as match to the nigiri::location. Matching distances are chosen
  // from the options.
  // Returns a list of all found matches.
  matching_results matching() override;

private:
  // Matches a single `nigiri::location` and returns the result as a
  // `matching_result` struct. Returns an additional boolean value indicating
  // whether a match was found or not.
  std::pair<bool, matching_result> match_by_distance(::nigiri::location const&);
};

}  // namespace motis::transfers
