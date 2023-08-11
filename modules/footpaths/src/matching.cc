#include "motis/footpaths/matching.h"

#include <limits>

#include "geo/latlng.h"

namespace n = nigiri;

namespace motis::footpaths {

std::pair<bool, matching_result> matching(
    n::location const& nloc, state const& old_state, state const& update_state,
    boost::strided_integer_range<int> const& dists,
    int const match_bus_stop_max_distance, matches_func const& matches) {
  auto mr = matching_result{};
  mr.nloc_idx_ = nloc.l_;
  mr.nloc_pos_ = nloc.pos_;

  auto matched{false};
  auto shortest_dist{std::numeric_limits<double>::max()};

  for (auto r : dists) {
    // use update_state and old_state
    auto pfs = update_state.pfs_idx_->in_radius_with_distance(nloc.pos_, r);
    auto const old_pfs =
        old_state.pfs_idx_->in_radius_with_distance(nloc.pos_, r);
    pfs.insert(pfs.end(), old_pfs.begin(), old_pfs.end());

    for (auto [dist, pf] : pfs) {
      // only match bus stops with a distance of up to a certain distance
      if (pf->info_.is_bus_stop_ && r > match_bus_stop_max_distance) {
        continue;
      }

      // only math platfoorms with a valid osm_id
      if (pf->info_.osm_id_ == -1) {
        continue;
      }

      if (!matches(pf, nloc)) {
        continue;
      }

      if (dist < shortest_dist) {
        mr.pf_ = pf;
        shortest_dist = dist;
        matched = true;
      }
    }
  }

  return {matched, mr};
}

// -- match function --
std::pair<bool, matching_result> match_by_name(
    n::location const& nloc, state const& old_state, state const& update_state,
    boost::strided_integer_range<int> const& dists,
    int const match_bus_stop_max_dist) {
  assert(nloc.type_ != n::location_type::kStation);

  auto [has_match, mr] = matching(nloc, old_state, update_state, dists,
                                  match_bus_stop_max_dist, name_match);

  if (has_match) {
    return {has_match, mr};
  }

  return matching(nloc, old_state, update_state, dists, match_bus_stop_max_dist,
                  first_number_match);
}

std::pair<bool, matching_result> match_by_distance(
    nigiri::location const& nloc, state const& old_state,
    state const& update_state, int const r, int const match_bus_stop_max_dist) {
  assert(nloc.type_ != n::location_type::kStation);
  auto mr = matching_result{};
  mr.nloc_idx_ = nloc.l_;
  mr.nloc_pos_ = nloc.pos_;

  auto matched{false};
  auto shortest_dist{std::numeric_limits<double>::max()};

  // use update_state and old_state
  auto pfs = update_state.pfs_idx_->in_radius_with_distance(nloc.pos_, r);
  auto const old_pfs =
      old_state.pfs_idx_->in_radius_with_distance(nloc.pos_, r);
  pfs.insert(pfs.end(), old_pfs.begin(), old_pfs.end());

  for (auto [dist, pf] : pfs) {
    // only match bus stops with a distance of up to a certain distance
    if (pf->info_.is_bus_stop_ && dist > match_bus_stop_max_dist) {
      continue;
    }

    if (dist < shortest_dist) {
      mr.pf_ = pf;
      shortest_dist = dist;
      matched = true;
    }
  }

  return {matched, mr};
}

// -- helper functions --
std::string remove_special_characters(std::string const& str) {
  // allow numbers: 48 - 57; 0 - 9;
  // allow upper case letters: 65 - 90; A - Z
  // allow lower case letters: 97 - 122; a - z
  std::string result{};

  for (auto c : str) {
    if (('0' <= c && c <= '9') || ('a' <= c && c <= 'z') ||
        ('A' <= c && c <= 'Z')) {
      result += c;
    }
  }

  return result;
}

/**
 * Searches the first consecutive sequence of numbers in a string.
 */
std::string get_first_number_sequence(std::string const& str) {
  // allow numbers: 48 - 57; 0 - 9
  std::string result{};
  auto found_number{false};

  for (auto c : str) {

    if ('0' <= c && c <= '9') {
      found_number = true;
      result += c;
      continue;
    }

    if (found_number) {
      break;
    }
  }

  return {result};
}

// -- string matcher --
bool name_match(platform const* pf, nigiri::location const& nloc) {
  auto str_a = std::string{pf->info_.name_};
  auto str_b = std::string{nloc.name_};
  return exact_str_match(str_a, str_b);
}

bool exact_str_match(std::string& str_a, std::string& str_b) {
  std::transform(str_a.begin(), str_a.end(), str_a.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::transform(str_b.begin(), str_b.end(), str_b.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (str_a.length() == 0 || str_b.length() == 0) {
    return false;
  }

  // 1st exact str match
  if (str_a == str_b) {
    return true;
  }

  // remove all special characters from strv_a and strv_b
  str_a = remove_special_characters(str_a);
  str_b = remove_special_characters(str_b);

  // 2nd exact str match
  return str_a == str_b;
}

bool first_number_match(platform const* pf, nigiri::location const& nloc) {
  auto str_a = std::string{pf->info_.name_};
  auto str_b = std::string{nloc.name_};
  return exact_first_number_match(str_a, str_b);
}

bool exact_first_number_match(std::string& str_a, std::string& str_b) {
  std::transform(str_a.begin(), str_a.end(), str_a.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::transform(str_b.begin(), str_b.end(), str_b.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // get first sequence of numbers
  str_a = get_first_number_sequence(str_a);
  str_b = get_first_number_sequence(str_b);

  if (str_a.length() == 0 || str_b.length() == 0) {
    return false;
  }

  // 2nd exact str match
  return str_a == str_b;
}

}  // namespace motis::footpaths
