#pragma once

#include <map>
#include <optional>

#include "osr/lookup.h"
#include "osr/types.h"

#include "nigiri/types.h"

#include "motis/data.h"
#include "motis/types.h"

namespace motis {

using platform_matches_t =
    vector_map<nigiri::location_idx_t, osr::platform_idx_t>;

struct way_matches_storage {
  way_matches_storage(std::filesystem::path,
                      cista::mmap::protection,
                      double max_matching_distance);

  cista::mmap mm(char const* file);

  cista::mmap::protection mode_;
  std::filesystem::path p_;

  osr::mm_vecvec<nigiri::location_idx_t, osr::raw_way_candidate> matches_;
  double max_matching_distance_;

  void preprocess_osr_matches(nigiri::timetable const&,
                              osr::platforms const&,
                              osr::ways const&,
                              osr::lookup const&,
                              platform_matches_t const&);
};

std::optional<geo::latlng> get_platform_center(osr::platforms const&,
                                               osr::ways const&,
                                               osr::platform_idx_t);

osr::platform_idx_t get_match(nigiri::timetable const&,
                              osr::platforms const&,
                              osr::ways const&,
                              nigiri::location_idx_t);

platform_matches_t get_matches(nigiri::timetable const&,
                               osr::platforms const&,
                               osr::ways const&);

std::optional<std::string_view> get_track(std::string_view);

std::vector<osr::match_t> get_reverse_platform_way_matches(
    osr::lookup const&,
    way_matches_storage const*,
    osr::search_profile,
    std::span<nigiri::location_idx_t const>,
    std::span<osr::location const>,
    osr::direction,
    double max_matching_distance);

}  // namespace motis