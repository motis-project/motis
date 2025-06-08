#pragma once

#include <map>
#include <optional>

#include "osr/lookup.h"
#include "osr/types.h"

#include "motis/data.h"
#include "motis/types.h"

namespace motis {

using platform_matches_t =
    vector_map<nigiri::location_idx_t, osr::platform_idx_t>;

struct way_matches_storage {
  way_matches_storage(std::filesystem::path, cista::mmap::protection);

  cista::mmap mm(char const* file);

  cista::mmap::protection mode_;
  std::filesystem::path p_;

  osr::mm_vecvec<nigiri::location_idx_t, osr::raw_way_candidate> matches_;

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

}  // namespace motis