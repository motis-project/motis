#include <map>
#include <optional>

#include "osr/types.h"

#include "icc/data.h"
#include "icc/types.h"

namespace icc {

using platform_matches_t =
    vector_map<nigiri::location_idx_t, osr::platform_idx_t>;

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

}  // namespace icc