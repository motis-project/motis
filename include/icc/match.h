#include <map>

#include "nigiri/timetable.h"

#include "osr/platforms.h"

namespace icc {

using matching_t =
    cista::offset::vector_map<nigiri::location_idx_t, osr::platform_idx_t>;

std::optional<geo::latlng> get_platform_center(osr::platforms const&,
                                               osr::ways const&,
                                               osr::platform_idx_t);

osr::platform_idx_t get_match(nigiri::timetable const&,
                              osr::platforms const&,
                              osr::ways const&,
                              nigiri::location_idx_t);

matching_t match(nigiri::timetable const&,
                 osr::platforms const&,
                 osr::ways const&);

}  // namespace icc