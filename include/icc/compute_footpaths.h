#pragma once

#include "nigiri/timetable.h"

#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/ways.h"

namespace icc {

void compute_footpaths(nigiri::timetable&,
                       osr::ways const&,
                       osr::lookup const&,
                       osr::platforms const&,
                       bool update_coordinates);

}  // namespace icc