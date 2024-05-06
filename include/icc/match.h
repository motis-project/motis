#include <map>

#include "nigiri/timetable.h"

#include "osr/platforms.h"

namespace icc {

struct matching {
  std::map<nigiri::location_idx_t, osr::platform_idx_t> lp_;
  std::map<osr::platform_idx_t, nigiri::location_idx_t> pl_;
};

matching match(nigiri::timetable const&,
               osr::platforms const&,
               osr::ways const&);

}  // namespace icc