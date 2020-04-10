#pragma once

#include "cista/containers/ptr.h"

namespace motis {

#if defined(MOTIS_SCHEDULE_MODE_OFFSET) && !defined(CLANG_TIDY)
namespace data = cista::offset;
#elif defined(MOTIS_SCHEDULE_MODE_RAW) || defined(CLANG_TIDY)
namespace data = cista::raw;
#else
#error "no ptr mode specified"
#endif

using data::ptr;

}  // namespace motis
