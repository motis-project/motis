#pragma once

#include "cista/containers/ptr.h"

namespace motis {

#if defined(MOTIS_SCHEDULE_MODE_OFFSET)
namespace data = cista::offset;
#elif defined(MOTIS_SCHEDULE_MODE_RAW)
namespace data = cista::raw;
#else
#error "no ptr mode specified"
#endif

using data::ptr;

}  // namespace motis
