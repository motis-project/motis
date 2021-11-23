#pragma once

#include <ctime>

#include "utl/parser/cstr.h"

#include "motis/core/common/unixtime.h"

namespace motis::ris::risml {

struct context;

unixtime parse_time(utl::cstr const&);

unixtime parse_schedule_time(context&, utl::cstr const&);

}  // namespace motis::ris::risml
