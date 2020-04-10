#pragma once

#include <ctime>

#include "utl/parser/cstr.h"

namespace motis::ris::risml {

struct context;

std::time_t parse_time(utl::cstr const&);

std::time_t parse_schedule_time(context&, utl::cstr const&);

}  // namespace motis::ris::risml
