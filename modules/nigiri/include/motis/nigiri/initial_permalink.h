#pragma once

#include <string>

namespace nigiri {
struct timetable;
}

namespace motis::nigiri {

std::string get_initial_permalink(::nigiri::timetable const&);

}  // namespace motis::nigiri