#pragma once

#include "motis/module/context/get_module.h"

#include "utl/verify.h"

#include "motis/csa/csa.h"
#include "motis/csa/csa_timetable.h"

namespace motis::csa {

inline csa_timetable const& csa_get_timetable() {
  auto const tt = motis::module::get_module<csa>("csa").get_timetable();
  utl::verify(tt != nullptr, "csa timetable not initialized!");
  return *tt;
}

}  // namespace motis::csa
