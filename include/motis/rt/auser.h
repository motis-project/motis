#pragma once

#include <string>

#include "nigiri/rt/vdv/vdv_update.h"

namespace motis {

struct auser {
  auser(nigiri::timetable const&, nigiri::source_idx_t);
  std::string fetch_url(std::string_view base_url);
  nigiri::rt::vdv::statistics consume_update(std::string,
                                             nigiri::rt_timetable&);

  std::string last_update_{"0"};
  nigiri::rt::vdv::updater upd_;
};

}  // namespace motis