#pragma once

#include <string>

#include "nigiri/rt/vdv_aus.h"

namespace motis {

struct auser {
  auser(nigiri::timetable const&,
        nigiri::source_idx_t,
        nigiri::rt::vdv_aus::updater::xml_format);
  std::string fetch_url(std::string_view base_url);
  nigiri::rt::vdv_aus::statistics consume_update(std::string const&,
                                                 nigiri::rt_timetable&,
                                                 bool inplace = false);

  std::chrono::nanoseconds::rep update_state_{0};
  nigiri::rt::vdv_aus::updater upd_;
};

}  // namespace motis