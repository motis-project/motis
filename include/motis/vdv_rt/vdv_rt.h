#pragma once

#include "nigiri/rt/vdv/vdv_update.h"

#include "motis/fwd.h"
#include "motis/vdv_rt/connection.h"

namespace motis::vdv_rt {

struct vdv_rt {
  explicit vdv_rt(config::timetable::dataset::vdv_rt const& vdv_rt_cfg,
                  nigiri::timetable const& tt,
                  nigiri::source_idx_t const src)
      : cfg_{vdv_rt_cfg}, con_{vdv_rt_cfg}, upd_{tt, src} {}

  config::timetable::dataset::vdv_rt const& cfg_;
  connection con_;
  nigiri::rt::vdv::updater upd_;
};

}  // namespace motis::vdv_rt