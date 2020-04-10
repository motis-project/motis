#pragma once

#include "utl/parser/cstr.h"

#include "motis/loader/hrd/model/hrd_service.h"

namespace motis::loader::hrd {

struct range {
  range() = default;
  range(std::vector<hrd_service::stop> const& stops, utl::cstr from_eva_or_idx,
        utl::cstr to_eva_or_idx, utl::cstr from_hhmm_or_idx,
        utl::cstr to_hhmm_or_idx);

  int from_idx_, to_idx_;
};

}  // namespace motis::loader::hrd
