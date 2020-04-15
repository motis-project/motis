#pragma once

#include "utl/parser/buffer.h"
#include "utl/parser/cstr.h"
#include "utl/verify.h"

#include "boost/filesystem/path.hpp"

#include "motis/loader/hrd/model/hrd_service.h"
#include "motis/loader/hrd/model/specification.h"
#include "motis/loader/loaded_file.h"
#include "motis/loader/util.h"

namespace motis::loader::hrd {

struct test_spec {
  test_spec(boost::filesystem::path const& root, char const* filename)
      : lf_(root / filename) {}

  std::vector<specification> get_specs() const;
  std::vector<hrd_service> get_hrd_services(config const&) const;
  loaded_file lf_;
};

}  // namespace motis::loader::hrd
