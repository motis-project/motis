#pragma once

#include <memory>
#include <string>

#include "motis/nigiri/tag_lookup.h"

namespace motis::nigiri {

struct gtfsrt {
  gtfsrt(tag_lookup const&, std::string const&);

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::nigiri