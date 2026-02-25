#pragma once

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/fwd.h"

namespace motis::ep {

struct one_to_many_post {
  api::oneToManyPost_response operator()(
      motis::api::OneToManyParams const&) const;

  config const& config_;
  osr::ways const& w_;
  osr::lookup const& l_;
  osr::elevation_storage const* elevations_;
};

}  // namespace motis::ep
