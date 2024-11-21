#include "motis/endpoints/suspects.h"

#include "fmt/base.h"

#include "motis/suspects.h"

namespace n = nigiri;

namespace motis::ep {

api::plan_response suspects::operator()(boost::urls::url_view const&) const {
  return {.debugOutput_ = {{"suspects", suspects_.routes_.size()}}};
}

}  // namespace motis::ep