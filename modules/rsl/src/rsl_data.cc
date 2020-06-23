#include "motis/rsl/rsl_data.h"

namespace motis::rsl {

const passenger_group& rsl_data::get_passenger_group(std::uint64_t id) const {
  return *graph_.passenger_groups_.at(id).get();
}

}  // namespace motis::rsl
