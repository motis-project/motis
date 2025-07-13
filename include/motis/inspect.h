#include <string_view>

#include "motis/data.h"

namespace motis {

bool analyze_shapes(data const&, std::vector<std::string> const& trip_ids);

bool analyze_shape(nigiri::shapes_storage const&, nigiri::trip_idx_t const&);

}  // namespace motis
