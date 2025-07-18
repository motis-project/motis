#include "motis/analyze_shapes.h"

#include "utl/verify.h"

#include "fmt/base.h"
#include "fmt/ranges.h"

#include "nigiri/shapes_storage.h"
#include "nigiri/types.h"

#include "motis/tag_lookup.h"

namespace motis {

bool analyze_shape(nigiri::shapes_storage const& shapes,
                   std::string const& trip_id,
                   nigiri::trip_idx_t const& trip_idx) {
  auto const offset_idx = shapes.trip_offset_indices_[trip_idx].second;
  if (offset_idx == nigiri::shape_offset_idx_t::invalid()) {
    fmt::println("No shape offsets for trip-id '{}'\n", trip_id);
    return false;
  }

  auto const offsets = shapes.offsets_[offset_idx];
  if (offsets.empty()) {
    fmt::println("Empty shape for trip-id '{}'\n", trip_id);
    return false;
  }

  fmt::println("Offsets for trip-id '{}':\n{}\n", trip_id, offsets);

  return true;
}

bool analyze_shapes(data const& d, std::vector<std::string> const& trip_ids) {
  utl::verify(d.tt_ != nullptr, "Missing timetable");
  utl::verify(d.tags_ != nullptr, "Missing tags");
  utl::verify(d.shapes_ != nullptr, "Missing shapes");

  auto success = true;
  for (auto const& trip_id : trip_ids) {
    auto const [run, trip_idx] = d.tags_->get_trip(*d.tt_, nullptr, trip_id);
    if (!run.valid()) {
      success = false;
      fmt::println("Failed to find trip-id '{}'\n", trip_id);
      continue;
    }
    success &= analyze_shape(*d.shapes_, trip_id, trip_idx);
  }
  return success;
}

}  // namespace motis
