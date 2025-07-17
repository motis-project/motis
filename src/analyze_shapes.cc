#include "motis/analyze_shapes.h"

#include "utl/verify.h"

#include "fmt/base.h"

#include "nigiri/shapes_storage.h"
#include "nigiri/types.h"

#include "motis/tag_lookup.h"

namespace motis {

bool analyze_shape(nigiri::shapes_storage const& shapes,
                   nigiri::trip_idx_t const& trip_idx) {
  auto const offset_idx = shapes.trip_offset_indices_[trip_idx].second;
  if (offset_idx == nigiri::shape_offset_idx_t::invalid()) {
    fmt::println("No shape offsets for '{}'", trip_idx);
    return false;
  }

  auto const offsets = shapes.offsets_[offset_idx];
  if (offsets.empty()) {
    fmt::println("Empty shape for '{}'", trip_idx);
    return false;
  }

  fmt::println("Offsets for '{}':", trip_idx);
  for (auto const& o : offsets) {
    fmt::print("{}, ", o);
  }
  fmt::println("");
  return true;
}

bool analyze_shapes(data const& d, std::vector<std::string> const& trip_ids) {
  utl::verify(d.tt_, "Missing timetable");
  utl::verify(d.rt_ && d.rt_->rtt_, "Missing realtime data");
  utl::verify(d.tags_, "Missing tags");
  utl::verify(!!d.shapes_, "Missing shapes");

  auto success = true;
  for (auto const& trip_id : trip_ids) {
    fmt::println("Searching trip-id '{}' ...", trip_id);
    auto const [run, trip_idx] =
        d.tags_->get_trip(*d.tt_, &*d.rt_->rtt_, trip_id);
    if (!run.valid()) {
      success = false;
      fmt::println("Did not find trip idx for trip-id '{}'", trip_id);
      continue;
    }
    fmt::println("Found trip idx for trip-id '{}': {}", trip_id, trip_idx);
    success &= analyze_shape(*d.shapes_, trip_idx);
    fmt::println("");
  }
  return success;
}

}  // namespace motis
