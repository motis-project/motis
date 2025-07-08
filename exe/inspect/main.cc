#include <filesystem>
#include <ranges>
#include <string_view>

#include "fmt/base.h"

#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/tag_lookup.h"

#if !defined(MOTIS_VERSION)
#define MOTIS_VERSION "unknown"
#endif

namespace fs = std::filesystem;

using namespace motis;

bool print_shape_offsets(data const& d,
                         std::string_view day,
                         std::string_view time,
                         std::string_view tag,
                         std::string_view trip_id) {
  if (!d.tt_) {
    fmt::println("Missing timetable");
    return false;
  }
  if (!d.rt_ || !d.rt_->rtt_) {
    fmt::println("Missing realtime data");
    return false;
  }
  if (!d.tags_) {
    fmt::println("Missing tags");
    return false;
  }
  if (!d.shapes_) {
    fmt::println("Missing shapes");
    return false;
  }
  auto const& tt = *d.tt_;
  auto const& rtt = *d.rt_->rtt_;
  auto const& tags = *d.tags_;
  auto const& shapes = *d.shapes_;

  auto const id = std::string{day.data()} + '_' + time.data() + '_' +
                  tag.data() + '_' + trip_id.data();
  auto const [run, trip_idx] = tags.get_trip(tt, &rtt, id);
  if (!run.valid()) {
    return false;
  }
  fmt::println("Found trip idx: {}", trip_idx);

  auto const offset_idx = shapes.trip_offset_indices_[trip_idx].second;
  if (offset_idx == nigiri::shape_offset_idx_t::invalid()) {
    fmt::println("No shape offsets for '{}'", trip_id);
    return false;
  }

  auto const offsets = shapes.offsets_[offset_idx];
  if (offsets.empty()) {
    fmt::println("Empty shape for '{}'", trip_id);
    return false;
  }

  fmt::println("Offsets for '{}':", trip_id);
  for (auto const& o : offsets) {
    fmt::print("{}, ", o);
  }
  fmt::println("");
  return true;
}

int main(int ac, char** av) {
  auto const motis_version = std::string_view{MOTIS_VERSION};
  if (ac != 5) {
    fmt::println("Usage: <tag> <trip-id> <day> <time>\nMOTIS {}",
                 motis_version);
    return 1;
  }
  auto const tag = std::string_view{av[1]};
  auto const trip_id = std::string_view{av[2]};
  auto const day = std::string_view{av[3]};
  auto const time = std::string_view{av[4]};

  auto data_path = fs::path{"data"};
  auto const c = config::read(data_path / "config.yml");
  auto const d = data{data_path, c};

  auto const success = print_shape_offsets(d, day, time, tag, trip_id);

  return success ? 0 : 1;
}
