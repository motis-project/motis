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

constexpr auto const kMotisVersion = std::string_view{MOTIS_VERSION};

void show_help() {
  fmt::println(
      "Usage: [command] [options...]\n"
      "\n"
      "Possible commands:\n"
      // Commands ordered lexicographically
      "  shape       Show statistics for a shape\n"
      "  stats       Show statistics about binary data\n"
      "\n"
      "MOTIS {}\n",
      kMotisVersion);
}

data get_data() {
  auto data_path = fs::path{"data"};
  auto const c = config::read(data_path / "config.yml");
  return data{data_path, c};
}

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

bool handle_shape(int ac, char** av) {
  if (ac != 4) {
    fmt::println("Usage: shape <tag> <trip-id> <day> <time>\nMOTIS {}",
                 kMotisVersion);
    return 1;
  }
  auto const tag = std::string_view{av[0]};
  auto const trip_id = std::string_view{av[1]};
  auto const day = std::string_view{av[2]};
  auto const time = std::string_view{av[3]};

  auto const d = get_data();

  auto const success = print_shape_offsets(d, day, time, tag, trip_id);

  return success ? 0 : 1;
}

bool handle_stats([[maybe_unused]] int ac, [[maybe_unused]] char** av) {
  auto const d = get_data();

  if (d.tt_) {
    auto const tt = *d.tt_;
    fmt::println(
        "Number of sources:    {}\n"
        "Number of agencies:   {}\n"
        "Number of routes:     {}\n"
        "Number of trips:      {}\n"
        "Number of locations:  {}\n"
        "\n",
        tt.n_sources(), tt.n_agencies(), tt.n_routes(), tt.n_trips(),
        tt.n_locations());
  } else {
    fmt::println("No timetable data\n\n");
  }

  return true;
}

int main(int ac, char** av) {
  if (ac < 2) {
    show_help();
    return 1;
  }
  auto const cmd = std::string_view{av[1]};
  switch (cista::hash(cmd)) {
    case cista::hash("-h"):
    case cista::hash("--help"): show_help(); return 0;
    case cista::hash("shape"): return handle_shape(ac - 2, av + 2);
    case cista::hash("stats"): return handle_stats(ac - 2, av + 2);
    default: fmt::println("Invalid command '{}'", cmd);
  }
}
