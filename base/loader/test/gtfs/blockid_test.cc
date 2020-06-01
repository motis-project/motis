#include "gtest/gtest.h"

#include <iostream>

#include "motis/core/common/date_time_util.h"
#include "motis/loader/loader.h"

#include "resources.h"

using namespace utl;
using namespace motis::loader;

namespace motis::loader::gtfs {

TEST(loader_gtfs_block_id, trips) {
  auto const sched = load_schedule(
      {(gtfs::SCHEDULES / "block_id").generic_string(), "20060701", 31});
  for (auto const& t : sched->gtfs_trip_ids_) {
    std::cout << t.first.trip_id_ << ", "
              << format_unix_time(t.first.start_date_, "%Y-%m-%d") << "\n";
  }
  std::cout << sched->trips_.size() << "\n";
  std::cout << sched->expanded_trips_.data_size() << "\n";
}

}  // namespace motis::loader::gtfs
