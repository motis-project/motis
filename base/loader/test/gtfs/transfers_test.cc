#include "gtest/gtest.h"

#include "motis/loader/gtfs/files.h"
#include "motis/loader/gtfs/transfers.h"

#include "./resources.h"

using namespace utl;
using namespace motis::loader;
using namespace motis::loader::gtfs;

stop_pair t(stop_map const& stops, std::string const& s1,
            std::string const& s2) {
  return std::make_pair(stops.at(s1).get(),  //
                        stops.at(s2).get());
}

TEST(loader_gtfs_transfer, read_transfers_example_data) {
  auto stops =
      read_stops(loaded_file{SCHEDULES / "example" / STOPS_FILE}, false);
  auto transfers = read_transfers(
      loaded_file{SCHEDULES / "example" / TRANSFERS_FILE}, stops, false);

  EXPECT_EQ(2, transfers.size());

  EXPECT_EQ(5, transfers[t(stops, "S6", "S7")].minutes_);
  EXPECT_EQ(transfer::MIN_TRANSFER_TIME, transfers[t(stops, "S6", "S7")].type_);
  EXPECT_EQ(transfer::NOT_POSSIBLE, transfers[t(stops, "S7", "S6")].type_);
}
