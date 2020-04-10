#include "gtest/gtest.h"

#include "motis/loader/gtfs/agency.h"
#include "motis/loader/gtfs/files.h"

#include "resources.h"

using namespace utl;
using namespace motis::loader;

namespace motis::loader::gtfs {

TEST(loader_gtfs_stop, read_stations_example_data) {
  auto agencies =
      read_agencies(loaded_file{SCHEDULES / "example" / AGENCY_FILE});

  ASSERT_EQ(1, agencies.size());
  ASSERT_NE(agencies.find("FunBus"), end(agencies));
  ASSERT_EQ("FunBus", agencies["FunBus"]->id_);
  ASSERT_EQ("The Fun Bus", agencies["FunBus"]->name_);
  ASSERT_EQ("America/Los_Angeles", agencies["FunBus"]->timezone_);
}

}  // namespace motis::loader::gtfs
