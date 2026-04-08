#include "gtest/gtest.h"

#include <chrono>

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/init_from.h"

#include "motis/endpoints/routing.h"

#include "../test_case.h"

namespace json = boost::json;
using namespace motis;
using namespace date;
using namespace std::chrono_literals;
namespace n = nigiri;

void print_short(std::ostream& out, api::Itinerary const& j);
std::string to_str(std::vector<api::Itinerary> const& x);

TEST(motis, siri_fm_routing) {
  auto [d, _] = get_test_case<test_case::FFM_siri_fm_routing>();

  auto const routing = utl::init_from<ep::routing>(d).value();

  // Route with wheelchair.
  {
    auto const res = routing(
        "?fromPlace=49.87263,8.63127"
        "&toPlace=50.11347,8.67664"
        "&time=2019-05-01T01:25Z"
        "&pedestrianProfile=WHEELCHAIR"
        "&useRoutedTransfers=true"
        "&timetableView=false");
    EXPECT_EQ(0U, res.itineraries_.size());
  }

  // Route w/o wheelchair.
  {
    auto const res = routing(
        "?fromPlace=49.87263,8.63127"
        "&toPlace=50.11347,8.67664"
        "&time=2019-05-01T01:25Z"
        "&useRoutedTransfers=true"
        "&timetableView=false");
    EXPECT_EQ(1U, res.itineraries_.size());
  }
}
