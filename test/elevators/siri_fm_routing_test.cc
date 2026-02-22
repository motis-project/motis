#include "gtest/gtest.h"

#include <chrono>

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/init_from.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"
#include "motis/import.h"

namespace json = boost::json;
using namespace std::string_view_literals;
using namespace motis;
using namespace date;
using namespace std::chrono_literals;
namespace n = nigiri;

constexpr auto const kElevatorIdOsm = R"(dhid,diid,osm_kind,osm_id
de:06412:10,diid:02b2be0f-c1da-1eef-a490-d5f7573837ae,node,3945358489
)";

constexpr auto const kSiriFm = R"__(
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Siri xmlns="http://www.siri.org.uk/siri" xmlns:ns2="http://www.ifopt.org.uk/acsb"
    xmlns:ns3="http://www.ifopt.org.uk/ifopt" xmlns:ns4="http://datex2.eu/schema/2_0RC1/2_0"
    xmlns:ns5="http://www.opengis.net/gml/3.2" version="siri:2.2">
    <ServiceDelivery>
        <ResponseTimestamp>2026-02-21T22:06:02Z</ResponseTimestamp>
        <ProducerRef>dbinfrago</ProducerRef>
        <FacilityMonitoringDelivery version="epiprt:2.1">
            <ResponseTimestamp>2026-02-21T22:06:02Z</ResponseTimestamp>
            <FacilityCondition>
                <FacilityRef>diid:02b2be0f-c1da-1eef-a490-d5f7573837ae</FacilityRef>
                <FacilityStatus>
                    <Status>notAvailable</Status>
                    <Description xml:lang="en">not available</Description>
                </FacilityStatus>
            </FacilityCondition>
      </FacilityMonitoringDelivery>
  </ServiceDelivery>
</Siri>
)__"sv;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
DA,DA Hbf,49.87260,8.63085,1,,
DA_3,DA Hbf,49.87355,8.63003,0,DA,3
DA_10,DA Hbf,49.87336,8.62926,0,DA,10
FFM,FFM Hbf,50.10701,8.66341,1,,
FFM_101,FFM Hbf,50.10739,8.66333,0,FFM,101
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10
FFM_12,FFM Hbf,50.10658,8.66178,0,FFM,12
de:6412:10:6:1,FFM Hbf U-Bahn,50.107577,8.6638173,0,FFM,U4
LANGEN,Langen,49.99359,8.65677,1,,1
FFM_HAUPT,FFM Hauptwache,50.11403,8.67835,1,,
FFM_HAUPT_U,Hauptwache U1/U2/U3/U8,50.11385,8.67912,0,FFM_HAUPT,
FFM_HAUPT_S,FFM Hauptwache S,50.11404,8.67824,0,FFM_HAUPT,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
S3,DB,S3,,,109
U4,DB,U4,,,402
ICE,DB,ICE,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
S3,S1,S3,,
U4,S1,U4,,
ICE,S1,ICE,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
S3,01:15:00,01:15:00,FFM_101,1,0,0
S3,01:20:00,01:20:00,FFM_HAUPT_S,2,0,0
U4,01:05:00,01:05:00,de:6412:10:6:1,0,0,0
U4,01:10:00,01:10:00,FFM_HAUPT_U,1,0,0
ICE,00:35:00,00:35:00,DA_10,0,0,0
ICE,00:45:00,00:45:00,FFM_10,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# frequencies.txt
trip_id,start_time,end_time,headway_secs
S3,01:15:00,25:15:00,3600
ICE,00:35:00,24:35:00,3600
U4,01:05:00,25:01:00,3600
)"sv;

void print_short(std::ostream& out, api::Itinerary const& j);
std::string to_str(std::vector<api::Itinerary> const& x);

TEST(motis, siri_fm_routing) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c =
      config{.server_ = {{.web_folder_ = "ui/build", .n_threads_ = 1U}},
             .osm_ = {"test/resources/test_case.osm.pbf"},
             .tiles_ = {{.profile_ = "deps/tiles/profile/full.lua",
                         .db_size_ = 1024U * 1024U * 25U}},
             .timetable_ =
                 config::timetable{
                     .first_day_ = "2019-05-01",
                     .num_days_ = 2,
                     .preprocess_max_matching_distance_ = 0.0,
                     .datasets_ = {{"test", {.path_ = std::string{kGTFS}}}}},
             .elevators_ =
                 config::elevators{.init_ = std::string{kSiriFm},
                                   .osm_mapping_ = std::string{kElevatorIdOsm}},
             .street_routing_ = true,
             .osr_footpath_ = true,
             .geocoding_ = true,
             .reverse_geocoding_ = true};
  import(c, "test/data", true);

  auto d = data{"test/data", c};

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
