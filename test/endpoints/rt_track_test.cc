#include "gtest/gtest.h"

#include "utl/init_from.h"

#include "nigiri/common/parse_time.h"
#include "nigiri/rt/rt_timetable.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/trip.h"
#include "motis/import.h"
#include "motis/rt/auser.h"
#include "motis/tag_lookup.h"

using namespace std::string_view_literals;
using namespace motis;
using namespace date;

namespace {

// Note: no platform_code in the GTFS - the tracks are expected to come purely
// from the SIRI real-time update (DeparturePlatformName / ArrivalPlatformName).
constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
SNCF,SNCF,https://www.sncf.com,Europe/Paris

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
87381509,Mantes-la-Jolie,48.9899,1.7038,0,,
87386763,Épône - Mézières,48.9599,1.8199,0,,
87386680,Les Mureaux,48.9899,1.9199,0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
J,SNCF,J,Ligne J,,109

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
J,S1,130150,Les Mureaux,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
130150,17:00:00,17:00:00,87381509,1,0,0
130150,17:08:00,17:09:00,87386763,2,0,0
130150,17:19:00,17:19:00,87386680,3,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20260630,1
)";

// Estimated timetable update carrying the platform names for the run.
// The real-time track strings (E / 2 / 22) live only in this XML.
constexpr auto const kSiriUpdate = R"(<?xml version="1.0" encoding="UTF-8"?>
<Siri xmlns="http://www.siri.org.uk/siri" version="2.0">
  <ServiceDelivery>
    <ResponseTimestamp>2026-06-30T14:29:46</ResponseTimestamp>
    <EstimatedTimetableDelivery version="2.0">
      <ResponseTimestamp>2026-06-30T14:29:46</ResponseTimestamp>
      <EstimatedJourneyVersionFrame>
        <RecordedAtTime>2026-06-30T14:29:46</RecordedAtTime>
        <EstimatedVehicleJourney>
          <LineRef>J</LineRef>
          <DirectionRef>OUTBOUND</DirectionRef>
          <FramedVehicleJourneyRef>
            <DataFrameRef>2026-06-30</DataFrameRef>
            <DatedVehicleJourneyRef>unknown</DatedVehicleJourneyRef>
          </FramedVehicleJourneyRef>
          <EstimatedCalls>
            <EstimatedCall>
              <StopPointRef>87381509</StopPointRef>
              <Order>1</Order>
              <AimedDepartureTime>2026-06-30T17:00:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2026-06-30T17:00:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>E</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>87386763</StopPointRef>
              <Order>2</Order>
              <AimedArrivalTime>2026-06-30T17:08:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2026-06-30T17:08:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>2</ArrivalPlatformName>
              <AimedDepartureTime>2026-06-30T17:09:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2026-06-30T17:09:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>2</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>87386680</StopPointRef>
              <Order>3</Order>
              <AimedArrivalTime>2026-06-30T17:19:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2026-06-30T17:19:00+02:00</ExpectedArrivalTime>
              <ArrivalPlatformName>22</ArrivalPlatformName>
            </EstimatedCall>
          </EstimatedCalls>
        </EstimatedVehicleJourney>
      </EstimatedJourneyVersionFrame>
    </EstimatedTimetableDelivery>
  </ServiceDelivery>
</Siri>
)";

}  // namespace

TEST(motis, rt_siri_track_string) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c =
      config{.timetable_ =
                 config::timetable{.first_day_ = "2026-06-30",
                                   .num_days_ = 2,
                                   .datasets_ = {{"test", {.path_ = kGTFS}}}},
             .street_routing_ = false};
  import(c, "test/data");
  auto d = data{"test/data", c};
  d.init_rtt(date::sys_days{2026_y / June / 30});
  auto& rtt = *d.rt_->rtt_;

  auto const trip_ep = utl::init_from<ep::trip>(d).value();
  auto const trip_id = "?tripId=20260630_17%3A00_test_130150"sv;

  // Baseline: without real-time data there is no track (GTFS has no
  // platform_code).
  {
    auto const res = trip_ep(std::string{trip_id});
    ASSERT_EQ(1, res.legs_.size());
    EXPECT_FALSE(res.legs_.front().from_.track_.has_value());
  }

  // Apply SIRI estimated timetable update carrying the platform names.
  auto siri_updater = auser(*d.tt_, d.tags_->get_src("test"),
                            nigiri::rt::vdv_aus::updater::xml_format::kSiri);
  auto const stats = siri_updater.consume_update(std::string{kSiriUpdate}, rtt);
  EXPECT_EQ(1U, stats.matched_runs_);

  // The real-time track string must be exposed on the MOTIS API level and
  // must equal the string from the SIRI XML.
  auto const res = trip_ep(std::string{trip_id});
  ASSERT_EQ(1, res.legs_.size());
  auto const& leg = res.legs_.front();

  ASSERT_TRUE(leg.from_.track_.has_value());
  EXPECT_EQ("E", *leg.from_.track_);  // <DeparturePlatformName>

  ASSERT_TRUE(leg.intermediateStops_.has_value());
  ASSERT_EQ(1, leg.intermediateStops_->size());
  ASSERT_TRUE(leg.intermediateStops_->at(0).track_.has_value());
  EXPECT_EQ("2", *leg.intermediateStops_->at(0).track_);

  ASSERT_TRUE(leg.to_.track_.has_value());
  EXPECT_EQ("22", *leg.to_.track_);  // <ArrivalPlatformName>
}
