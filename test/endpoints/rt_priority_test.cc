#include "gtest/gtest.h"

#include <chrono>

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/init_from.h"

#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/stop_times.h"
#include "motis/endpoints/trip.h"
#include "motis/import.h"
#include "motis/rt/auser.h"
#include "motis/tag_lookup.h"

#include "../util.h"

using namespace std::string_view_literals;
using namespace motis;
using namespace date;
using namespace std::chrono_literals;
using namespace test;
namespace n = nigiri;

namespace {

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
A,Stop A,49.90,8.00,1,
B,Stop B,50.00,8.00,1,
C,Stop C,50.10,8.00,1,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
ICE,DB,ICE,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
ICE,SVC,T1,,
ICE,SVC,T2,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,09:00:00,09:00:00,A,0,0,0
T1,09:10:00,09:10:00,B,1,0,0
T1,09:20:00,09:20:00,C,2,0,0
T2,09:00:00,09:00:00,A,0,0,0
T2,09:10:00,09:10:00,B,1,0,0
T2,09:20:00,09:20:00,C,2,0,0

# calendar_dates.txt
service_id,date,exception_type
SVC,20190501,1
)";

data import_gtfs(char const* data_dir) {
  auto ec = std::error_code{};
  std::filesystem::remove_all(data_dir, ec);

  auto const c = config{.timetable_ = config::timetable{
                            .first_day_ = "2019-05-01",
                            .num_days_ = 2,
                            .datasets_ = {{"test", {.path_ = kGTFS}}}}};
  import(c, data_dir);
  auto d = data{data_dir, c};
  d.init_rtt(date::sys_days{2019_y / May / 1});
  return d;
}

api::StopTime b_stoptime(ep::stop_times const& stop_times,
                         std::string_view const trip_id_suffix) {
  auto const res = stop_times(
      "/api/v5/stoptimes?stopId=test_B"
      "&time=2019-05-01T08:30:00.000Z"
      "&arriveBy=true"
      "&n=5"
      "&language=de"
      "&fetchStops=true");
  for (auto const& st : res.stopTimes_) {
    if (st.tripId_.ends_with(trip_id_suffix)) {
      return st;
    }
  }
  ADD_FAILURE() << "no stop time at B matching trip id suffix \""
                << trip_id_suffix << "\"";
  return {};
}

std::string format_time(auto&& t) { return date::format("%F %H:%M", *t); }

}  // namespace

TEST(motis, gtfsrt_priority_skip_existing_update) {
  auto d = import_gtfs("test/data_rt_priority_gtfsrt");
  auto const stop_times = utl::init_from<ep::stop_times>(d).value();

  // Regular priority: creates a realtime trip for T1 with a 5 min delay.
  auto const stats1 = n::rt::gtfsrt_update_msg(
      *d.tt_, *d.rt_->rtt_, n::source_idx_t{0}, "test",
      to_feed_msg(
          {trip_update{.trip_ = {.trip_id_ = "T1", .date_ = {"20190501"}},
                       .stop_updates_ = {{.stop_id_ = "B",
                                          .ev_type_ = n::event_type::kArr,
                                          .delay_minutes_ = 5}}}},
          date::sys_days{2019_y / May / 1} + 9h),
      /* use_vehicle_position= */ false, /* skip_existing_update= */ false);
  EXPECT_EQ(1U, stats1.total_entities_success_);
  {
    auto const t1 = b_stoptime(stop_times, "T1");
    EXPECT_TRUE(t1.realTime_);
    EXPECT_EQ("2019-05-01 07:15", format_time(t1.place_.arrival_.value()));
  }

  // Priority 0: rt trip already exists, so the 20 min delay is ignored.
  auto const stats2 = n::rt::gtfsrt_update_msg(
      *d.tt_, *d.rt_->rtt_, n::source_idx_t{0}, "test",
      to_feed_msg(
          {trip_update{.trip_ = {.trip_id_ = "T1", .date_ = {"20190501"}},
                       .stop_updates_ = {{.stop_id_ = "B",
                                          .ev_type_ = n::event_type::kArr,
                                          .delay_minutes_ = 20}}}},
          date::sys_days{2019_y / May / 1} + 9h),
      /* use_vehicle_position= */ false, /* skip_existing_update= */ true);
  EXPECT_EQ(1U, stats2.total_entities_success_);
  {
    auto const t1 = b_stoptime(stop_times, "T1");
    EXPECT_TRUE(t1.realTime_);
    EXPECT_EQ("2019-05-01 07:15", format_time(t1.place_.arrival_.value()));
  }

  // Priority 0, but T2 has no rt trip yet: still creates one.
  auto const stats3 = n::rt::gtfsrt_update_msg(
      *d.tt_, *d.rt_->rtt_, n::source_idx_t{0}, "test",
      to_feed_msg(
          {trip_update{.trip_ = {.trip_id_ = "T2", .date_ = {"20190501"}},
                       .stop_updates_ = {{.stop_id_ = "B",
                                          .ev_type_ = n::event_type::kArr,
                                          .delay_minutes_ = 7}}}},
          date::sys_days{2019_y / May / 1} + 9h),
      /* use_vehicle_position= */ false, /* skip_existing_update= */ true);
  EXPECT_EQ(1U, stats3.total_entities_success_);
  {
    auto const t2 = b_stoptime(stop_times, "T2");
    EXPECT_TRUE(t2.realTime_);
    EXPECT_EQ("2019-05-01 07:17", format_time(t2.place_.arrival_.value()));
  }
}

constexpr auto const kSiriUpdate1 = R"(<?xml version="1.0" encoding="UTF-8"?>
<Siri xmlns="http://www.siri.org.uk/siri" version="2.0">
  <ServiceDelivery>
    <ResponseTimestamp>2019-05-01T08:29:46</ResponseTimestamp>
    <EstimatedTimetableDelivery version="2.0">
      <ResponseTimestamp>2019-05-01T08:29:46</ResponseTimestamp>
      <EstimatedJourneyVersionFrame>
        <RecordedAtTime>2019-05-01T08:29:46</RecordedAtTime>
        <EstimatedVehicleJourney>
          <LineRef>ICE</LineRef>
          <DirectionRef>OUTBOUND</DirectionRef>
          <FramedVehicleJourneyRef>
            <DataFrameRef>2019-05-01</DataFrameRef>
            <DatedVehicleJourneyRef>unknown</DatedVehicleJourneyRef>
          </FramedVehicleJourneyRef>
          <EstimatedCalls>
            <EstimatedCall>
              <StopPointRef>A</StopPointRef>
              <Order>1</Order>
              <AimedDepartureTime>2019-05-01T09:00:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2019-05-01T09:00:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>1</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>B</StopPointRef>
              <Order>2</Order>
              <AimedArrivalTime>2019-05-01T09:10:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2019-05-01T09:10:00+02:00</ExpectedArrivalTime>
            </EstimatedCall>
          </EstimatedCalls>
        </EstimatedVehicleJourney>
      </EstimatedJourneyVersionFrame>
    </EstimatedTimetableDelivery>
  </ServiceDelivery>
</Siri>
)";

constexpr auto const kSiriUpdate2 = R"(<?xml version="1.0" encoding="UTF-8"?>
<Siri xmlns="http://www.siri.org.uk/siri" version="2.0">
  <ServiceDelivery>
    <ResponseTimestamp>2019-05-01T08:35:46</ResponseTimestamp>
    <EstimatedTimetableDelivery version="2.0">
      <ResponseTimestamp>2019-05-01T08:35:46</ResponseTimestamp>
      <EstimatedJourneyVersionFrame>
        <RecordedAtTime>2019-05-01T08:35:46</RecordedAtTime>
        <EstimatedVehicleJourney>
          <LineRef>ICE</LineRef>
          <DirectionRef>OUTBOUND</DirectionRef>
          <FramedVehicleJourneyRef>
            <DataFrameRef>2019-05-01</DataFrameRef>
            <DatedVehicleJourneyRef>unknown</DatedVehicleJourneyRef>
          </FramedVehicleJourneyRef>
          <EstimatedCalls>
            <EstimatedCall>
              <StopPointRef>A</StopPointRef>
              <Order>1</Order>
              <AimedDepartureTime>2019-05-01T09:00:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2019-05-01T09:00:00+02:00</ExpectedDepartureTime>
              <DeparturePlatformName>99</DeparturePlatformName>
            </EstimatedCall>
            <EstimatedCall>
              <StopPointRef>B</StopPointRef>
              <Order>2</Order>
              <AimedArrivalTime>2019-05-01T09:10:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2019-05-01T09:10:00+02:00</ExpectedArrivalTime>
            </EstimatedCall>
          </EstimatedCalls>
        </EstimatedVehicleJourney>
      </EstimatedJourneyVersionFrame>
    </EstimatedTimetableDelivery>
  </ServiceDelivery>
</Siri>
)";

// Same as gtfsrt_priority_skip_existing_update, but for the VDV/SIRI path
// via motis::auser.
TEST(motis, vdv_priority_skip_existing_update) {
  auto d = import_gtfs("test/data_rt_priority_vdv");
  auto const trip_ep = utl::init_from<ep::trip>(d).value();
  auto const trip_id = "?tripId=20190501_09%3A00_test_T1"sv;

  // Regular priority: creates a realtime trip, sets track to "1".
  auto normal_updater = auser(*d.tt_, d.tags_->get_src("test"),
                              nigiri::rt::vdv_aus::updater::xml_format::kSiri,
                              /* skip_existing_update= */ false);
  auto const stats1 =
      normal_updater.consume_update(std::string{kSiriUpdate1}, *d.rt_->rtt_);
  EXPECT_EQ(1U, stats1.matched_runs_);
  {
    auto const res = trip_ep(std::string{trip_id});
    ASSERT_EQ(1, res.legs_.size());
    ASSERT_TRUE(res.legs_.front().from_.track_.has_value());
    EXPECT_EQ("1", *res.legs_.front().from_.track_);
  }

  // Priority 0: rt trip already exists, so track "99" is ignored.
  auto priority0_updater =
      auser(*d.tt_, d.tags_->get_src("test"),
            nigiri::rt::vdv_aus::updater::xml_format::kSiri,
            /* skip_existing_update= */ true);
  auto const stats2 =
      priority0_updater.consume_update(std::string{kSiriUpdate2}, *d.rt_->rtt_);
  EXPECT_EQ(1U, stats2.matched_runs_);
  {
    auto const res = trip_ep(std::string{trip_id});
    ASSERT_EQ(1, res.legs_.size());
    ASSERT_TRUE(res.legs_.front().from_.track_.has_value());
    EXPECT_EQ("1", *res.legs_.front().from_.track_);
  }
}
