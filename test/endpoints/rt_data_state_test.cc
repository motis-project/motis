#include "gtest/gtest.h"

#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/init_from.h"

#include "nigiri/rt/rt_timetable.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/gtfsrt.h"
#include "motis/endpoints/map/trips.h"
#include "motis/endpoints/metrics.h"
#include "motis/endpoints/stop_times.h"
#include "motis/endpoints/trip.h"
#include "motis/import.h"
#include "motis/metrics_registry.h"
#include "motis/railviz.h"
#include "motis/rt/auser.h"
#include "motis/tag_lookup.h"

using namespace std::string_view_literals;
using namespace motis;
using namespace date;

namespace {

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

// SIRI update for the first and the last stop. The intermediate stop is only
// contained in the update if `intermediate_call` is not empty.
std::string siri_update(std::string_view expected_departure_first,
                        std::string_view expected_arrival_last,
                        std::string_view intermediate_call = "",
                        bool const is_complete_stop_sequence = false) {
  return std::string{R"(<?xml version="1.0" encoding="UTF-8"?>
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
              <ExpectedDepartureTime>)"} +
         std::string{expected_departure_first} +
         R"(</ExpectedDepartureTime>
            </EstimatedCall>)" +
         std::string{intermediate_call} + R"(
            <EstimatedCall>
              <StopPointRef>87386680</StopPointRef>
              <Order>3</Order>
              <AimedArrivalTime>2026-06-30T17:19:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>)" +
         std::string{expected_arrival_last} +
         R"(</ExpectedArrivalTime>
            </EstimatedCall>
          </EstimatedCalls>
          <IsCompleteStopSequence>)" +
         std::string{is_complete_stop_sequence ? "true" : "false"} +
         R"(</IsCompleteStopSequence>
        </EstimatedVehicleJourney>
      </EstimatedJourneyVersionFrame>
    </EstimatedTimetableDelivery>
  </ServiceDelivery>
</Siri>
)";
}

// Stop time updates of the exported GTFS-RT feed as "seq: arr=... dep=...".
std::vector<std::string> gtfsrt_export(data& d) {
  auto const gtfsrt_ep = utl::init_from<ep::gtfsrt>(d).value();
  auto const reply =
      gtfsrt_ep(net::route_request{net::web_server::http_req_t{
                    boost::beast::http::verb::get, "/gtfs-rt", 11}},
                false);
  auto const* res = std::get_if<net::web_server::string_res_t>(&reply);
  EXPECT_NE(nullptr, res);
  if (res == nullptr) {
    return {};
  }

  auto fm = transit_realtime::FeedMessage{};
  EXPECT_TRUE(fm.ParseFromString(res->body()));
  EXPECT_EQ(1, fm.entity_size());
  if (fm.entity_size() != 1) {
    return {};
  }

  auto stus = std::vector<std::string>{};
  for (auto const& stu : fm.entity(0).trip_update().stop_time_update()) {
    auto str = std::to_string(stu.stop_sequence()) + ":";
    if (stu.schedule_relationship() ==
        transit_realtime::
            TripUpdate_StopTimeUpdate_ScheduleRelationship_NO_DATA) {
      str += " NO_DATA";
    }
    if (stu.schedule_relationship() ==
        transit_realtime::
            TripUpdate_StopTimeUpdate_ScheduleRelationship_SKIPPED) {
      str += " SKIPPED";
    }
    if (stu.has_arrival()) {
      str += " arr=" + std::to_string(stu.arrival().delay());
    }
    if (stu.has_departure()) {
      str += " dep=" + std::to_string(stu.departure().delay());
    }
    stus.push_back(str);
  }
  return stus;
}

// Values of the total_rt_events_count metric by rt_state label.
std::map<std::string, double> rt_state_metrics(data& d) {
  auto registry = metrics_registry{};
  auto const metrics_ep = ep::metrics{.tt_ = d.tt_.get(),
                                      .tags_ = d.tags_.get(),
                                      .rt_ = d.rt_,
                                      .metrics_ = &registry};
  auto const reply =
      metrics_ep(net::route_request{net::web_server::http_req_t{
                     boost::beast::http::verb::get, "/metrics", 11}},
                 false);
  auto const* res = std::get_if<net::web_server::string_res_t>(&reply);
  EXPECT_NE(nullptr, res);
  if (res == nullptr) {
    return {};
  }

  auto values = std::map<std::string, double>{};
  auto body = std::istringstream{res->body()};
  auto line = std::string{};
  while (std::getline(body, line)) {
    if (!line.starts_with("total_rt_events_count{")) {
      continue;
    }
    auto const state_start = line.find("rt_state=\"") + 10U;
    auto const state_end = line.find('"', state_start);
    values[line.substr(state_start, state_end - state_start)] +=
        std::stod(line.substr(line.rfind(' ') + 1U));
  }
  return values;
}

}  // namespace

TEST(motis, rt_data_state) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data_rt_data_state", ec);

  auto const c =
      config{.timetable_ =
                 config::timetable{.first_day_ = "2026-06-30",
                                   .num_days_ = 2,
                                   .datasets_ = {{"test", {.path_ = kGTFS}}}},
             .street_routing_ = false};
  import(c, "test/data_rt_data_state");
  auto d = data{"test/data_rt_data_state", c};

  auto const format_time = [](auto&& t) {
    return date::format("%H:%M", *t.value());  // UTC
  };
  auto const trip_id = "?tripId=20260630_17%3A00_test_130150"sv;

  {
    // First stop +5min, last stop +5min.
    // Intermediate stop: not contained -> delay is propagated.
    d.init_rtt(date::sys_days{2026_y / June / 30});
    auto siri_updater = auser(*d.tt_, d.tags_->get_src("test"),
                              nigiri::rt::vdv_aus::updater::xml_format::kSiri);
    auto const stats = siri_updater.consume_update(
        siri_update("2026-06-30T17:05:00+02:00", "2026-06-30T17:24:00+02:00"),
        *d.rt_->rtt_);
    EXPECT_EQ(1U, stats.matched_runs_);

    auto const trip_ep = utl::init_from<ep::trip>(d).value();
    auto const res = trip_ep(std::string{trip_id});
    ASSERT_EQ(1, res.legs_.size());
    auto const& leg = res.legs_.front();

    ASSERT_TRUE(leg.intermediateStops_.has_value());
    ASSERT_EQ(1, leg.intermediateStops_->size());
    auto const& intermediate = leg.intermediateStops_->at(0);
    EXPECT_EQ("15:13", format_time(intermediate.arrival_));
    EXPECT_EQ("15:14", format_time(intermediate.departure_));
    EXPECT_EQ(api::RealTimeStateEnum::PROPAGATED,
              intermediate.arrivalRealTimeState_);
    EXPECT_EQ(api::RealTimeStateEnum::PROPAGATED,
              intermediate.departureRealTimeState_);

    auto const stop_times = utl::init_from<ep::stop_times>(d).value();
    auto const st = stop_times(
        "/api/v5/stoptimes?stopId=test_87386763"
        "&time=2026-06-30T15:00:00.000Z"
        "&n=1");
    ASSERT_EQ(1, st.stopTimes_.size());
    auto const& dep = st.stopTimes_.front();
    EXPECT_TRUE(dep.realTime_);
    EXPECT_EQ("15:14", format_time(dep.place_.departure_));
    EXPECT_EQ(api::RealTimeStateEnum::PROPAGATED,
              dep.place_.departureRealTimeState_);

    // GTFS-RT export: unchanged delay is not repeated.
    EXPECT_EQ((std::vector<std::string>{"1: dep=300"}), gtfsrt_export(d));
  }

  {
    // First stop +5min, last stop +5min.
    // The update contains a stop that cannot be matched. Therefore, we don't
    // know whether the intermediate stop is contained in the update.
    // Intermediate stop: not matched -> the propagated delay is inconsistent
    // (returned by the API but not marked as real-time).
    d.init_rtt(date::sys_days{2026_y / June / 30});
    auto siri_updater = auser(*d.tt_, d.tags_->get_src("test"),
                              nigiri::rt::vdv_aus::updater::xml_format::kSiri);
    auto const stats = siri_updater.consume_update(
        siri_update("2026-06-30T17:05:00+02:00", "2026-06-30T17:24:00+02:00",
                    R"(
            <EstimatedCall>
              <StopPointRef>99999999</StopPointRef>
              <Order>2</Order>
              <AimedArrivalTime>2026-06-30T17:08:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2026-06-30T17:13:00+02:00</ExpectedArrivalTime>
              <AimedDepartureTime>2026-06-30T17:09:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2026-06-30T17:14:00+02:00</ExpectedDepartureTime>
            </EstimatedCall>)",
                    true),
        *d.rt_->rtt_);
    EXPECT_EQ(1U, stats.matched_runs_);

    auto const trip_ep = utl::init_from<ep::trip>(d).value();
    auto const res = trip_ep(std::string{trip_id});
    ASSERT_EQ(1, res.legs_.size());
    auto const& leg = res.legs_.front();
    EXPECT_TRUE(leg.realTime_);

    EXPECT_EQ("15:05", format_time(leg.from_.departure_));
    EXPECT_EQ(api::RealTimeStateEnum::PREDICTED,
              leg.from_.departureRealTimeState_);

    ASSERT_TRUE(leg.intermediateStops_.has_value());
    ASSERT_EQ(1, leg.intermediateStops_->size());
    auto const& intermediate = leg.intermediateStops_->at(0);
    EXPECT_EQ("15:13", format_time(intermediate.arrival_));
    EXPECT_EQ("15:14", format_time(intermediate.departure_));
    EXPECT_EQ(api::RealTimeStateEnum::INCONSISTENT,
              intermediate.arrivalRealTimeState_);
    EXPECT_EQ(api::RealTimeStateEnum::INCONSISTENT,
              intermediate.departureRealTimeState_);

    EXPECT_EQ("15:24", format_time(leg.to_.arrival_));
    EXPECT_EQ(api::RealTimeStateEnum::PREDICTED, leg.to_.arrivalRealTimeState_);

    auto const stop_times = utl::init_from<ep::stop_times>(d).value();
    auto const st = stop_times(
        "/api/v5/stoptimes?stopId=test_87386763"
        "&time=2026-06-30T15:00:00.000Z"
        "&n=1");
    ASSERT_EQ(1, st.stopTimes_.size());
    auto const& dep = st.stopTimes_.front();
    EXPECT_FALSE(dep.realTime_);
    EXPECT_EQ("15:14", format_time(dep.place_.departure_));
    EXPECT_EQ(api::RealTimeStateEnum::INCONSISTENT,
              dep.place_.departureRealTimeState_);

    // GTFS-RT export: intermediate stop without real-time data -> NO_DATA.
    EXPECT_EQ(
        (std::vector<std::string>{"1: dep=300", "2: NO_DATA", "3: arr=300"}),
        gtfsrt_export(d));

    EXPECT_EQ((std::map<std::string, double>{{"INCONSISTENT", 2.0},
                                             {"NO_RT_DATA", 0.0},
                                             {"OBSERVED", 0.0},
                                             {"PREDICTED", 2.0},
                                             {"PROPAGATED", 0.0}}),
              rt_state_metrics(d));
  }

  {
    // First stop +15min, last stop +1min.
    // Intermediate stop: not contained -> the propagated delay (+15min) would
    // be after the arrival at the last stop -> times are adjusted to 17:20.
    // Inconsistent times are returned by the API but not marked as real-time.
    d.init_rtt(date::sys_days{2026_y / June / 30});
    auto siri_updater = auser(*d.tt_, d.tags_->get_src("test"),
                              nigiri::rt::vdv_aus::updater::xml_format::kSiri);
    auto const stats = siri_updater.consume_update(
        siri_update("2026-06-30T17:15:00+02:00", "2026-06-30T17:20:00+02:00"),
        *d.rt_->rtt_);
    EXPECT_EQ(1U, stats.matched_runs_);

    auto const trip_ep = utl::init_from<ep::trip>(d).value();
    auto const res = trip_ep(std::string{trip_id});
    ASSERT_EQ(1, res.legs_.size());
    auto const& leg = res.legs_.front();

    ASSERT_TRUE(leg.intermediateStops_.has_value());
    ASSERT_EQ(1, leg.intermediateStops_->size());
    auto const& intermediate = leg.intermediateStops_->at(0);
    EXPECT_EQ("15:20", format_time(intermediate.arrival_));
    EXPECT_EQ("15:20", format_time(intermediate.departure_));
    EXPECT_EQ(api::RealTimeStateEnum::INCONSISTENT,
              intermediate.arrivalRealTimeState_);
    EXPECT_EQ(api::RealTimeStateEnum::INCONSISTENT,
              intermediate.departureRealTimeState_);

    auto const stop_times = utl::init_from<ep::stop_times>(d).value();
    auto const st = stop_times(
        "/api/v5/stoptimes?stopId=test_87386763"
        "&time=2026-06-30T15:00:00.000Z"
        "&n=1");
    ASSERT_EQ(1, st.stopTimes_.size());
    auto const& dep = st.stopTimes_.front();
    EXPECT_FALSE(dep.realTime_);
    EXPECT_EQ("15:20", format_time(dep.place_.departure_));
    EXPECT_EQ(api::RealTimeStateEnum::INCONSISTENT,
              dep.place_.departureRealTimeState_);

    // GTFS-RT export: inconsistent times are not exported -> NO_DATA.
    EXPECT_EQ(
        (std::vector<std::string>{"1: dep=900", "2: NO_DATA", "3: arr=60"}),
        gtfsrt_export(d));
  }

  {
    // Intermediate stop: arrival -2min, no expected departure time.
    // -> arrival predicted, departure without real-time data.
    d.init_rtt(date::sys_days{2026_y / June / 30});
    auto siri_updater = auser(*d.tt_, d.tags_->get_src("test"),
                              nigiri::rt::vdv_aus::updater::xml_format::kSiri);
    auto const stats = siri_updater.consume_update(
        siri_update("2026-06-30T17:00:00+02:00", "2026-06-30T17:19:00+02:00",
                    R"(
            <EstimatedCall>
              <StopPointRef>87386763</StopPointRef>
              <Order>2</Order>
              <AimedArrivalTime>2026-06-30T17:08:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2026-06-30T17:06:00+02:00</ExpectedArrivalTime>
              <AimedDepartureTime>2026-06-30T17:09:00+02:00</AimedDepartureTime>
            </EstimatedCall>)"),
        *d.rt_->rtt_);
    EXPECT_EQ(1U, stats.matched_runs_);

    auto const trip_ep = utl::init_from<ep::trip>(d).value();
    auto const res = trip_ep(std::string{trip_id});
    ASSERT_EQ(1, res.legs_.size());
    auto const& leg = res.legs_.front();

    ASSERT_TRUE(leg.intermediateStops_.has_value());
    ASSERT_EQ(1, leg.intermediateStops_->size());
    auto const& intermediate = leg.intermediateStops_->at(0);
    EXPECT_EQ("15:06", format_time(intermediate.arrival_));
    EXPECT_EQ("15:09", format_time(intermediate.departure_));
    EXPECT_EQ(api::RealTimeStateEnum::PREDICTED,
              intermediate.arrivalRealTimeState_);
    EXPECT_EQ(api::RealTimeStateEnum::NO_RT_DATA,
              intermediate.departureRealTimeState_);

    // GTFS-RT export: departure without real-time data is omitted.
    EXPECT_EQ((std::vector<std::string>{"1: dep=0", "2: arr=-120", "3: arr=0"}),
              gtfsrt_export(d));

    EXPECT_EQ((std::map<std::string, double>{{"INCONSISTENT", 0.0},
                                             {"NO_RT_DATA", 1.0},
                                             {"OBSERVED", 0.0},
                                             {"PREDICTED", 3.0},
                                             {"PROPAGATED", 0.0}}),
              rt_state_metrics(d));
  }

  {
    // Intermediate stop cancelled, no real-time data for the last stop.
    // GTFS-RT export: the cancelled stop is SKIPPED, NO_DATA has to be set
    // for the last stop (otherwise, the delay of the first stop would apply).
    d.init_rtt(date::sys_days{2026_y / June / 30});
    auto siri_updater = auser(*d.tt_, d.tags_->get_src("test"),
                              nigiri::rt::vdv_aus::updater::xml_format::kSiri);
    auto const stats =
        siri_updater.consume_update(siri_update("2026-06-30T17:05:00+02:00", "",
                                                R"(
            <EstimatedCall>
              <StopPointRef>87386763</StopPointRef>
              <Order>2</Order>
              <Cancellation>true</Cancellation>
              <AimedArrivalTime>2026-06-30T17:08:00+02:00</AimedArrivalTime>
              <AimedDepartureTime>2026-06-30T17:09:00+02:00</AimedDepartureTime>
            </EstimatedCall>)"),
                                    *d.rt_->rtt_);
    EXPECT_EQ(1U, stats.matched_runs_);
    EXPECT_EQ(
        (std::vector<std::string>{"1: dep=300", "2: SKIPPED", "3: NO_DATA"}),
        gtfsrt_export(d));
  }

  {
    // Only the first stop has an expected time.
    // GTFS-RT export: NO_DATA also applies to the last stop.
    // Map trips: segments are real-time if their departure or arrival is.
    d.init_rtt(date::sys_days{2026_y / June / 30});
    auto siri_updater = auser(*d.tt_, d.tags_->get_src("test"),
                              nigiri::rt::vdv_aus::updater::xml_format::kSiri);
    auto const stats =
        siri_updater.consume_update(siri_update("2026-06-30T17:05:00+02:00", "",
                                                R"(
            <EstimatedCall>
              <StopPointRef>87386763</StopPointRef>
              <Order>2</Order>
              <AimedArrivalTime>2026-06-30T17:08:00+02:00</AimedArrivalTime>
              <AimedDepartureTime>2026-06-30T17:09:00+02:00</AimedDepartureTime>
            </EstimatedCall>)"),
                                    *d.rt_->rtt_);
    EXPECT_EQ(1U, stats.matched_runs_);
    EXPECT_EQ((std::vector<std::string>{"1: dep=300", "2: NO_DATA"}),
              gtfsrt_export(d));

    d.rt_->railviz_rt_ =
        std::make_unique<railviz_rt_index>(*d.tt_, *d.rt_->rtt_);
    auto const trips_ep = utl::init_from<ep::trips>(d).value();
    auto const segments = trips_ep(
        "/api/v5/map/trips?zoom=18&min=48.9,1.6&max=49.1,2.0"
        "&startTime=2026-06-30T14:30:00Z&endTime=2026-06-30T16:00:00Z"
        "&precision=5");
    auto segment_rt = std::vector<std::string>{};
    for (auto const& segment : segments) {
      segment_rt.push_back(segment.from_.name_ + " -> " + segment.to_.name_ +
                           ": " + (segment.realTime_ ? "rt" : "no rt"));
    }
    EXPECT_EQ(
        (std::vector<std::string>{"Mantes-la-Jolie -> Épône - Mézières: rt",
                                  "Épône - Mézières -> Les Mureaux: no rt"}),
        segment_rt);
  }

  {
    // Only the intermediate stop has expected times.
    // Leg: neither the departure at the first stop nor the arrival at the last
    // stop has real-time data -> not real-time (even though the trip has).
    d.init_rtt(date::sys_days{2026_y / June / 30});
    auto siri_updater = auser(*d.tt_, d.tags_->get_src("test"),
                              nigiri::rt::vdv_aus::updater::xml_format::kSiri);
    auto const stats = siri_updater.consume_update(siri_update("", "", R"(
            <EstimatedCall>
              <StopPointRef>87386763</StopPointRef>
              <Order>2</Order>
              <AimedArrivalTime>2026-06-30T17:08:00+02:00</AimedArrivalTime>
              <ExpectedArrivalTime>2026-06-30T17:13:00+02:00</ExpectedArrivalTime>
              <AimedDepartureTime>2026-06-30T17:09:00+02:00</AimedDepartureTime>
              <ExpectedDepartureTime>2026-06-30T17:14:00+02:00</ExpectedDepartureTime>
            </EstimatedCall>)"),
                                                   *d.rt_->rtt_);
    EXPECT_EQ(1U, stats.matched_runs_);

    auto const trip_ep = utl::init_from<ep::trip>(d).value();
    auto const res = trip_ep(std::string{trip_id});
    ASSERT_EQ(1, res.legs_.size());
    auto const& leg = res.legs_.front();
    EXPECT_FALSE(leg.realTime_);
    EXPECT_EQ(api::RealTimeStateEnum::NO_RT_DATA,
              leg.from_.departureRealTimeState_);
    EXPECT_EQ(api::RealTimeStateEnum::NO_RT_DATA,
              leg.to_.arrivalRealTimeState_);

    ASSERT_TRUE(leg.intermediateStops_.has_value());
    ASSERT_EQ(1, leg.intermediateStops_->size());
    auto const& intermediate = leg.intermediateStops_->at(0);
    EXPECT_EQ("15:13", format_time(intermediate.arrival_));
    EXPECT_EQ(api::RealTimeStateEnum::PREDICTED,
              intermediate.arrivalRealTimeState_);
  }
}
