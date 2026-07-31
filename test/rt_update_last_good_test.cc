#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <initializer_list>
#include <string>
#include <string_view>
#include <system_error>

#include "boost/asio/io_context.hpp"
#include "date/date.h"
#include "prometheus/text_serializer.h"

#ifdef NO_DATA
#undef NO_DATA
#endif

#include "utl/init_from.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/stop_times.h"
#include "motis/import.h"
#include "motis/metrics_registry.h"
#include "motis/rt_update.h"

#include "util.h"

namespace {

namespace fs = std::filesystem;
using namespace std::chrono_literals;
using namespace date;
using namespace motis;
using namespace motis::test;
using namespace testing;

constexpr auto const kGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
test,Test,https://example.test,UTC

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
stop-1,Stop 1,50.061,19.938
stop-2,Stop 2,50.071,19.948

# routes.txt
route_id,agency_id,route_short_name,route_type
route-1,test,1,3

# trips.txt
route_id,service_id,trip_id,trip_headsign
route-1,service-1,trip-1,Stop 2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
trip-1,10:00:00,10:00:00,stop-1,1
trip-1,10:05:00,10:05:00,stop-2,2

# calendar_dates.txt
service_id,date,exception_type
service-1,{},1
)";

struct cwd_guard {
  explicit cwd_guard(fs::path cwd) : old_{fs::current_path()} {
    fs::current_path(std::move(cwd));
  }
  ~cwd_guard() { fs::current_path(old_); }
  fs::path old_;
};

void write_dump(std::string_view const bytes) {
  auto out = std::ofstream{"dump_rt/test-https___example_test_trip_updates",
                           std::ios::binary | std::ios::trunc};
  ASSERT_TRUE(out.good());
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void write_dump(std::string_view const endpoint, std::string_view const bytes) {
  auto out = std::ofstream{
      std::string{"dump_rt/test-https___example_test_"}.append(endpoint),
      std::ios::binary | std::ios::trunc};
  ASSERT_TRUE(out.good());
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

api::stoptimes_response query_stop_times(data& d) {
  auto const today =
      std::chrono::floor<date::days>(std::chrono::system_clock::now());
  return utl::init_from<ep::stop_times>(d).value()(
      std::format("/api/v5/stoptimes?stopId=test_stop-1"
                  "&time={}T09:00:00Z&n=1",
                  date::format("%F", today)));
}

double event_total(std::string const& metrics, std::string_view const event) {
  auto total = 0.0;
  auto const event_label = std::string{"event=\""}.append(event).append("\"");
  auto line_begin = std::size_t{0};
  while (line_begin != std::string::npos && line_begin < metrics.size()) {
    auto const line_end = metrics.find('\n', line_begin);
    auto const line = std::string_view{metrics}.substr(
        line_begin, line_end == std::string::npos ? std::string::npos
                                                  : line_end - line_begin);
    if (line.starts_with("nigiri_gtfsrt_source_events_total{") &&
        line.find(event_label) != std::string_view::npos) {
      total += std::stod(std::string{line.substr(line.rfind(' ') + 1)});
    }
    line_begin = line_end == std::string::npos ? line_end : line_end + 1U;
  }
  return total;
}

double metric_value(std::string const& metrics,
                    std::string_view const metric,
                    std::initializer_list<std::string_view> const labels) {
  auto line_begin = std::size_t{0};
  while (line_begin != std::string::npos && line_begin < metrics.size()) {
    auto const line_end = metrics.find('\n', line_begin);
    auto const line = std::string_view{metrics}.substr(
        line_begin, line_end == std::string::npos ? std::string::npos
                                                  : line_end - line_begin);
    auto const has_labels =
        std::ranges::all_of(labels, [&](std::string_view const label) {
          return line.find(label) != std::string_view::npos;
        });
    if (line.starts_with(metric) && has_labels) {
      return std::stod(std::string{line.substr(line.rfind(' ') + 1)});
    }
    line_begin = line_end == std::string::npos ? line_end : line_end + 1U;
  }
  return 0.0;
}

TEST(motis_rt_update, fresh_last_good_survives_bad_cycle_and_expires) {
  auto const test_dir = fs::absolute("test/data/rt-last-good");
  auto ec = std::error_code{};
  fs::remove_all(test_dir, ec);
  fs::create_directories(test_dir);
  auto cwd = cwd_guard{test_dir};
  auto const today =
      std::chrono::floor<date::days>(std::chrono::system_clock::now());
  auto const service_date = date::format("%Y%m%d", today);
  auto const first_day = date::format("%F", today);
  auto const gtfs = std::vformat(kGtfs, std::make_format_args(service_date));

  auto const c = config{
      .timetable_ = {config::timetable{
          .first_day_ = first_day,
          .num_days_ = 2,
          .update_interval_ = 1,
          .incremental_rt_update_ = true,
          .canned_rt_ = true,
          .datasets_ = {{"test",
                         {.path_ = gtfs,
                          .rt_ = {{{.url_ = "https://example.test/trip_updates",
                                    .last_good_ttl_ = 5U}}}}}}}}};
  import(c, "data");
  auto d = data{"data", c};
  fs::create_directory("dump_rt");

  auto good = to_feed_msg(
      {trip_update{.trip_ = {.trip_id_ = "trip-1",
                             .start_time_ = "10:00:00",
                             .date_ = service_date},
                   .stop_updates_ = {{.stop_id_ = "stop-1",
                                      .seq_ = 1U,
                                      .ev_type_ = nigiri::event_type::kDep,
                                      .delay_minutes_ = 10}}}},
      today + 9h);
  good.mutable_header()->clear_timestamp();

  auto ioc = boost::asio::io_context{};
  auto first_differential = good;
  first_differential.mutable_header()->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_DIFFERENTIAL);
  write_dump(first_differential.SerializeAsString());
  run_rt_update(ioc, c, d);
  ioc.run_for(100ms);
  auto const first_differential_departure =
      query_stop_times(d).stopTimes_.front().place_.departure_;
  EXPECT_TRUE(query_stop_times(d).stopTimes_.front().realTime_);

  write_dump("malformed");
  ioc.restart();
  ioc.run_for(1100ms);
  auto const after_first_differential_failure = query_stop_times(d);
  ASSERT_EQ(after_first_differential_failure.stopTimes_.size(), 1U);
  EXPECT_TRUE(after_first_differential_failure.stopTimes_.front().realTime_);
  EXPECT_EQ(
      first_differential_departure,
      after_first_differential_failure.stopTimes_.front().place_.departure_);

  write_dump(good.SerializeAsString());
  ioc.restart();
  ioc.run_for(1100ms);
  auto const after_good = query_stop_times(d);
  ASSERT_EQ(after_good.stopTimes_.size(), 1U);
  EXPECT_TRUE(after_good.stopTimes_.front().realTime_);
  auto const good_departure = after_good.stopTimes_.front().place_.departure_;

  auto differential = good;
  differential.mutable_header()->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_DIFFERENTIAL);
  differential.mutable_entity(0)
      ->mutable_trip_update()
      ->mutable_stop_time_update(0)
      ->mutable_departure()
      ->set_delay(15 * 60);
  write_dump(differential.SerializeAsString());
  ioc.restart();
  ioc.run_for(1100ms);
  auto const after_differential = query_stop_times(d);
  ASSERT_EQ(after_differential.stopTimes_.size(), 1U);
  EXPECT_TRUE(after_differential.stopTimes_.front().realTime_);
  EXPECT_NE(good_departure,
            after_differential.stopTimes_.front().place_.departure_);
  auto const differential_departure =
      after_differential.stopTimes_.front().place_.departure_;

  auto missing_header = transit_realtime::FeedMessage{};
  missing_header.add_entity()->set_id("missing-header");
  write_dump(missing_header.SerializePartialAsString());
  ioc.restart();
  ioc.run_for(1100ms);
  auto const after_missing_header = query_stop_times(d);
  EXPECT_TRUE(after_missing_header.stopTimes_.front().realTime_);
  EXPECT_EQ(differential_departure,
            after_missing_header.stopTimes_.front().place_.departure_);

  write_dump("");
  ioc.restart();
  ioc.run_for(1100ms);
  EXPECT_TRUE(query_stop_times(d).stopTimes_.front().realTime_);

  write_dump("malformed");
  ioc.restart();
  ioc.run_for(1100ms);
  EXPECT_TRUE(query_stop_times(d).stopTimes_.front().realTime_);

  ioc.restart();
  ioc.run_for(3100ms);
  EXPECT_FALSE(query_stop_times(d).stopTimes_.front().realTime_);
  auto const expired_metrics =
      prometheus::TextSerializer{}.Serialize(d.metrics_->registry_.Collect());
  EXPECT_EQ(0.0, metric_value(expired_metrics, "nigiri_gtfsrt_source_state{",
                              {"endpoint=\"0\"", "state=\"live\""}));
  EXPECT_EQ(1.0, metric_value(expired_metrics, "nigiri_gtfsrt_source_state{",
                              {"endpoint=\"0\"", "state=\"expired\""}));

  auto recovered = good;
  recovered.mutable_entity(0)
      ->mutable_trip_update()
      ->mutable_stop_time_update(0)
      ->mutable_departure()
      ->set_delay(20 * 60);
  write_dump(recovered.SerializeAsString());
  ioc.restart();
  ioc.run_for(1100ms);
  auto const after_recovery = query_stop_times(d);
  ASSERT_EQ(after_recovery.stopTimes_.size(), 1U);
  EXPECT_TRUE(after_recovery.stopTimes_.front().realTime_);
  EXPECT_NE(good_departure,
            after_recovery.stopTimes_.front().place_.departure_);

  auto authoritative_empty = transit_realtime::FeedMessage{};
  authoritative_empty.mutable_header()->set_gtfs_realtime_version("2.0");
  authoritative_empty.mutable_header()->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  write_dump(authoritative_empty.SerializeAsString());
  ioc.restart();
  ioc.run_for(1100ms);
  EXPECT_FALSE(query_stop_times(d).stopTimes_.front().realTime_);

  auto old_payload = recovered;
  old_payload.mutable_header()->set_timestamp(static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch() - 1min)
          .count()));
  write_dump(old_payload.SerializeAsString());
  ioc.restart();
  ioc.run_for(1100ms);
  EXPECT_FALSE(query_stop_times(d).stopTimes_.front().realTime_);
  auto const stale_metrics =
      prometheus::TextSerializer{}.Serialize(d.metrics_->registry_.Collect());
  EXPECT_EQ(0.0, metric_value(stale_metrics, "nigiri_gtfsrt_source_state{",
                              {"endpoint=\"0\"", "state=\"live\""}));
  EXPECT_EQ(1.0, metric_value(stale_metrics, "nigiri_gtfsrt_source_state{",
                              {"endpoint=\"0\"", "state=\"replay\""}));
  write_dump("malformed");
  ioc.restart();
  ioc.run_for(1100ms);
  EXPECT_FALSE(query_stop_times(d).stopTimes_.front().realTime_);

  ASSERT_TRUE(fs::remove("dump_rt/test-https___example_test_trip_updates"));
  ioc.restart();
  ioc.run_for(1100ms);

  auto const metrics =
      prometheus::TextSerializer{}.Serialize(d.metrics_->registry_.Collect());
  EXPECT_GT(event_total(metrics, "missing_header"), 0.0);
  EXPECT_GT(event_total(metrics, "empty_body"), 0.0);
  EXPECT_GT(event_total(metrics, "decode_error"), 0.0);
  EXPECT_GT(event_total(metrics, "fetch_error"), 0.0);
  EXPECT_GT(event_total(metrics, "last_good_reuse"), 0.0);
  EXPECT_GT(event_total(metrics, "last_good_expiry"), 0.0);
  EXPECT_GT(event_total(metrics, "recovery"), 0.0);
  EXPECT_THAT(metrics, HasSubstr("nigiri_gtfsrt_source_cache_age_seconds{"));
  EXPECT_THAT(metrics, HasSubstr("nigiri_gtfsrt_source_cache_fresh{"));
  EXPECT_THAT(metrics, HasSubstr("nigiri_gtfsrt_source_state{"));
  EXPECT_THAT(metrics, HasSubstr("endpoint=\"0\""));
  EXPECT_THAT(metrics, Not(HasSubstr("https___example_test")));
}

void test_good_malformed_good_recovery(bool const incremental) {
  auto const mode = incremental ? "incremental" : "full";
  auto const test_dir = fs::absolute(
      std::string{"test/data/rt-last-good-recovery-"}.append(mode));
  auto ec = std::error_code{};
  fs::remove_all(test_dir, ec);
  fs::create_directories(test_dir);
  auto cwd = cwd_guard{test_dir};
  auto const today =
      std::chrono::floor<date::days>(std::chrono::system_clock::now());
  auto const service_date = date::format("%Y%m%d", today);
  auto const first_day = date::format("%F", today);
  auto const gtfs = std::vformat(kGtfs, std::make_format_args(service_date));
  auto const c = config{
      .timetable_ = {config::timetable{
          .first_day_ = first_day,
          .num_days_ = 2,
          .update_interval_ = 1,
          .incremental_rt_update_ = incremental,
          .canned_rt_ = true,
          .datasets_ = {{"test",
                         {.path_ = gtfs,
                          .rt_ = {{{.url_ = "https://example.test/trip_updates",
                                    .last_good_ttl_ = 30U}}}}}}}}};
  import(c, "data");
  auto d = data{"data", c};
  fs::create_directory("dump_rt");

  auto good = to_feed_msg(
      {trip_update{.trip_ = {.trip_id_ = "trip-1",
                             .start_time_ = "10:00:00",
                             .date_ = service_date},
                   .stop_updates_ = {{.stop_id_ = "stop-1",
                                      .seq_ = 1U,
                                      .ev_type_ = nigiri::event_type::kDep,
                                      .delay_minutes_ = 10}}}},
      today + 9h);
  good.mutable_header()->clear_timestamp();

  write_dump(good.SerializeAsString());
  auto ioc = boost::asio::io_context{};
  run_rt_update(ioc, c, d);
  ioc.run_for(100ms);
  auto const after_good = query_stop_times(d);
  ASSERT_EQ(after_good.stopTimes_.size(), 1U);
  ASSERT_TRUE(after_good.stopTimes_.front().realTime_);
  auto const good_departure = after_good.stopTimes_.front().place_.departure_;

  write_dump("malformed");
  ioc.restart();
  ioc.run_for(1100ms);
  auto const after_malformed = query_stop_times(d);
  ASSERT_EQ(after_malformed.stopTimes_.size(), 1U);
  EXPECT_TRUE(after_malformed.stopTimes_.front().realTime_);
  EXPECT_EQ(good_departure,
            after_malformed.stopTimes_.front().place_.departure_);

  good.mutable_entity(0)
      ->mutable_trip_update()
      ->mutable_stop_time_update(0)
      ->mutable_departure()
      ->set_delay(20 * 60);
  write_dump(good.SerializeAsString());
  ioc.restart();
  ioc.run_for(1100ms);
  auto const after_recovery = query_stop_times(d);
  ASSERT_EQ(after_recovery.stopTimes_.size(), 1U);
  EXPECT_TRUE(after_recovery.stopTimes_.front().realTime_);
  EXPECT_NE(good_departure,
            after_recovery.stopTimes_.front().place_.departure_);
  auto const metrics =
      prometheus::TextSerializer{}.Serialize(d.metrics_->registry_.Collect());
  EXPECT_GT(event_total(metrics, "last_good_reuse"), 0.0);
  EXPECT_GT(event_total(metrics, "recovery"), 0.0);
}

TEST(motis_rt_update,
     good_malformed_good_recovers_in_full_and_incremental_modes) {
  for (auto const incremental : {false, true}) {
    SCOPED_TRACE(incremental ? "incremental" : "full");
    test_good_malformed_good_recovery(incremental);
  }
}

TEST(motis_rt_update, trip_update_and_vehicle_position_caches_are_independent) {
  auto const test_dir = fs::absolute("test/data/rt-last-good-mixed-gtfsrt");
  auto ec = std::error_code{};
  fs::remove_all(test_dir, ec);
  fs::create_directories(test_dir);
  auto cwd = cwd_guard{test_dir};
  auto const today =
      std::chrono::floor<date::days>(std::chrono::system_clock::now());
  auto const service_date = date::format("%Y%m%d", today);
  auto const first_day = date::format("%F", today);
  auto const gtfs = std::vformat(kGtfs, std::make_format_args(service_date));
  auto const c = config{
      .timetable_ = {config::timetable{
          .first_day_ = first_day,
          .num_days_ = 2,
          .update_interval_ = 1,
          .incremental_rt_update_ = true,
          .canned_rt_ = true,
          .datasets_ = {
              {"test",
               {.path_ = gtfs,
                .rt_ = {{{.url_ = "https://example.test/trip_updates",
                          .last_good_ttl_ = 30U},
                         {.url_ = "https://example.test/vehicle_positions",
                          .last_good_ttl_ = 30U}}}}}}}}};
  import(c, "data");
  auto d = data{"data", c};
  fs::create_directory("dump_rt");

  auto trip_updates = to_feed_msg(
      {trip_update{.trip_ = {.trip_id_ = "trip-1",
                             .start_time_ = "10:00:00",
                             .date_ = service_date},
                   .stop_updates_ = {{.stop_id_ = "stop-1",
                                      .seq_ = 1U,
                                      .ev_type_ = nigiri::event_type::kDep,
                                      .delay_minutes_ = 10}}}},
      today + 9h);
  trip_updates.mutable_header()->clear_timestamp();

  auto vehicle_positions = transit_realtime::FeedMessage{};
  vehicle_positions.mutable_header()->set_gtfs_realtime_version("2.0");
  vehicle_positions.mutable_header()->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  auto* vehicle_entity = vehicle_positions.add_entity();
  vehicle_entity->set_id("vehicle-1");
  vehicle_entity->mutable_vehicle()->mutable_trip()->set_trip_id("trip-1");
  vehicle_entity->mutable_vehicle()->mutable_position()->set_latitude(50.061F);
  vehicle_entity->mutable_vehicle()->mutable_position()->set_longitude(19.938F);

  write_dump("trip_updates", trip_updates.SerializeAsString());
  write_dump("vehicle_positions", vehicle_positions.SerializeAsString());
  auto ioc = boost::asio::io_context{};
  run_rt_update(ioc, c, d);
  ioc.run_for(100ms);

  trip_updates.mutable_entity(0)
      ->mutable_trip_update()
      ->mutable_stop_time_update(0)
      ->mutable_departure()
      ->set_delay(15 * 60);
  write_dump("trip_updates", trip_updates.SerializeAsString());
  write_dump("vehicle_positions", "malformed");
  ioc.restart();
  ioc.run_for(1100ms);
  auto after_bad_positions = query_stop_times(d);
  ASSERT_EQ(after_bad_positions.stopTimes_.size(), 1U);
  EXPECT_TRUE(after_bad_positions.stopTimes_.front().realTime_);
  auto const updated_departure =
      after_bad_positions.stopTimes_.front().place_.departure_;
  auto metrics =
      prometheus::TextSerializer{}.Serialize(d.metrics_->registry_.Collect());
  EXPECT_EQ(1.0, metric_value(metrics, "nigiri_gtfsrt_source_state{",
                              {"endpoint=\"0\"", "state=\"live\""}));
  EXPECT_EQ(1.0, metric_value(metrics, "nigiri_gtfsrt_source_state{",
                              {"endpoint=\"1\"", "state=\"replay\""}));

  write_dump("trip_updates", "malformed");
  write_dump("vehicle_positions", vehicle_positions.SerializeAsString());
  ioc.restart();
  ioc.run_for(1100ms);
  auto const after_bad_trip_updates = query_stop_times(d);
  ASSERT_EQ(after_bad_trip_updates.stopTimes_.size(), 1U);
  EXPECT_TRUE(after_bad_trip_updates.stopTimes_.front().realTime_);
  EXPECT_EQ(updated_departure,
            after_bad_trip_updates.stopTimes_.front().place_.departure_);
  metrics =
      prometheus::TextSerializer{}.Serialize(d.metrics_->registry_.Collect());
  EXPECT_EQ(1.0, metric_value(metrics, "nigiri_gtfsrt_source_state{",
                              {"endpoint=\"0\"", "state=\"replay\""}));
  EXPECT_EQ(1.0, metric_value(metrics, "nigiri_gtfsrt_source_state{",
                              {"endpoint=\"1\"", "state=\"live\""}));
}

TEST(motis_rt_update, gtfsrt_coexistence_preserves_auser_delta_state) {
  auto const test_dir = fs::absolute("test/data/rt-last-good-mixed-auser");
  auto ec = std::error_code{};
  fs::remove_all(test_dir, ec);
  fs::create_directories(test_dir);
  auto cwd = cwd_guard{test_dir};
  auto const today =
      std::chrono::floor<date::days>(std::chrono::system_clock::now());
  auto const service_date = date::format("%Y%m%d", today);
  auto const service_day = date::format("%F", today);
  auto const gtfs = std::vformat(kGtfs, std::make_format_args(service_date));
  auto const c = config{
      .timetable_ = {config::timetable{
          .first_day_ = service_day,
          .num_days_ = 2,
          .update_interval_ = 1,
          .incremental_rt_update_ = true,
          .canned_rt_ = true,
          .datasets_ = {
              {"test",
               {.path_ = gtfs,
                .rt_ = {{{.url_ = "https://example.test/auser",
                          .protocol_ =
                              config::timetable::dataset::rt::protocol::auser},
                         {.url_ = "https://example.test/trip_updates",
                          .last_good_ttl_ = 30U}}}}}}}}};
  import(c, "data");
  auto d = data{"data", c};
  fs::create_directory("dump_rt");

  auto const auser = std::format(
      R"(<?xml version="1.0" encoding="UTF-8"?>
<DatenAbrufenAntwort>
  <AUSNachricht AboID="1" auser_id="1">
    <IstFahrt Zst="{}T10:00:00">
      <LinienID>route-1</LinienID>
      <FahrtRef><FahrtID><FahrtBezeichner>trip-1</FahrtBezeichner><Betriebstag>{}</Betriebstag></FahrtID></FahrtRef>
      <Komplettfahrt>true</Komplettfahrt>
      <BetreiberID>test</BetreiberID>
      <IstHalt><HaltID>stop-1</HaltID><Abfahrtszeit>{}T10:00:00</Abfahrtszeit><IstAbfahrtPrognose>{}T10:07:00</IstAbfahrtPrognose></IstHalt>
      <IstHalt><HaltID>stop-2</HaltID><Ankunftszeit>{}T10:05:00</Ankunftszeit><IstAnkunftPrognose>{}T10:12:00</IstAnkunftPrognose></IstHalt>
      <Zusatzfahrt>false</Zusatzfahrt><FaelltAus>false</FaelltAus>
    </IstFahrt>
  </AUSNachricht>
</DatenAbrufenAntwort>)",
      service_day, service_day, service_day, service_day, service_day,
      service_day);
  auto empty_gtfsrt = transit_realtime::FeedMessage{};
  empty_gtfsrt.mutable_header()->set_gtfs_realtime_version("2.0");
  empty_gtfsrt.mutable_header()->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  write_dump("auser", auser);
  write_dump("trip_updates", empty_gtfsrt.SerializeAsString());

  auto ioc = boost::asio::io_context{};
  run_rt_update(ioc, c, d);
  ioc.run_for(100ms);
  auto const after_auser = query_stop_times(d);
  ASSERT_EQ(after_auser.stopTimes_.size(), 1U);
  ASSERT_TRUE(after_auser.stopTimes_.front().realTime_);
  auto const auser_departure = after_auser.stopTimes_.front().place_.departure_;

  ASSERT_TRUE(fs::remove("dump_rt/test-https___example_test_auser"));
  ioc.restart();
  ioc.run_for(1100ms);
  auto const after_missing_auser = query_stop_times(d);
  ASSERT_EQ(after_missing_auser.stopTimes_.size(), 1U);
  EXPECT_TRUE(after_missing_auser.stopTimes_.front().realTime_);
  EXPECT_EQ(auser_departure,
            after_missing_auser.stopTimes_.front().place_.departure_);
}

}  // namespace
