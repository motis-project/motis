#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

#include "boost/asio/io_context.hpp"
#include "boost/asio/ip/tcp.hpp"
#include "boost/asio/read_until.hpp"
#include "boost/asio/streambuf.hpp"
#include "boost/asio/write.hpp"
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

struct response_server {
  response_server(std::string body, std::chrono::milliseconds const delay)
      : acceptor_{ioc_, {boost::asio::ip::tcp::v4(), 0}},
        socket_{ioc_},
        thread_{[this, body = std::move(body), delay] {
          try {
            acceptor_.accept(socket_);
            auto request = boost::asio::streambuf{};
            boost::asio::read_until(socket_, request, "\r\n\r\n");
            std::this_thread::sleep_for(delay);
            auto response = std::format(
                "HTTP/1.1 200 OK\r\nContent-Type: application/x-protobuf\r\n"
                "Content-Length: {}\r\nConnection: close\r\n\r\n",
                body.size());
            response.append(body);
            boost::asio::write(socket_, boost::asio::buffer(response));
          } catch (...) {
          }
        }} {}

  ~response_server() {
    auto ec = boost::system::error_code{};
    acceptor_.cancel(ec);
    acceptor_.close(ec);
    socket_.cancel(ec);
    socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
    socket_.close(ec);
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  response_server(response_server const&) = delete;
  response_server& operator=(response_server const&) = delete;

  std::string url() const {
    return std::format("http://127.0.0.1:{}/rt",
                       acceptor_.local_endpoint().port());
  }

  boost::asio::io_context ioc_;
  boost::asio::ip::tcp::acceptor acceptor_;
  boost::asio::ip::tcp::socket socket_;
  std::jthread thread_;
};

void write_dump(std::string_view const bytes) {
  auto out = std::ofstream{"dump_rt/test-https___example_test_trip_updates",
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

double metric_total(std::string const& metrics, std::string_view const name) {
  auto total = 0.0;
  auto line_begin = std::size_t{0};
  while (line_begin != std::string::npos && line_begin < metrics.size()) {
    auto const line_end = metrics.find('\n', line_begin);
    auto const line = std::string_view{metrics}.substr(
        line_begin, line_end == std::string::npos ? std::string::npos
                                                  : line_end - line_begin);
    if (line.starts_with(name)) {
      total += std::stod(std::string{line.substr(line.rfind(' ') + 1)});
    }
    line_begin = line_end == std::string::npos ? line_end : line_end + 1U;
  }
  return total;
}

struct realtime_ingestion_test : TestWithParam<bool> {};

TEST_P(realtime_ingestion_test,
       applies_each_message_once_in_configuration_order) {
  auto const incremental = GetParam();
  auto const test_dir = fs::absolute(std::format(
      "test/data/rt-ingestion-{}", incremental ? "incremental" : "full"));
  auto ec = std::error_code{};
  fs::remove_all(test_dir, ec);
  fs::create_directories(test_dir);
  auto cwd = cwd_guard{test_dir};
  auto const today =
      std::chrono::floor<date::days>(std::chrono::system_clock::now());
  auto const service_date = date::format("%Y%m%d", today);
  auto const first_day = date::format("%F", today);
  auto const gtfs = std::vformat(kGtfs, std::make_format_args(service_date));

  auto first = to_feed_msg(
      {trip_update{.trip_ = {.trip_id_ = "trip-1",
                             .start_time_ = "10:00:00",
                             .date_ = service_date},
                   .stop_updates_ = {{.stop_id_ = "stop-1",
                                      .seq_ = 1U,
                                      .ev_type_ = nigiri::event_type::kDep,
                                      .delay_minutes_ = 10}}}},
      today + 9h);
  auto second = first;
  second.mutable_header()->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_DIFFERENTIAL);
  second.mutable_entity(0)
      ->mutable_trip_update()
      ->mutable_stop_time_update(0)
      ->mutable_departure()
      ->set_delay(15 * 60);

  // The second configured endpoint completes first. Application must still
  // follow configuration order after all responses have been collected.
  auto slow_first = response_server{first.SerializeAsString(), 100ms};
  auto fast_second = response_server{second.SerializeAsString(), 0ms};
  auto const c =
      config{.timetable_ = {config::timetable{
                 .first_day_ = first_day,
                 .num_days_ = 2,
                 .update_interval_ = 60,
                 .incremental_rt_update_ = incremental,
                 .datasets_ = {{"test",
                                {.path_ = gtfs,
                                 .rt_ = {{{.url_ = slow_first.url()},
                                          {.url_ = fast_second.url()}}}}}}}}};
  import(c, "data");
  auto d = data{"data", c};

  auto ioc = boost::asio::io_context{};
  run_rt_update(ioc, c, d);
  ioc.run_for(500ms);
  ioc.stop();

  auto const response = query_stop_times(d);
  ASSERT_EQ(response.stopTimes_.size(), 1U);
  EXPECT_TRUE(response.stopTimes_.front().realTime_);
  ASSERT_TRUE(response.stopTimes_.front().place_.departure_.has_value());
  EXPECT_EQ(static_cast<std::chrono::sys_seconds>(
                *response.stopTimes_.front().place_.departure_),
            today + 10h + 15min);

  auto const metrics =
      prometheus::TextSerializer{}.Serialize(d.metrics_->registry_.Collect());
  EXPECT_EQ(metric_total(metrics, "nigiri_gtfsrt_total_entities_total{"), 2.0);
  EXPECT_EQ(metric_total(metrics, "nigiri_gtfsrt_updates_successful_total{"),
            2.0);
}

INSTANTIATE_TEST_SUITE_P(full_and_incremental,
                         realtime_ingestion_test,
                         Values(false, true));

TEST(motis_rt_update,
     primary_and_fallback_apply_failures_do_not_stop_later_cycles) {
  auto const test_dir = fs::absolute("test/data/rt-apply-failure");
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
                                    .last_good_ttl_ = 60U}}}}}}}}};
  import(c, "data");
  auto d = data{"data", c};
  fs::create_directory("dump_rt");

  auto feed = to_feed_msg(
      {trip_update{.trip_ = {.trip_id_ = "trip-1",
                             .start_time_ = "10:00:00",
                             .date_ = service_date},
                   .stop_updates_ = {{.stop_id_ = "stop-1",
                                      .seq_ = 1U,
                                      .ev_type_ = nigiri::event_type::kDep,
                                      .delay_minutes_ = 10}}}},
      today + 9h);
  feed.mutable_header()->clear_timestamp();
  write_dump(feed.SerializeAsString());

  auto fail_applies = false;
  auto primary_failures = 0U;
  auto fallback_failures = 0U;
  auto hooks =
      rt_update_hooks{.after_gtfsrt_apply_ = [&](std::size_t const endpoint_idx,
                                                 bool const fallback) {
        EXPECT_EQ(endpoint_idx, 0U);
        if (!fail_applies) {
          return;
        }
        ++(fallback ? fallback_failures : primary_failures);
        throw std::runtime_error{fallback ? "injected fallback failure"
                                          : "injected primary failure"};
      }};

  auto ioc = boost::asio::io_context{};
  run_rt_update(ioc, c, d, std::move(hooks));
  ioc.run_for(100ms);
  EXPECT_TRUE(query_stop_times(d).stopTimes_.front().realTime_);

  feed.mutable_entity(0)
      ->mutable_trip_update()
      ->mutable_stop_time_update(0)
      ->mutable_departure()
      ->set_delay(15 * 60);
  write_dump(feed.SerializeAsString());
  fail_applies = true;
  ioc.restart();
  ioc.run_for(1100ms);
  EXPECT_EQ(primary_failures, 1U);
  EXPECT_EQ(fallback_failures, 1U);
  EXPECT_FALSE(query_stop_times(d).stopTimes_.front().realTime_);

  feed.mutable_entity(0)
      ->mutable_trip_update()
      ->mutable_stop_time_update(0)
      ->mutable_departure()
      ->set_delay(20 * 60);
  write_dump(feed.SerializeAsString());
  fail_applies = false;
  ioc.restart();
  ioc.run_for(1100ms);
  auto const recovered = query_stop_times(d);
  ASSERT_EQ(recovered.stopTimes_.size(), 1U);
  EXPECT_TRUE(recovered.stopTimes_.front().realTime_);
  ASSERT_TRUE(recovered.stopTimes_.front().place_.departure_.has_value());
  EXPECT_EQ(static_cast<std::chrono::sys_seconds>(
                *recovered.stopTimes_.front().place_.departure_),
            today + 10h + 20min);
  ioc.stop();
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
  EXPECT_TRUE(query_stop_times(d).stopTimes_.front().realTime_);

  write_dump("malformed");
  ioc.restart();
  ioc.run_for(1100ms);
  EXPECT_FALSE(query_stop_times(d).stopTimes_.front().realTime_);

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
  EXPECT_TRUE(query_stop_times(d).stopTimes_.front().realTime_);
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

}  // namespace
