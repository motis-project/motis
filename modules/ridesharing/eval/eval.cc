#include <time.h>

#include <ctime>

#include "boost/geometry.hpp"

#include "motis/core/common/logging.h"
#include "motis/core/common/timing.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"

#include "motis/module/message.h"

#include "motis/ridesharing/routing_result.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "./eval_super_itest.h"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "geo/constants.h"
#include "geo/detail/register_latlng.h"
#include "geo/latlng.h"
#include "gtest/gtest.h"
#include "utl/parser/csv.h"

using namespace geo;
using namespace flatbuffers;
using namespace motis::osrm;
using namespace motis::test;
using namespace motis::test::schedule;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::intermodal;
using motis::logging::info;
using motis::test::schedule::simple_realtime::dataset_opt;

namespace motis {
namespace ridesharing {

constexpr long SECONDS_PER_DAY = 60 * 60 * 24;
constexpr long SECONDS_PER_WEEK = 60 * 60 * 24 * 7;
using rs_csv = std::tuple<std::string, bool, std::string, std::string, bool,
                          bool, bool, bool, bool, bool, bool, std::string,
                          std::string, double, double, double, double>;

struct eval_initial_itest : public eval_super_itest {
  uint64_t no_journey_{};
  uint64_t price_improvement_{};
  uint64_t price_worse_{};
  uint64_t duration_improvement_{};
  uint64_t duration_worse_{};
  uint64_t same_journey_{};
  uint64_t new_journey_{};
  double total_duration_improvement_{};
  double total_duration_improvement_only_{};
  double total_price_improvement_{};
  uint64_t same_leg_{};
  uint64_t different_leg_{};
  uint64_t door_in_{};
  uint64_t door_out_{};
  uint64_t direct_connection_{};
  uint64_t weekend_{};
  uint64_t weekday_{};
  int max_c = 10000;
  eval_initial_itest() : eval_super_itest() {}

  long convert_to_timeframe(rs_csv const& rs) {
    struct tm tm;
    memset(&tm, 0, sizeof(tm));
    strptime(std::get<11>(rs).c_str(), "%Y-%m-%d %H:%M:%S %z", &tm);
    time_t ts = mktime(&tm);
    return ts % (3600 * 24) + unix_time(100);
  }

  void print_stats() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
    auto const fname = oss.str();
    std::ofstream stats_write;
    stats_write.open(fname);

    auto const stats_msg = call(ridesharing_stats());
    auto const stats = motis_content(RidesharingStatistics, stats_msg);

    for (auto const& e : *stats->stats()->entries()) {
      stats_write << e->name()->str() << ": " << e->value() << std::endl;
    }
    stats_write << "no_journey_: " << no_journey_ << std::endl;
    stats_write << "duration_improvement_: " << duration_improvement_
                << std::endl;
    stats_write << "duration_worse_: " << duration_worse_ << std::endl;
    stats_write << "same_journey_: " << same_journey_ << std::endl;
    stats_write << "new_journey_: " << new_journey_ << std::endl;
    stats_write << "same_leg_: " << same_leg_ << std::endl;
    stats_write << "different_leg_: " << different_leg_ << std::endl;
    stats_write << "new_journey_: " << new_journey_ << std::endl;
    stats_write << "weekend_: " << weekend_ << std::endl;
    stats_write << "weekday_: " << weekday_ << std::endl;
    stats_write << "door_in_: " << door_in_ << std::endl;
    stats_write << "door_out_: " << door_out_ << std::endl;
    stats_write << "direct_connection_: " << direct_connection_ << std::endl;
    stats_write << "total_duration_improvement: " << total_duration_improvement_
                << std::endl;
    stats_write << "avg_duration_improvement: "
                << total_duration_improvement_ /
                       (duration_improvement_ + duration_worse_ + same_journey_)
                << std::endl;
    stats_write << "total_duration_improvement_only: "
                << total_duration_improvement_only_ << std::endl;
    stats_write << "avg_duration_improvement_only: "
                << total_duration_improvement_only_ / (duration_improvement_)
                << std::endl;
  }

  void call_ridesharing_book(RidesharingEdge const* rsedge, int& i) {
    auto const lk = rsedge->lift_key()->str();
    auto const delim = std::string{";"};
    auto const it = lk.find(delim);
    auto const time_lift_start = std::stol(lk.substr(0, it));
    auto const driver = std::stoi(lk.substr(it + 1));
    call(ridesharing_book(
        driver, time_lift_start, ++i,
        {rsedge->from_pos()->lat(), rsedge->from_pos()->lng()},
        {rsedge->to_pos()->lat(), rsedge->to_pos()->lng()}, rsedge->from_leg(),
        rsedge->to_leg(), time_lift_start + rsedge->rs_duration() + 15 * 60));
  }

  void call_ridesharing_book(journey::transport const& t, int& i,
                             long const required_arrival) {
    auto const lk = t.provider_;
    auto const delim = std::string{";"};
    auto const it = lk.find(delim);
    auto const time_lift_start = std::stol(lk.substr(0, it));
    auto const driver = std::stoi(lk.substr(it + 1));
    if (t.from_leg_ != t.to_leg_) {
      ++different_leg_;
    } else {
      ++same_leg_;
    }
    call(ridesharing_book(driver, time_lift_start, ++i,
                          {t.from_loc_lat_, t.from_loc_lng_},
                          {t.to_loc_lat_, t.to_loc_lng_}, t.from_leg_,
                          t.to_leg_, required_arrival));
  }

  void execute_query_and_book(long const t, rs_csv const& e, int& i) {
    auto const res =
        call(ridesharing_edges((long)t, {std::get<13>(e), std::get<14>(e)},
                               {std::get<15>(e), std::get<16>(e)}));
    auto const content = motis_content(RidesharingResponse, res);
    if (content->direct_connections()->size() != 0) {
      call_ridesharing_book(content->direct_connections()->Get(0), i);
    } else if (content->arrs()->size() != 0) {
      call_ridesharing_book(content->arrs()->Get(0), i);
    } else if (content->deps()->size() != 0) {
      call_ridesharing_book(content->deps()->Get(0), i);
    }
  }

  void execute_intermodal_and_compare(long const t, rs_csv const& e, int& i,
                                      bool weekday) {
    auto res = call(
        execute_intermodal_pure((long)t, {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
    auto content = motis_content(RoutingResponse, res);
    auto journeys = message_to_journeys(content);
    bool no_journey = journeys.empty();
    uint16_t duration{};
    if (!no_journey) {
      auto const fast = std::min_element(begin(journeys), end(journeys),
                                         [](auto const& j1, auto const& j2) {
                                           return j1.duration_ < j2.duration_;
                                         });
      duration = (*fast).duration_;
    }
    MOTIS_START_TIMING(individual_intermodal);
    res = call(execute_intermodal((long)t, {std::get<13>(e), std::get<14>(e)},
                                  {std::get<15>(e), std::get<16>(e)}));
    MOTIS_STOP_TIMING(individual_intermodal);
    std::ofstream air_dist_to_query_time(
        "d_to_t.txt", std::ios_base::app | std::ios_base::out);
    air_dist_to_query_time << distance({std::get<13>(e), std::get<14>(e)},
                                       {std::get<15>(e), std::get<16>(e)})
                           << "," << MOTIS_TIMING_US(individual_intermodal)
                           << "\n";
    content = motis_content(RoutingResponse, res);
    journeys = message_to_journeys(content);
    if (journeys.empty()) {
      ++no_journey_;
      return;
    }
    auto const fast = *std::min_element(begin(journeys), end(journeys),
                                        [](auto const& j1, auto const& j2) {
                                          return j1.duration_ < j2.duration_;
                                        });
    if (no_journey) {
      ++new_journey_;
    } else {
      if (duration > fast.duration_) {
        total_duration_improvement_only_ += ((double)fast.duration_) / duration;
        ++duration_improvement_;
      } else if (duration < fast.duration_) {
        ++duration_worse_;
      } else {
        ++same_journey_;
      }
      if (duration > 0.000001) {
        total_duration_improvement_ += ((double)fast.duration_) / duration;
      }
    }
    mark_door_or_direct(fast);
    book_ridesharing_on_journey(fast, i, weekday);
  }

  void execute_intermodal_and_compare_delete(long const t, rs_csv const& e,
                                             int& i, bool weekday) {
    auto res = call(
        execute_intermodal_pure((long)t, {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
    auto content = motis_content(RoutingResponse, res);
    auto journeys = message_to_journeys(content);
    bool no_journey = journeys.empty();
    uint16_t duration{};
    if (!no_journey) {
      auto const fast = std::min_element(begin(journeys), end(journeys),
                                         [](auto const& j1, auto const& j2) {
                                           return j1.duration_ < j2.duration_;
                                         });
      duration = (*fast).duration_;
    }
    LOG(logging::info) << "Post Pure";
    MOTIS_START_TIMING(individual_intermodal);
    res = call(execute_intermodal((long)t, {std::get<13>(e), std::get<14>(e)},
                                  {std::get<15>(e), std::get<16>(e)}));
    MOTIS_STOP_TIMING(individual_intermodal);
    LOG(logging::info) << "Post inter";
    std::ofstream air_dist_to_query_time(
        "d_to_t.txt", std::ios_base::app | std::ios_base::out);
    air_dist_to_query_time << distance({std::get<13>(e), std::get<14>(e)},
                                       {std::get<15>(e), std::get<16>(e)})
                           << "," << MOTIS_TIMING_US(individual_intermodal)
                           << "\n";
    content = motis_content(RoutingResponse, res);
    journeys = message_to_journeys(content);
    if (journeys.empty()) {
      ++no_journey_;
      return;
    }
    auto const fast = *std::min_element(begin(journeys), end(journeys),
                                        [](auto const& j1, auto const& j2) {
                                          return j1.duration_ < j2.duration_;
                                        });
    if (no_journey) {
      ++new_journey_;
    } else {
      if (duration > fast.duration_) {
        total_duration_improvement_only_ += ((double)fast.duration_) / duration;
        ++duration_improvement_;
      } else if (duration < fast.duration_) {
        ++duration_worse_;
      } else {
        ++same_journey_;
      }
      total_duration_improvement_ += ((double)fast.duration_) / duration;
    }
    LOG(logging::info) << "Pre-Mark";
    mark_door_or_direct(fast);
    for (auto const& t : fast.transports_) {
      if (t.mumo_type_ == "ridesharing") {
        auto const lk = t.provider_;
        auto const delim = std::string{";"};
        auto const it = lk.find(delim);
        auto const time_lift_start = std::stol(lk.substr(0, it));
        auto const driver = std::stoi(lk.substr(it + 1));
        LOG(logging::info) << "Pre-Delete";
        call(ridesharing_remove(driver, time_lift_start));
      }
    }
    LOG(logging::info) << "DONE";
  }

  void execute_intermodal_and_book(long const t, rs_csv const& e, int& i,
                                   bool weekday) {
    MOTIS_START_TIMING(individual_intermodal);
    auto const res =
        call(execute_intermodal((long)t, {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
    MOTIS_STOP_TIMING(individual_intermodal);
    std::ofstream air_dist_to_query_time(
        "d_to_t.txt", std::ios_base::app | std::ios_base::out);
    air_dist_to_query_time << distance({std::get<13>(e), std::get<14>(e)},
                                       {std::get<15>(e), std::get<16>(e)})
                           << "," << MOTIS_TIMING_US(individual_intermodal)
                           << "\n";
    auto const content = motis_content(RoutingResponse, res);
    auto const journeys = message_to_journeys(content);
    if (journeys.empty()) {
      ++no_journey_;
      return;
    }
    auto const fast = *std::min_element(begin(journeys), end(journeys),
                                        [](auto const& j1, auto const& j2) {
                                          return j1.duration_ < j2.duration_;
                                        });
    mark_door_or_direct(fast);
    book_ridesharing_on_journey(fast, i, weekday);
  }

  void book_ridesharing_on_journey(journey const& fast, int& i, bool weekday) {
    LOG(logging::info) << "Pre-Booking Search";
    for (auto it = fast.transports_.begin(); it != fast.transports_.end();
         it++) {
      if (it->mumo_type_ == "ridesharing") {
        if (weekday) {
          ++weekday_;
        } else {
          ++weekend_;
        }
        auto nx = std::next(it);
        if (nx == fast.transports_.end()) {
          call_ridesharing_book(
              (*it), i, fast.stops_.back().arrival_.timestamp_ + 15 * 60);
        } else {
          if (nx->mumo_type_ == "foot") {
            call_ridesharing_book(
                (*it), i,
                std::prev(fast.stops_.end())->arrival_.timestamp_ + 15 * 60);
          } else {
            auto const stop_it = std::find_if(
                fast.stops_.begin(), fast.stops_.end(), [&](auto const& s) {
                  return it->to_loc_lat_ == s.lat_ && it->to_loc_lng_ == s.lng_;
                });
            if (stop_it == fast.stops_.end()) {
              LOG(logging::info) << "Stop could not be found in journey.";
              call_ridesharing_book(
                  (*it), i, fast.stops_.back().arrival_.timestamp_ + 15 * 60);
            } else {
              call_ridesharing_book((*it), i, stop_it->departure_.timestamp_);
            }
          }
        }
      }
    }
    LOG(logging::info) << "Post-Booking Search";
  }

  void mark_door_or_direct(journey const& j) {
    if (j.transports_.size() == 1 &&
        j.transports_.front().mumo_type_ == "ridesharing") {
      ++direct_connection_;
      return;
    }
    if (j.transports_.front().mumo_type_ == "ridesharing") {
      ++door_in_;
    }
    if (j.transports_.back().mumo_type_ == "ridesharing") {
      ++door_out_;
    }
  }

  void clear(std::vector<rs_csv> const& ridesharing_data) {
    int i = 0;
    int c = 0;
    for (auto const e : ridesharing_data) {
      if (std::get<0>(e) == "RideOffer") {
        std::cout << c++ << std::endl;
        auto t = convert_to_timeframe(e);
        if (std::get<4>(e)) {
          call(ridesharing_remove(++i, (long)t));
        }
        t += SECONDS_PER_DAY;
        if (std::get<5>(e)) {
          call(ridesharing_remove(++i, (long)t));
        }
        t += SECONDS_PER_DAY;
        if (std::get<6>(e)) {
          call(ridesharing_remove(++i, (long)t));
        }
        t += SECONDS_PER_DAY;
        if (std::get<7>(e)) {
          call(ridesharing_remove(++i, (long)t));
        }
        t += SECONDS_PER_DAY;
        if (std::get<8>(e)) {
          call(ridesharing_remove(++i, (long)t));
        }
        t += SECONDS_PER_DAY;
        if (std::get<9>(e)) {
          call(ridesharing_remove(++i, (long)t));
        }
        t += SECONDS_PER_DAY;
        if (std::get<10>(e)) {
          call(ridesharing_remove(++i, (long)t));
        }
      }
    }
  }

  // void execute_intermodal_and_book_price(long const t, rs_csv const& e, int&
  // i) {
  //   auto const res = call(execute_intermodal((long)t, {std::get<13>(e),
  //   std::get<14>(e)}, {std::get<15>(e), std::get<16>(e)})); auto const
  //   content = motis_content(RoutingResponse, res); auto const journeys =
  //   message_to_journeys(content); if (journeys.empty()) {
  //     ++no_journey_;
  //     return;
  //   }
  //   auto const fast = std::min_element(begin(journeys), end(journeys),
  //   [](auto const& j1, auto const& j2) {
  //     return j1.price_ < j2.price_;
  //   });
  //   for (auto const t : (*fast).transports_) {
  //     if (t.mumo_type_ == "ridesharing") {
  //       call_ridesharing_book(t, i);
  //     }
  //   }
  // }

  // void execute_intermodal_and_book_access(long const t, rs_csv const& e, int&
  // i) {
  //   auto const res = call(execute_intermodal((long)t, {std::get<13>(e),
  //   std::get<14>(e)}, {std::get<15>(e), std::get<16>(e)})); auto const
  //   content = motis_content(RoutingResponse, res); auto const journeys =
  //   message_to_journeys(content); if (journeys.empty()) {
  //     ++no_journey_;
  //     return;
  //   }
  //   auto const fast = std::min_element(begin(journeys), end(journeys),
  //   [](auto const& j1, auto const& j2) {
  //     return j1.accessibility_ < j2.accessibility_;
  //   });
  //   for (auto const t : (*fast).transports_) {
  //     if (t.mumo_type_ == "ridesharing") {
  //       call_ridesharing_book(t, i);
  //     }
  //   }
  // }
};

TEST_F(eval_initial_itest, initialize_break) {
  print_stats();

  std::vector<int> v{};
  v[4] = v[5];
}

TEST_F(eval_initial_itest, initialize_db) {
  utl::column_mapping<rs_csv> const ridesharing_columns = {{
      "ride_type", "recurring", "recurring_beginning_date",
      "recurring_ending_date",  // 3
      "recurring_monday", "recurring_tuesday", "recurring_wednesday",
      "recurring_thursday",  // 7
      "recurring_friday", "recurring_saturday", "recurring_sunday",  // 10
      "first_departure_time", "first_arrival_time",  // 12
      "start_geo_lat", "start_geo_lng", "destination_geo_lat",
      "destination_geo_lng"  // 16
  }};

  std::vector<rs_csv> ridesharing_data;
  utl::read_file<rs_csv>("rides.csv", ridesharing_data, ridesharing_columns);

  int week_rs = 0;
  int week_ro = 0;
  for (auto const e : ridesharing_data) {
    if (std::get<4>(e) && std::get<5>(e) && std::get<6>(e) && std::get<7>(e) &&
        std::get<8>(e)) {
      if (std::get<0>(e) == "RideSearch") {
        week_rs++;
      } else {
        week_ro++;
      }
    }
  }
  LOG(logging::info) << "Weekday stuff: " << week_ro << " -> "
                     << ((double)week_ro) / 6040 << " " << week_rs << " -> "
                     << ((double)week_rs) / 3960;

  print_stats();
}

TEST_F(eval_initial_itest, clear) {
  std::chrono::milliseconds timespan(40000);
  std::this_thread::sleep_for(timespan);
  LOG(logging::info) << "Starting test endpoints";
  message_creator mc;

  utl::column_mapping<rs_csv> const ridesharing_columns = {{
      "ride_type", "recurring", "recurring_beginning_date",
      "recurring_ending_date",  // 3
      "recurring_monday", "recurring_tuesday", "recurring_wednesday",
      "recurring_thursday",  // 7
      "recurring_friday", "recurring_saturday", "recurring_sunday",  // 10
      "first_departure_time", "first_arrival_time",  // 12
      "start_geo_lat", "start_geo_lng", "destination_geo_lat",
      "destination_geo_lng"  // 16
  }};

  std::vector<rs_csv> ridesharing_data;
  utl::read_file<rs_csv>("rides.csv", ridesharing_data, ridesharing_columns);
  int c = 0;
  int i = 0;
  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideOffer") {
      std::cout << c++ << std::endl;
      auto t = convert_to_timeframe(e);
      if (std::get<4>(e)) {
        call(ridesharing_remove(++i, (long)t));
        call(ridesharing_remove(++i, (long)t + SECONDS_PER_WEEK));
      }
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        call(ridesharing_remove(++i, (long)t));
        call(ridesharing_remove(++i, (long)t + SECONDS_PER_WEEK));
      }
      t += SECONDS_PER_DAY;
      if (std::get<6>(e)) {
        call(ridesharing_remove(++i, (long)t));
        call(ridesharing_remove(++i, (long)t + SECONDS_PER_WEEK));
      }
      t += SECONDS_PER_DAY;
      if (std::get<7>(e)) {
        call(ridesharing_remove(++i, (long)t));
        call(ridesharing_remove(++i, (long)t + SECONDS_PER_WEEK));
      }
      t += SECONDS_PER_DAY;
      if (std::get<8>(e)) {
        call(ridesharing_remove(++i, (long)t));
        call(ridesharing_remove(++i, (long)t + SECONDS_PER_WEEK));
      }
      t += SECONDS_PER_DAY;
      if (std::get<9>(e)) {
        call(ridesharing_remove(++i, (long)t));
        call(ridesharing_remove(++i, (long)t + SECONDS_PER_WEEK));
      }
      t += SECONDS_PER_DAY;
      if (std::get<10>(e)) {
        call(ridesharing_remove(++i, (long)t));
        call(ridesharing_remove(++i, (long)t + SECONDS_PER_WEEK));
      }
    }
  }
  auto const stats_msg = call(ridesharing_stats());
  auto const stats = motis_content(RidesharingStatistics, stats_msg);
  for (auto const& e : *stats->stats()->entries()) {
    LOG(logging::info) << e->name()->str() << ": " << e->value();
  }
  LOG(logging::info) << instance_->schedule_->schedule_begin_ << " - "
                     << instance_->schedule_->schedule_end_;
  auto station_locations = std::vector<geo::latlng>{};
  for (auto const& s : instance_->schedule_->stations_) {
    if (s->eva_nr_.rfind("80", 0) == 0) {
      station_locations.emplace_back(s->lat(), s->lng());
    }
  }
  LOG(logging::info) << unix_time(0);
  LOG(logging::info) << "Starting test endpoints";
}

TEST_F(eval_initial_itest, endpoints) {
  std::chrono::milliseconds timespan(40000);
  std::this_thread::sleep_for(timespan);
  LOG(logging::info) << "Starting test endpoints";
  message_creator mc;
  std::ofstream progress;
  progress.open("progress");

  utl::column_mapping<rs_csv> const ridesharing_columns = {{
      "ride_type", "recurring", "recurring_beginning_date",
      "recurring_ending_date",  // 3
      "recurring_monday", "recurring_tuesday", "recurring_wednesday",
      "recurring_thursday",  // 7
      "recurring_friday", "recurring_saturday", "recurring_sunday",  // 10
      "first_departure_time", "first_arrival_time",  // 12
      "start_geo_lat", "start_geo_lng", "destination_geo_lat",
      "destination_geo_lng"  // 16
  }};

  std::vector<rs_csv> ridesharing_data;
  utl::read_file<rs_csv>("rides.csv", ridesharing_data, ridesharing_columns);

  int c = 0;
  int i = 0;

  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideOffer") {
      progress << c++ << std::endl;
      if (c == max_c) break;

      auto t = convert_to_timeframe(e);
      if (std::get<4>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
        call(ridesharing_create(++i, (long)t + SECONDS_PER_WEEK,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
        call(ridesharing_create(++i, (long)t + SECONDS_PER_WEEK,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<6>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
        call(ridesharing_create(++i, (long)t + SECONDS_PER_WEEK,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<7>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
        call(ridesharing_create(++i, (long)t + SECONDS_PER_WEEK,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<8>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
        call(ridesharing_create(++i, (long)t + SECONDS_PER_WEEK,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<9>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
        call(ridesharing_create(++i, (long)t + SECONDS_PER_WEEK,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<10>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
        call(ridesharing_create(++i, (long)t + SECONDS_PER_WEEK,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
    }
  }
  c = 0;

  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideSearch") {
      progress << c++ << std::endl;
      if (c == max_c) break;
      auto t = convert_to_timeframe(e);
      if (std::get<4>(e)) {
        execute_query_and_book(t, e, i);
        execute_query_and_book(t + SECONDS_PER_WEEK, e, i);
      }
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        execute_query_and_book(t, e, i);
        execute_query_and_book(t + SECONDS_PER_WEEK, e, i);
      }
      t += SECONDS_PER_DAY;
      if (std::get<6>(e)) {
        execute_query_and_book(t, e, i);
        execute_query_and_book(t + SECONDS_PER_WEEK, e, i);
      }
      t += SECONDS_PER_DAY;
      if (std::get<7>(e)) {
        execute_query_and_book(t, e, i);
        execute_query_and_book(t + SECONDS_PER_WEEK, e, i);
      }
      t += SECONDS_PER_DAY;
      if (std::get<8>(e)) {
        execute_query_and_book(t, e, i);
        execute_query_and_book(t + SECONDS_PER_WEEK, e, i);
      }
      t += SECONDS_PER_DAY;
      if (std::get<9>(e)) {
        execute_query_and_book(t, e, i);
        execute_query_and_book(t + SECONDS_PER_WEEK, e, i);
      }
      t += SECONDS_PER_DAY;
      if (std::get<10>(e)) {
        execute_query_and_book(t, e, i);
        execute_query_and_book(t + SECONDS_PER_WEEK, e, i);
      }
    }
  }
  i = 0;
  c = 0;
  int p = 10000;
  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideOffer") {
      progress << c++ << std::endl;
      if (c == max_c) break;
      auto t = convert_to_timeframe(e);
      if (std::get<4>(e)) {
        call(ridesharing_book(
            ++i, (long)t, ++p, {std::get<13>(e) - 0.01, std::get<14>(e) - 0.01},
            {std::get<15>(e) - 0.01, std::get<16>(e) - 0.01}, 0, 0));
        call(ridesharing_book(++i, (long)t + SECONDS_PER_WEEK, ++p,
                              {std::get<13>(e) - 0.01, std::get<14>(e) - 0.01},
                              {std::get<15>(e) - 0.01, std::get<16>(e) - 0.01},
                              0, 0));
      }
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        call(ridesharing_book(
            ++i, (long)t, ++p, {std::get<13>(e) - 0.01, std::get<14>(e) - 0.01},
            {std::get<15>(e) - 0.01, std::get<16>(e) - 0.01}, 0, 0));
        call(ridesharing_book(++i, (long)t + SECONDS_PER_WEEK, ++p,
                              {std::get<13>(e) - 0.01, std::get<14>(e) - 0.01},
                              {std::get<15>(e) - 0.01, std::get<16>(e) - 0.01},
                              0, 0));
      }
      t += SECONDS_PER_DAY;
      if (std::get<6>(e)) {
        call(ridesharing_book(
            ++i, (long)t, ++p, {std::get<13>(e) - 0.01, std::get<14>(e) - 0.01},
            {std::get<15>(e) - 0.01, std::get<16>(e) - 0.01}, 0, 0));
        call(ridesharing_book(++i, (long)t + SECONDS_PER_WEEK, ++p,
                              {std::get<13>(e) - 0.01, std::get<14>(e) - 0.01},
                              {std::get<15>(e) - 0.01, std::get<16>(e) - 0.01},
                              0, 0));
      }
      t += SECONDS_PER_DAY;
      if (std::get<7>(e)) {
        call(ridesharing_book(
            ++i, (long)t, ++p, {std::get<13>(e) - 0.01, std::get<14>(e) - 0.01},
            {std::get<15>(e) - 0.01, std::get<16>(e) - 0.01}, 0, 0));
        call(ridesharing_book(++i, (long)t + SECONDS_PER_WEEK, ++p,
                              {std::get<13>(e) - 0.01, std::get<14>(e) - 0.01},
                              {std::get<15>(e) - 0.01, std::get<16>(e) - 0.01},
                              0, 0));
      }
      t += SECONDS_PER_DAY;
      if (std::get<8>(e)) {
        call(ridesharing_book(
            ++i, (long)t, ++p, {std::get<13>(e) - 0.01, std::get<14>(e) - 0.01},
            {std::get<15>(e) - 0.01, std::get<16>(e) - 0.01}, 0, 0));
        call(ridesharing_book(++i, (long)t + SECONDS_PER_WEEK, ++p,
                              {std::get<13>(e) - 0.01, std::get<14>(e) - 0.01},
                              {std::get<15>(e) - 0.01, std::get<16>(e) - 0.01},
                              0, 0));
      }
      t += SECONDS_PER_DAY;
      if (std::get<9>(e)) {
        call(ridesharing_book(
            ++i, (long)t, ++p, {std::get<13>(e) - 0.01, std::get<14>(e) - 0.01},
            {std::get<15>(e) - 0.01, std::get<16>(e) - 0.01}, 0, 0));
        call(ridesharing_book(++i, (long)t + SECONDS_PER_WEEK, ++p,
                              {std::get<13>(e) - 0.01, std::get<14>(e) - 0.01},
                              {std::get<15>(e) - 0.01, std::get<16>(e) - 0.01},
                              0, 0));
      }
      t += SECONDS_PER_DAY;
      if (std::get<10>(e)) {
        call(ridesharing_book(
            ++i, (long)t, ++p, {std::get<13>(e) - 0.01, std::get<14>(e) - 0.01},
            {std::get<15>(e) - 0.01, std::get<16>(e) - 0.01}, 0, 0));
        call(ridesharing_book(++i, (long)t + SECONDS_PER_WEEK, ++p,
                              {std::get<13>(e) - 0.01, std::get<14>(e) - 0.01},
                              {std::get<15>(e) - 0.01, std::get<16>(e) - 0.01},
                              0, 0));
      }
    }
  }
  i = 0;
  c = 0;
  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideOffer") {
      progress << c++ << std::endl;
      if (c == max_c) break;
      auto t = convert_to_timeframe(e);
      if (std::get<4>(e)) {
        call(ridesharing_remove(++i, (long)t));
        call(ridesharing_remove(++i, (long)t + SECONDS_PER_WEEK));
      }
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        call(ridesharing_remove(++i, (long)t));
        call(ridesharing_remove(++i, (long)t + SECONDS_PER_WEEK));
      }
      t += SECONDS_PER_DAY;
      if (std::get<6>(e)) {
        call(ridesharing_remove(++i, (long)t));
        call(ridesharing_remove(++i, (long)t + SECONDS_PER_WEEK));
      }
      t += SECONDS_PER_DAY;
      if (std::get<7>(e)) {
        call(ridesharing_remove(++i, (long)t));
        call(ridesharing_remove(++i, (long)t + SECONDS_PER_WEEK));
      }
      t += SECONDS_PER_DAY;
      if (std::get<8>(e)) {
        call(ridesharing_remove(++i, (long)t));
        call(ridesharing_remove(++i, (long)t + SECONDS_PER_WEEK));
      }
      t += SECONDS_PER_DAY;
      if (std::get<9>(e)) {
        call(ridesharing_remove(++i, (long)t));
        call(ridesharing_remove(++i, (long)t + SECONDS_PER_WEEK));
      }
      t += SECONDS_PER_DAY;
      if (std::get<10>(e)) {
        call(ridesharing_remove(++i, (long)t));
        call(ridesharing_remove(++i, (long)t + SECONDS_PER_WEEK));
      }
    }
  }
  auto const stats_msg = call(ridesharing_stats());
  auto const stats = motis_content(RidesharingStatistics, stats_msg);
  for (auto const& e : *stats->stats()->entries()) {
    LOG(logging::info) << e->name()->str() << ": " << e->value();
  }
  LOG(logging::info) << instance_->schedule_->schedule_begin_ << " - "
                     << instance_->schedule_->schedule_end_;
  auto station_locations = std::vector<geo::latlng>{};
  for (auto const& s : instance_->schedule_->stations_) {
    if (s->eva_nr_.rfind("80", 0) == 0) {
      station_locations.emplace_back(s->lat(), s->lng());
    }
  }
  LOG(logging::info) << unix_time(0);
  LOG(logging::info) << "Starting test endpoints";
}

TEST_F(eval_initial_itest, basic) {
  LOG(logging::info) << "Starting test basic";
  message_creator mc;
  std::chrono::milliseconds timespan(20000);
  std::this_thread::sleep_for(timespan);
  std::ofstream progress;
  progress.open("progress");
  utl::column_mapping<rs_csv> const ridesharing_columns = {{
      "ride_type", "recurring", "recurring_beginning_date",
      "recurring_ending_date",  // 3
      "recurring_monday", "recurring_tuesday", "recurring_wednesday",
      "recurring_thursday",  // 7
      "recurring_friday", "recurring_saturday", "recurring_sunday",  // 10
      "first_departure_time", "first_arrival_time",  // 12
      "start_geo_lat", "start_geo_lng", "destination_geo_lat",
      "destination_geo_lng"  // 16
  }};

  std::vector<rs_csv> ridesharing_data;
  utl::read_file<rs_csv>("rides.csv", ridesharing_data, ridesharing_columns);

  int c = 0;
  int i = 0;
  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideOffer") {
      std::cout << c++ << std::endl;
      auto t = convert_to_timeframe(e);
      if (std::get<4>(e)) {
        call(ridesharing_remove(++i, (long)t));
      }
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        call(ridesharing_remove(++i, (long)t));
      }
      t += SECONDS_PER_DAY;
      if (std::get<6>(e)) {
        call(ridesharing_remove(++i, (long)t));
      }
      t += SECONDS_PER_DAY;
      if (std::get<7>(e)) {
        call(ridesharing_remove(++i, (long)t));
      }
      t += SECONDS_PER_DAY;
      if (std::get<8>(e)) {
        call(ridesharing_remove(++i, (long)t));
      }
      t += SECONDS_PER_DAY;
      if (std::get<9>(e)) {
        call(ridesharing_remove(++i, (long)t));
      }
      t += SECONDS_PER_DAY;
      if (std::get<10>(e)) {
        call(ridesharing_remove(++i, (long)t));
      }
    }
  }
  i = 0;
  c = 0;

  progress << "Creation" << std::endl;
  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideOffer") {
      progress << c << std::endl;
      std::cout << "\n\n\n\n\n" << c << std::endl;
      if (++c == max_c) {
        break;
      }
      auto t = convert_to_timeframe(e);
      if (std::get<4>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<6>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<7>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<8>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<9>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<10>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
    }
  }
  c = 0;
  progress << "Queries" << std::endl;
  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideSearch") {
      progress << c << std::endl;
      std::cout << "\n\n\n\n\n" << c << std::endl;
      if (++c == max_c) {
        break;
      }
      auto t = convert_to_timeframe(e);
      if (std::get<4>(e)) {
        execute_intermodal_and_book(t, e, i, true);
        execute_intermodal_and_book(t + SECONDS_PER_WEEK, e, i, true);
      }
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        execute_intermodal_and_book(t, e, i, true);
        execute_intermodal_and_book(t + SECONDS_PER_WEEK, e, i, true);
      }
      t += SECONDS_PER_DAY;
      if (std::get<6>(e)) {
        execute_intermodal_and_book(t, e, i, true);
        execute_intermodal_and_book(t + SECONDS_PER_WEEK, e, i, true);
      }
      t += SECONDS_PER_DAY;
      if (std::get<7>(e)) {
        execute_intermodal_and_book(t, e, i, true);
        execute_intermodal_and_book(t + SECONDS_PER_WEEK, e, i, true);
      }
      t += SECONDS_PER_DAY;
      if (std::get<8>(e)) {
        execute_intermodal_and_book(t, e, i, true);
        execute_intermodal_and_book(t + SECONDS_PER_WEEK, e, i, true);
      }
      t += SECONDS_PER_DAY;
      if (std::get<9>(e)) {
        execute_intermodal_and_book(t, e, i, false);
        execute_intermodal_and_book(t + SECONDS_PER_WEEK, e, i, false);
      }
      t += SECONDS_PER_DAY;
      if (std::get<10>(e)) {
        execute_intermodal_and_book(t, e, i, false);
        execute_intermodal_and_book(t + SECONDS_PER_WEEK, e, i, false);
      }
    }
  }
  print_stats();
  LOG(logging::info) << "Finished test basic";
}

TEST_F(eval_initial_itest, basic_small) {
  LOG(logging::info) << "Starting test basic";
  message_creator mc;
  std::chrono::milliseconds timespan(20000);
  std::this_thread::sleep_for(timespan);
  std::ofstream progress;
  progress.open("progress");
  utl::column_mapping<rs_csv> const ridesharing_columns = {{
      "ride_type", "recurring", "recurring_beginning_date",
      "recurring_ending_date",  // 3
      "recurring_monday", "recurring_tuesday", "recurring_wednesday",
      "recurring_thursday",  // 7
      "recurring_friday", "recurring_saturday", "recurring_sunday",  // 10
      "first_departure_time", "first_arrival_time",  // 12
      "start_geo_lat", "start_geo_lng", "destination_geo_lat",
      "destination_geo_lng"  // 16
  }};

  std::vector<rs_csv> ridesharing_data;
  utl::read_file<rs_csv>("rides.csv", ridesharing_data, ridesharing_columns);

  int c = 0;
  int i = 0;
  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideOffer") {
      std::cout << c++ << std::endl;
      auto t = convert_to_timeframe(e);
      if (std::get<4>(e)) {
        call(ridesharing_remove(++i, (long)t));
      }
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        call(ridesharing_remove(++i, (long)t));
      }
      t += SECONDS_PER_DAY;
      if (std::get<6>(e)) {
        call(ridesharing_remove(++i, (long)t));
      }
    }
  }
  i = 0;
  c = 0;

  progress << "Creation" << std::endl;
  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideOffer") {
      progress << c << std::endl;
      std::cout << "\n\n\n\n\n" << c << std::endl;
      if (++c == max_c) {
        break;
      }
      auto t = convert_to_timeframe(e);
      if (std::get<4>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<6>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
    }
  }
  c = 0;
  progress << "Queries" << std::endl;
  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideSearch") {
      progress << c << std::endl;
      std::cout << "\n\n\n\n\n" << c << std::endl;
      if (++c == max_c) {
        break;
      }
      auto t = convert_to_timeframe(e);
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        execute_intermodal_and_book(t, e, i, true);
      }
      t += SECONDS_PER_DAY;
    }
  }
  print_stats();
  LOG(logging::info) << "Finished test basic";
}

TEST_F(eval_initial_itest, compare) {
  LOG(logging::info) << "Starting test compare";
  message_creator mc;
  std::chrono::milliseconds timespan(20000);
  std::this_thread::sleep_for(timespan);
  std::ofstream progress;
  progress.open("progress");
  utl::column_mapping<rs_csv> const ridesharing_columns = {{
      "ride_type", "recurring", "recurring_beginning_date",
      "recurring_ending_date",  // 3
      "recurring_monday", "recurring_tuesday", "recurring_wednesday",
      "recurring_thursday",  // 7
      "recurring_friday", "recurring_saturday", "recurring_sunday",  // 10
      "first_departure_time", "first_arrival_time",  // 12
      "start_geo_lat", "start_geo_lng", "destination_geo_lat",
      "destination_geo_lng"  // 16
  }};

  std::vector<rs_csv> ridesharing_data;
  utl::read_file<rs_csv>("ride_export.csv", ridesharing_data,
                         ridesharing_columns);

  clear(ridesharing_data);
  int c = 0;
  int i = 0;

  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideOffer") {
      progress << c << std::endl;
      if (++c == max_c) {
        break;
      }
      auto t = convert_to_timeframe(e);
      if (std::get<4>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<6>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<7>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<8>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<9>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<10>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
    }
  }
  c = 0;

  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideSearch") {
      progress << c << std::endl;
      if (++c == max_c) {
        break;
      }
      auto t = convert_to_timeframe(e);
      if (std::get<4>(e)) {
        execute_intermodal_and_compare(t, e, i, true);
      }
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        execute_intermodal_and_compare(t, e, i, true);
      }
      t += SECONDS_PER_DAY;
      if (std::get<6>(e)) {
        execute_intermodal_and_compare(t, e, i, true);
      }
      t += SECONDS_PER_DAY;
      if (std::get<7>(e)) {
        execute_intermodal_and_compare(t, e, i, true);
      }
      t += SECONDS_PER_DAY;
      if (std::get<8>(e)) {
        execute_intermodal_and_compare(t, e, i, true);
      }
      t += SECONDS_PER_DAY;
      if (std::get<9>(e)) {
        execute_intermodal_and_compare(t, e, i, false);
      }
      t += SECONDS_PER_DAY;
      if (std::get<10>(e)) {
        execute_intermodal_and_compare(t, e, i, false);
      }
    }
  }
  print_stats();
}

TEST_F(eval_initial_itest, compare_small) {
  LOG(logging::info) << "Starting test compare";
  message_creator mc;
  std::chrono::milliseconds timespan(20000);
  std::this_thread::sleep_for(timespan);
  utl::column_mapping<rs_csv> const ridesharing_columns = {{
      "ride_type", "recurring", "recurring_beginning_date",
      "recurring_ending_date",  // 3
      "recurring_monday", "recurring_tuesday", "recurring_wednesday",
      "recurring_thursday",  // 7
      "recurring_friday", "recurring_saturday", "recurring_sunday",  // 10
      "first_departure_time", "first_arrival_time",  // 12
      "start_geo_lat", "start_geo_lng", "destination_geo_lat",
      "destination_geo_lng"  // 16
  }};

  std::vector<rs_csv> ridesharing_data;
  utl::read_file<rs_csv>("rides.csv", ridesharing_data, ridesharing_columns);

  clear(ridesharing_data);
  int c = 0;
  int i = 0;

  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideOffer") {
      std::cout << "\n\n\n\n\n" << c << std::endl;
      if (++c == max_c) {
        break;
      }
      auto t = convert_to_timeframe(e);
      if (std::get<4>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<6>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
    }
  }
  c = 0;

  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideSearch") {
      std::cout << "\n\n\n\n\n" << c << std::endl;
      if (++c == max_c) {
        break;
      }
      auto t = convert_to_timeframe(e);
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        execute_intermodal_and_compare(t, e, i, true);
      }
    }
  }
  print_stats();
}

TEST_F(eval_initial_itest, compare_delete) {
  LOG(logging::info) << "Starting test compare";
  message_creator mc;
  std::chrono::milliseconds timespan(20000);
  std::this_thread::sleep_for(timespan);
  std::ofstream stats_write;
  stats_write.open("progress");

  utl::column_mapping<rs_csv> const ridesharing_columns = {{
      "ride_type", "recurring", "recurring_beginning_date",
      "recurring_ending_date",  // 3
      "recurring_monday", "recurring_tuesday", "recurring_wednesday",
      "recurring_thursday",  // 7
      "recurring_friday", "recurring_saturday", "recurring_sunday",  // 10
      "first_departure_time", "first_arrival_time",  // 12
      "start_geo_lat", "start_geo_lng", "destination_geo_lat",
      "destination_geo_lng"  // 16
  }};

  std::vector<rs_csv> ridesharing_data;
  utl::read_file<rs_csv>("rides.csv", ridesharing_data, ridesharing_columns);

  clear(ridesharing_data);
  int c = 0;
  int i = 0;

  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideOffer") {
      std::cout << "\n\n\n\n\n" << c << std::endl;
      if (++c == max_c) {
        break;
      }
      auto t = convert_to_timeframe(e);
      if (std::get<4>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<6>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
    }
  }
  c = 0;

  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideSearch") {
      std::cout << "\n\n\n\n\n" << c << std::endl;
      if (++c == max_c) {
        break;
      }
      auto t = convert_to_timeframe(e);
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        execute_intermodal_and_compare_delete(t, e, i, true);
      }
    }
  }
  print_stats();
}

TEST_F(eval_initial_itest, removal) {
  LOG(logging::info) << "Starting test compare";
  message_creator mc;
  std::chrono::milliseconds timespan(20000);
  std::this_thread::sleep_for(timespan);
  std::ofstream progress;
  progress.open("progress");
  utl::column_mapping<rs_csv> const ridesharing_columns = {{
      "ride_type", "recurring", "recurring_beginning_date",
      "recurring_ending_date",  // 3
      "recurring_monday", "recurring_tuesday", "recurring_wednesday",
      "recurring_thursday",  // 7
      "recurring_friday", "recurring_saturday", "recurring_sunday",  // 10
      "first_departure_time", "first_arrival_time",  // 12
      "start_geo_lat", "start_geo_lng", "destination_geo_lat",
      "destination_geo_lng"  // 16
  }};

  std::vector<rs_csv> ridesharing_data;
  utl::read_file<rs_csv>("ride_export.csv", ridesharing_data,
                         ridesharing_columns);

  int c = 0;
  int i = 0;

  for (auto const e : ridesharing_data) {
    if (std::get<0>(e) == "RideOffer") {
      progress << c << std::endl;
      if (++c == max_c) {
        break;
      }
      auto t = convert_to_timeframe(e);
      if (std::get<4>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<5>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<6>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<7>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<8>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<9>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
      t += SECONDS_PER_DAY;
      if (std::get<10>(e)) {
        call(ridesharing_create(++i, (long)t,
                                {std::get<13>(e), std::get<14>(e)},
                                {std::get<15>(e), std::get<16>(e)}));
      }
    }
  }
  clear(ridesharing_data);
  print_stats();
}

}  // namespace ridesharing
}  // namespace motis
