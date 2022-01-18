#pragma once

#include "motis/protocol/Statistics_generated.h"

namespace motis::ridesharing {

struct rs_statistics {
  rs_statistics() = default;

  uint64_t total_creation_time_{};
  uint64_t total_deletion_time_{};
  uint64_t total_booking_time_{};
  uint64_t total_query_time_{};
  uint64_t total_routing_time_query_{};
  uint64_t total_edges_time_query_{};
  uint64_t total_close_station_time_query_{};
  uint64_t total_routing_time_book_{};
  uint64_t total_routing_time_create_{};
  uint64_t creations_{};
  uint64_t deletions_{};
  uint64_t bookings_{};
  uint64_t queries_{};
  uint64_t invalid_lifts_{};
  uint64_t requests_without_edges_{};
  uint64_t total_edges_constructed_{};
  uint64_t direct_edges_constructed_{};
  uint64_t door_in_edges_constructed_{};
  uint64_t door_out_edges_constructed_{};
  uint64_t two_passenger_{};
  uint64_t three_passenger_{};
  uint64_t four_passenger_{};
  uint64_t rest_passenger_{};
  uint64_t max_passengers_{};
  uint64_t parking_time_{};
  uint64_t init_time_{};
  uint64_t parking_time_db_{};
  uint64_t parking_time_not_db_{};
  uint64_t parking_db_{};
  uint64_t parking_not_db_{};

  friend flatbuffers::Offset<Statistics> to_fbs(
      flatbuffers::FlatBufferBuilder& fbb, char const* category,
      rs_statistics const& s) {
    std::vector<flatbuffers::Offset<StatisticsEntry>> stats{};

    auto const add_entry = [&](char const* key, auto const val) {
      stats.emplace_back(
          CreateStatisticsEntry(fbb, fbb.CreateString(key), val));
    };

    add_entry("parking_time_db_", s.parking_time_db_);
    add_entry("parking_time_not_db_", s.parking_time_not_db_);
    add_entry("parking_db_", s.parking_db_);
    add_entry("parking_not_db_", s.parking_not_db_);
    add_entry("total_creation_time_", s.total_creation_time_);
    add_entry("total_routing_time_query_", s.total_routing_time_query_);
    add_entry("total_edges_time_query_", s.total_edges_time_query_);
    add_entry("total_close_station_time_query_",
              s.total_close_station_time_query_);
    add_entry("total_routing_time_book_", s.total_routing_time_book_);
    add_entry("total_routing_time_create_", s.total_routing_time_create_);
    add_entry("total_deletion_time_", s.total_deletion_time_);
    add_entry("total_booking_time_", s.total_booking_time_);
    add_entry("total_query_time_", s.total_query_time_);
    add_entry("creations_", s.creations_);
    add_entry("deletions_", s.deletions_);
    add_entry("bookings_", s.bookings_);
    add_entry("queries_", s.queries_);
    add_entry("invalid_lifts_", s.invalid_lifts_);
    add_entry("requests_without_edges_", s.requests_without_edges_);
    add_entry("total_edges_constructed_", s.total_edges_constructed_);
    add_entry("two_passenger_", s.two_passenger_);
    add_entry("three_passenger_", s.three_passenger_);
    add_entry("four_passenger_", s.four_passenger_);
    add_entry("rest_passenger_", s.rest_passenger_);
    add_entry("max_passengers_", s.max_passengers_);
    add_entry("parking_time_", s.parking_time_);
    add_entry("init_time_", s.init_time_);

    return CreateStatistics(fbb, fbb.CreateString(category),
                            fbb.CreateVectorOfSortedTables(&stats));
  }
};

}  // namespace motis::ridesharing
