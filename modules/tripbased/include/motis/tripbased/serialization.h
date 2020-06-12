#pragma once

#include <cstdint>
#include <ctime>
#include <string>

#include "motis/core/schedule/schedule.h"
#include "motis/tripbased/data.h"

namespace motis::tripbased::serialization {

struct array_offset {
  array_offset() = default;
  array_offset(uint64_t start, uint64_t length)
      : start_(start), length_(length) {}

  uint64_t start_{};
  uint64_t length_{};
};

struct fws_multimap_offset {
  fws_multimap_offset() = default;
  fws_multimap_offset(array_offset const& index, array_offset const& data)
      : index_(index), data_(data) {}

  array_offset index_{};
  array_offset data_{};
};

struct header {
  uint64_t version_{};

  char schedule_name_[1024]{};
  int64_t schedule_begin_{};
  int64_t schedule_end_{};

  uint64_t trip_count_{};
  uint64_t line_count_{};

  // offsets
  array_offset line_to_first_trip_{};
  array_offset line_to_last_trip_{};
  array_offset trip_to_line_{};
  array_offset line_stop_count_{};

  fws_multimap_offset footpaths_{};
  fws_multimap_offset reverse_footpaths_{};
  fws_multimap_offset lines_at_stop_{};
  fws_multimap_offset stops_on_line_{};

  fws_multimap_offset arrival_times_{};
  array_offset departure_times_data_{};
  fws_multimap_offset transfers_{};
  fws_multimap_offset reverse_transfers_{};

  fws_multimap_offset in_allowed_{};
  array_offset out_allowed_data_{};
};

void write_data(tb_data const& data, std::string const& filename,
                schedule const& sched);

bool data_okay_for_schedule(std::string const& filename, schedule const& sched);

std::unique_ptr<tb_data> read_data(std::string const& filename,
                                   schedule const& sched);

}  // namespace motis::tripbased::serialization
