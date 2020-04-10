#include "motis/path/prepare/schedule/schedule_wrapper.h"

#include <iostream>

#include "boost/filesystem.hpp"

#include "utl/parser/file.h"

#include "motis/path/prepare/fbs/use_64bit_flatbuffers.h"

#include "motis/schedule-format/Schedule_generated.h"

namespace fs = boost::filesystem;
using namespace motis::loader;

namespace motis::path {

schedule_wrapper::schedule_wrapper(std::string const& schedule_path) {
  auto sched_file = fs::path(schedule_path) / "schedule.raw";
  if (!fs::is_regular_file(sched_file)) {
    throw std::runtime_error("cannot open schedule.raw");
  }

  schedule_buffer_ = utl::file(sched_file.string().c_str(), "r").content();
}

std::vector<station_seq> schedule_wrapper::load_station_sequences() const {
  return motis::path::load_station_sequences(
      GetSchedule(schedule_buffer_.buf_));
}

}  // namespace motis::path
