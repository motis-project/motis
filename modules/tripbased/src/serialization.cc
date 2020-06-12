#include "motis/tripbased/serialization.h"

#include <cstdio>
#include <cstring>

#include "boost/filesystem.hpp"

#include "utl/enumerate.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"

namespace fs = boost::filesystem;
using namespace motis::logging;

namespace motis::tripbased::serialization {

constexpr uint64_t CURRENT_VERSION = 11;

struct file {
  file(char const* path, char const* mode) : f_(std::fopen(path, mode)) {
    utl::verify(f_ != nullptr, "unable to open file");
  }

  ~file() {
    if (f_ != nullptr) {
      fclose(f_);
    }
    f_ = nullptr;
  }

  file(file const&) = delete;
  file& operator=(file const&) = delete;

  file(file&&) = delete;
  file& operator=(file&&) = delete;

  std::size_t size() const {
    auto const err = std::fseek(f_, 0, SEEK_END);
    utl::verify(err == 0, "fseek to SEEK_END error");
    auto const size = std::ftell(f_);
    std::rewind(f_);
    return static_cast<std::size_t>(size);
  }

  void write(void const* buf, std::size_t size) const {
    auto const bytes_written = std::fwrite(buf, 1, size, f_);
    utl::verify(bytes_written == size, "file write error");
  }

  void read(void* buf, std::size_t offset, std::size_t size) const {
    auto const err = std::fseek(f_, offset, SEEK_SET);
    utl::verify(err == 0, "fseek error");
    auto const bytes_read = std::fread(buf, 1, size, f_);
    utl::verify(bytes_read == size, "file read error");
  }

  explicit operator FILE*() const { return f_; }

  FILE* f_;
};

std::string serialize_schedule_names(schedule const& sched) {
  std::stringstream ss;
  for (auto const& [i, name] : utl::enumerate(sched.names_)) {
    if (i != 0) {
      ss << "\n";
    }
    ss << name;
  }
  return ss.str();
}

template <typename T>
void set_array_offset(uint64_t& current_offset, array_offset& off,
                      mcd::vector<T> const& data) {
  off.start_ = current_offset;
  off.length_ = data.size() * sizeof(T);
  current_offset += off.length_;
}

template <typename T, typename Index>
void set_fws_multimap_offset(uint64_t& current_offset, fws_multimap_offset& off,
                             fws_multimap<T, Index> const& map) {
  set_array_offset(current_offset, off.index_, map.index_);
  set_array_offset(current_offset, off.data_, map.data_);
}

template <typename T, typename Index>
void set_fws_multimap_offset(uint64_t& current_offset, fws_multimap_offset& off,
                             nested_fws_multimap<T, Index> const& map) {
  set_array_offset(current_offset, off.index_, map.index_);
  set_array_offset(current_offset, off.data_, map.data_);
}

template <typename T>
void write_array(file& f, mcd::vector<T> const& data) {
  if (!data.empty()) {
    f.write(data.data(), data.size() * sizeof(T));
  }
}

template <typename T, typename Index>
void write_fws_multimap(file& f, fws_multimap<T, Index> const& map) {
  write_array(f, map.index_);
  write_array(f, map.data_);
}

template <typename T, typename Index>
void write_fws_multimap(file& f, nested_fws_multimap<T, Index> const& map) {
  write_array(f, map.index_);
  write_array(f, map.data_);
}

void write_data(tb_data const& data, std::string const& filename,
                schedule const& sched) {
  file f(filename.c_str(), "wb");

  header h{};
  h.version_ = CURRENT_VERSION;
  auto const& schedule_name = serialize_schedule_names(sched);
  if (schedule_name.size() > sizeof(h.schedule_name_)) {
    LOG(warn) << "tripbased: serialized schedule name to long";
  }
  std::strncpy(h.schedule_name_, schedule_name.data(),
               std::min(static_cast<size_t>(schedule_name.size()),
                        sizeof(h.schedule_name_)));

  h.schedule_begin_ = static_cast<int64_t>(sched.schedule_begin_);
  h.schedule_end_ = static_cast<int64_t>(sched.schedule_end_);
  h.trip_count_ = data.trip_count_;
  h.line_count_ = data.line_count_;

  uint64_t offset = sizeof(header);

  set_array_offset(offset, h.line_to_first_trip_, data.line_to_first_trip_);
  set_array_offset(offset, h.line_to_last_trip_, data.line_to_last_trip_);
  set_array_offset(offset, h.trip_to_line_, data.trip_to_line_);
  set_array_offset(offset, h.line_stop_count_, data.line_stop_count_);

  set_fws_multimap_offset(offset, h.footpaths_, data.footpaths_);
  set_fws_multimap_offset(offset, h.reverse_footpaths_,
                          data.reverse_footpaths_);
  set_fws_multimap_offset(offset, h.lines_at_stop_, data.lines_at_stop_);
  set_fws_multimap_offset(offset, h.stops_on_line_, data.stops_on_line_);

  set_fws_multimap_offset(offset, h.arrival_times_, data.arrival_times_);
  set_array_offset(offset, h.departure_times_data_,
                   data.departure_times_.data_);
  set_fws_multimap_offset(offset, h.transfers_, data.transfers_);
  set_fws_multimap_offset(offset, h.reverse_transfers_,
                          data.reverse_transfers_);

  set_fws_multimap_offset(offset, h.in_allowed_, data.in_allowed_);
  set_array_offset(offset, h.out_allowed_data_, data.out_allowed_.data_);

  f.write(&h, sizeof(header));

  write_array(f, data.line_to_first_trip_);
  write_array(f, data.line_to_last_trip_);
  write_array(f, data.trip_to_line_);
  write_array(f, data.line_stop_count_);

  write_fws_multimap(f, data.footpaths_);
  write_fws_multimap(f, data.reverse_footpaths_);
  write_fws_multimap(f, data.lines_at_stop_);
  write_fws_multimap(f, data.stops_on_line_);

  write_fws_multimap(f, data.arrival_times_);
  write_array(f, data.departure_times_.data_);
  write_fws_multimap(f, data.transfers_);
  write_fws_multimap(f, data.reverse_transfers_);

  write_fws_multimap(f, data.in_allowed_);
  write_array(f, data.out_allowed_.data_);
}

template <typename T>
void read_array(file& f, array_offset const& off, mcd::vector<T>& data) {
  assert(off.length_ % sizeof(T) == 0);
  data.resize(off.length_ / sizeof(T));
  f.read(data.data(), off.start_, off.length_);
}

template <typename T, typename Index>
void read_fws_multimap(file& f, fws_multimap_offset const& off,
                       fws_multimap<T, Index>& map) {
  read_array(f, off.index_, map.index_);
  read_array(f, off.data_, map.data_);
}

template <typename T, typename Index>
void read_fws_multimap(file& f, fws_multimap_offset const& off,
                       nested_fws_multimap<T, Index>& map) {
  read_array(f, off.index_, map.index_);
  read_array(f, off.data_, map.data_);
}

bool data_okay_for_schedule(header const& h, schedule const& sched) {
  if (h.version_ != CURRENT_VERSION) {
    LOG(info) << "trip-based data file is old version (" << h.version_
              << "), expected " << CURRENT_VERSION;
    return false;
  }

  auto const& schedule_name = serialize_schedule_names(sched);
  if (schedule_name != h.schedule_name_) {
    LOG(info) << "trip-based data file contains data for different schedule: "
              << h.schedule_name_;
    return false;
  }

  if (sched.schedule_begin_ != static_cast<std::time_t>(h.schedule_begin_) ||
      sched.schedule_end_ != static_cast<std::time_t>(h.schedule_end_)) {
    LOG(info) << "trip-based data file contains different schedule range: "
                 "schedule=["
              << format_unix_time(sched.schedule_begin_) << " ("
              << sched.schedule_begin_ << ") - "
              << format_unix_time(sched.schedule_end_) << " ("
              << sched.schedule_end_ << ")], serialized=["
              << format_unix_time(h.schedule_begin_) << " ("
              << h.schedule_begin_ << "), " << format_unix_time(h.schedule_end_)
              << " (" << h.schedule_end_ << ")]";
    return false;
  }

  if (sched.expanded_trips_.data_size() != h.trip_count_) {
    LOG(info)
        << "trip-based data file contains different number of trips: schedule="
        << sched.expanded_trips_.data_size()
        << ", serialized=" << h.trip_count_;
    return false;
  }

  return true;
}

bool data_okay_for_schedule(std::string const& filename,
                            schedule const& sched) {
  if (!fs::exists(filename)) {
    LOG(info) << "trip-based data file not found";
    return false;
  }
  file f(filename.c_str(), "rb");
  if (f.size() < sizeof(header)) {
    LOG(info) << "trip-based data file does not contain header";
    return false;
  }
  header h{};
  f.read(&h, 0, sizeof(header));
  return data_okay_for_schedule(h, sched);
}

std::unique_ptr<tb_data> read_data(std::string const& filename,
                                   schedule const& sched) {
  utl::verify(fs::exists(filename), "read_data: does not exist: {}", filename);

  file f(filename.c_str(), "rb");
  utl::verify(f.size() >= sizeof(header),
              "trip-based data file does not contain header");

  header h{};
  f.read(&h, 0, sizeof(header));
  utl::verify(data_okay_for_schedule(h, sched), "trip-based data file ist");

  auto data = std::make_unique<tb_data>();

  data->trip_count_ = h.trip_count_;
  data->line_count_ = h.line_count_;

  read_array(f, h.line_to_first_trip_, data->line_to_first_trip_);
  read_array(f, h.line_to_last_trip_, data->line_to_last_trip_);
  read_array(f, h.trip_to_line_, data->trip_to_line_);
  read_array(f, h.line_stop_count_, data->line_stop_count_);

  read_fws_multimap(f, h.footpaths_, data->footpaths_);
  read_fws_multimap(f, h.reverse_footpaths_, data->reverse_footpaths_);
  read_fws_multimap(f, h.lines_at_stop_, data->lines_at_stop_);
  read_fws_multimap(f, h.stops_on_line_, data->stops_on_line_);

  read_fws_multimap(f, h.arrival_times_, data->arrival_times_);
  read_array(f, h.departure_times_data_, data->departure_times_.data_);
  read_fws_multimap(f, h.transfers_, data->transfers_);
  read_fws_multimap(f, h.reverse_transfers_, data->reverse_transfers_);

  read_fws_multimap(f, h.in_allowed_, data->in_allowed_);
  read_array(f, h.out_allowed_data_, data->out_allowed_.data_);

  return data;
}

}  // namespace motis::tripbased::serialization
