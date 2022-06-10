#include "motis/loader/loader_options.h"

#include <sstream>

#include "boost/date_time/local_time/local_time.hpp"
#include "boost/filesystem.hpp"

#include "motis/core/common/date_time_util.h"

namespace fs = boost::filesystem;

namespace motis::loader {

std::pair<std::time_t, std::time_t> loader_options::interval() const {
  std::pair<std::time_t, std::time_t> interval;

  if (schedule_begin_ == "TODAY") {
    auto now = boost::posix_time::second_clock::universal_time().date();
    interval.first = to_unix_time(now.year(), now.month(), now.day());
  } else {
    interval.first = to_unix_time(std::stoi(schedule_begin_.substr(0, 4)),
                                  std::stoi(schedule_begin_.substr(4, 2)),
                                  std::stoi(schedule_begin_.substr(6, 2)));
  }

  interval.second = interval.first + num_days_ * 24 * 3600;

  return interval;
}

std::string loader_options::graph_path(std::string const& data_dir) const {
  if (graph_path_ == "default") {
    auto const [from, to] = interval();
    std::stringstream ss;
    ss << "graph_" << from << "-" << to << "af" << adjust_footpaths_ << "ar"
       << apply_rules_ << "ef" << expand_footpaths_ << "ptd"
       << planned_transfer_delta_ << "nlt" << no_local_transport_ << ".raw";
    return (fs::path{data_dir} / "schedule" / ss.str()).generic_string();
  } else {
    return graph_path_;
  }
}

std::string loader_options::fbs_schedule_path(std::string const& data_dir,
                                              size_t const id) const {
  if (dataset_prefix_.empty()) {
    return (fs::path{data_dir} / "schedule" / "schedule.raw").generic_string();
  } else {
    return (fs::path{data_dir} / "schedule" /
            (dataset_prefix_.at(id) + "_schedule.raw"))
        .generic_string();
  }
}

}  // namespace motis::loader
