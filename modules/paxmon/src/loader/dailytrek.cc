#include "motis/paxmon/loader/dailytrek.h"

#include <chrono>
#include <map>
#include <regex>

#include "boost/filesystem.hpp"

#include "date/date.h"

#include "utl/to_vec.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/access/time_access.h"

namespace fs = boost::filesystem;

using namespace motis::logging;

namespace motis::paxmon::loader {

std::vector<std::string> get_dailytrek_files(schedule const& sched,
                                             std::string const& dir) {
  struct file_info {
    fs::path path_{};
    unixtime modified_{};
  };
  auto newest_file_per_day = std::map<date::year_month_day, file_info>{};

  auto const csv_ext = fs::path{".csv"};
  auto const fn_re =
      std::regex{R"(^DailyTReK_.*?((\d{4})-(\d{2})-(\d{2})).*$)"};

  auto const first_day =
      date::floor<date::days>(std::chrono::system_clock::time_point{
          std::chrono::seconds{external_schedule_begin(sched)}}) -
      date::days{1};
  auto const last_day =
      date::floor<date::days>(std::chrono::system_clock::time_point{
          std::chrono::seconds{external_schedule_end(sched)}}) -
      date::days{1};

  LOG(info) << "searching for DailyTReK files in " << dir;
  for (auto const& entry : fs::directory_iterator{dir}) {
    auto const& p = entry.path();
    if (!fs::is_regular_file(entry) || p.extension() != csv_ext) {
      continue;
    }
    auto const fn = p.stem().string();
    auto m = std::smatch{};
    if (std::regex_match(fn, m, fn_re)) {
      auto const file_day = date::year_month_day{
          date::year{std::stoi(m[2])},
          date::month{static_cast<unsigned>(std::stoi(m[3]))},
          date::day{static_cast<unsigned>(std::stoi(m[4]))}};
      if (file_day < first_day || file_day > last_day) {
        continue;
      }
      auto const file_time = static_cast<unixtime>(fs::last_write_time(p));
      auto const fi = file_info{p, file_time};
      if (auto it = newest_file_per_day.insert({file_day, fi}); !it.second) {
        if (it.first->second.modified_ < file_time) {
          it.first->second = fi;
        }
      }
    }
  }

  for (auto const& [day, fi] : newest_file_per_day) {
    LOG(info) << "found DailyTReK file for " << day << ": " << fi.path_
              << " (modified " << format_unix_time(fi.modified_) << ")";
  }

  return utl::to_vec(newest_file_per_day, [](auto const& entry) {
    return entry.second.path_.string();
  });
}

}  // namespace motis::paxmon::loader
