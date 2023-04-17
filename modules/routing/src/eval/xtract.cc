#include "motis/routing/eval/commands.h"

#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <string>

#include "cista/mmap.h"
#include "cista/reflection/comparable.h"

#include "utl/enumerate.h"
#include "utl/for_each_line_in_file.h"
#include "utl/parser/arg_parser.h"
#include "utl/verify.h"

#include "motis/module/message.h"
#include "motis/loader/gtfs/files.h"
#include "motis/loader/gtfs/gtfs_parser.h"
#include "motis/loader/hrd/hrd_parser.h"
#include "motis/loader/hrd/parse_config.h"

namespace fs = std::filesystem;
using namespace motis;
using namespace motis::loader::hrd;
using namespace motis::loader::gtfs;
using namespace motis::module;
using motis::routing::RoutingResponse;

namespace motis::routing::eval {

std::tuple<std::string, int, int> get_service_source(std::string const& d) {
  auto const file_end = d.find_first_of(':');
  auto const line_end = d.find_last_of(':');
  utl::verify(file_end != std::string::npos && file_end != line_end,
              "malformed service_source info");
  auto file = d.substr(0, file_end);
  if (auto const slash = file.find_last_of('/'); slash != std::string::npos) {
    file = file.substr(slash + 1);
  }
  return {file, std::stoi(d.substr(file_end + 1, line_end - file_end)),
          std::stoi(d.substr(line_end + 1))};
}

std::map<std::string /* filename / trip_id */,
         std::set<std::pair<int, int>> /* first/last line */>
get_services(std::vector<fs::path> const& response_files) {
  std::map<std::string, std::set<std::pair<int, int>>> services;
  for (auto const& [res_idx, path] : utl::enumerate(response_files)) {
    if (!std::filesystem::is_regular_file(path)) {
      std::cout << "not a regular file: " << path << "\n";
      continue;
    }

    int i = res_idx;
    std::cout << "reading " << path << "\n";
    utl::for_each_line_in_file(path, [&](std::string const& response_str) {
      auto const msg = make_msg(response_str);
      auto const routing_res = motis_content(RoutingResponse, msg);
      auto service_sources = std::vector<std::string>{};

      std::cout << "number of connections "
                << routing_res->connections()->size() << "\n";

      auto c_idx = 0U;
      for (auto const& c : *routing_res->connections()) {
        for (auto const& trip : *c->trips()) {
          auto const source = trip->debug()->str();
          if (source.empty()) {
            std::cout << "Error: Response " << i << " connection " << c_idx
                      << ": trip without service source info: train_nr="
                      << trip->id()->train_nr() << "\n";
            continue;
          }

          auto const [filename, line_from, line_to] =
              get_service_source(source);

          services[filename].emplace(line_from, line_to);
          std::cout << "Service \"" << filename << "\": " << line_from << " - "
                    << line_to << "\n";
        }

        ++c_idx;
      }
    });
  }
  return services;
}

void xtract_hrd(
    fs::path const& schedule_path, fs::path const& new_schedule_path,
    std::map<std::string, std::set<std::pair<int, int>>> const& services) {
  fs::remove_all(new_schedule_path);
  fs::create_directories(new_schedule_path / SCHEDULE_DATA);
  fs::create_symlink(absolute(schedule_path) / CORE_DATA,
                     new_schedule_path / CORE_DATA);

  std::cout << "writing services.txt\n";
  std::ofstream services_file{
      (new_schedule_path / SCHEDULE_DATA / "services.txt").c_str()};
  for (auto const& [filename, line_ranges] : services) {
    if (!fs::is_regular_file(schedule_path / SCHEDULE_DATA / filename)) {
      std::cout << "Error: Schedule file " << filename << " not found\n";
      continue;
    }

    cista::mmap f{
        (schedule_path / SCHEDULE_DATA / filename).generic_string().c_str(),
        cista::mmap::protection::READ};
    auto const file_content = utl::cstr{f.data(), f.size()};

    auto line_number = 0U;
    auto line_range_it = begin(line_ranges);
    for (auto const& line : utl::lines(file_content)) {
      ++line_number;

      if (line_number > line_range_it->second) {
        ++line_range_it;
        if (line_range_it == end(line_ranges)) {
          break;
        }
      }
      if (line_number < line_range_it->first) {
        continue;
      }

      services_file << line.view() << "\n";
    }
  }
}

void xtract_gtfs(
    fs::path const& schedule_path, fs::path const& new_schedule_path,
    std::map<std::string, std::set<std::pair<int, int>>> const& services) {
  fs::remove_all(new_schedule_path);
  fs::create_directories(new_schedule_path);

  fs::create_symlink(absolute(schedule_path) / AGENCY_FILE,
                     new_schedule_path / AGENCY_FILE);
  fs::create_symlink(absolute(schedule_path) / STOPS_FILE,
                     new_schedule_path / STOPS_FILE);
  fs::create_symlink(absolute(schedule_path) / CALENDAR_FILE,
                     new_schedule_path / CALENDAR_FILE);
  fs::create_symlink(absolute(schedule_path) / CALENDAR_DATES_FILE,
                     new_schedule_path / CALENDAR_DATES_FILE);
  fs::create_symlink(absolute(schedule_path) / TRANSFERS_FILE,
                     new_schedule_path / TRANSFERS_FILE);
  fs::create_symlink(absolute(schedule_path) / FREQUENCIES_FILE,
                     new_schedule_path / FREQUENCIES_FILE);
  fs::create_symlink(absolute(schedule_path) / ROUTES_FILE,
                     new_schedule_path / ROUTES_FILE);
  fs::create_symlink(absolute(schedule_path) / TRIPS_FILE,
                     new_schedule_path / TRIPS_FILE);

  std::cout << "writing stop_times.txt\n";
  std::ofstream services_file{(new_schedule_path / "stop_times.txt").c_str()};
  std::set<std::pair<int, int>> line_ranges;
  for (auto const& [trip_id, ranges] : services) {
    line_ranges.insert(ranges.begin(), ranges.end());
  }
  cista::mmap f{(schedule_path / STOP_TIMES_FILE).generic_string().c_str(),
                cista::mmap::protection::READ};
  auto line_number = 0U;
  auto line_range_it = begin(line_ranges);
  for (auto const& line : utl::lines(utl::cstr{f.data(), f.size()})) {
    ++line_number;

    if (line_number == 1) {
      services_file << line.view() << "\n";
    }

    if (line_number > line_range_it->second) {
      ++line_range_it;
      if (line_range_it == end(line_ranges)) {
        break;
      }
    }

    if (line_number < line_range_it->first) {
      continue;
    }

    services_file << line.view() << "\n";
  }
}

int xtract(int argc, char const** argv) {
  if (argc < 4) {
    std::cout << "usage: " << argv[0]
              << " src-schedule target-schedule response [response, ...]";
    return 0;
  }

  auto const schedule_path = fs::path{argv[1]};
  auto const new_schedule_path = fs::path{argv[2]};
  auto const response_files = [&]() {
    std::vector<fs::path> r;
    for (auto i = 3U; i != argc; ++i) {
      r.emplace_back(argv[i]);
    }
    return r;
  }();

  auto const services = get_services(response_files);
  auto const number_of_trips = std::accumulate(
      begin(services), end(services), 0U, [](unsigned const sum, auto&& entry) {
        return sum + entry.second.size();
      });
  std::cout << "number of trips: " << number_of_trips << "\n";

  if (hrd_parser{}.applicable(schedule_path)) {
    xtract_hrd(schedule_path, new_schedule_path, services);
  } else if (gtfs_parser{}.applicable(schedule_path)) {
    xtract_gtfs(schedule_path, new_schedule_path, services);
  } else {
    std::cout << "no timetable format detected: " << schedule_path << "\n";
  }

  return 0;
}

}  // namespace motis::routing::eval
