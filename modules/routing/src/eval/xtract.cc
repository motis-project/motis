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
#include "utl/helpers/algorithm.h"
#include "utl/parser/arg_parser.h"
#include "utl/pipes.h"
#include "utl/verify.h"

#include "conf/options_parser.h"

#include "motis/module/message.h"
#include "motis/bootstrap/import_files.h"
#include "motis/bootstrap/import_settings.h"
#include "motis/loader/gtfs/files.h"
#include "motis/loader/gtfs/gtfs_parser.h"
#include "motis/loader/hrd/hrd_parser.h"
#include "motis/loader/hrd/parse_config.h"

namespace fs = std::filesystem;
namespace mb = motis::bootstrap;
using namespace motis;
using namespace motis::loader::hrd;
using namespace motis::loader::gtfs;
using namespace motis::module;
using motis::routing::RoutingResponse;

namespace motis::routing::eval {

enum class schedule_format { kGTFS, kHRD };

std::tuple<fs::path, int, int> get_service_source(std::string const& d) {
  auto const file_end = d.find_first_of(':');
  auto const line_end = d.find_last_of(':');
  utl::verify(file_end != std::string::npos && file_end != line_end,
              "malformed service_source info");
  auto file = d.substr(0, file_end);
  return {file, std::stoi(d.substr(file_end + 1, line_end - file_end)),
          std::stoi(d.substr(line_end + 1))};
}

std::map<fs::path, std::set<std::pair<int, int>> /* first/last line */>
get_services(std::vector<fs::path> const& response_files) {
  std::map<fs::path, std::set<std::pair<int, int>>> services;
  for (auto const& [res_idx, res_file_path] : utl::enumerate(response_files)) {
    if (!std::filesystem::is_regular_file(res_file_path)) {
      std::cout << "not a regular file: " << res_file_path << "\n";
      continue;
    }

    int i = res_idx;
    std::cout << "reading " << res_file_path << "\n";
    utl::for_each_line_in_file(
        res_file_path.generic_string(), [&](auto const& response_str) {
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

              auto const [path, line_from, line_to] =
                  get_service_source(source);
              utl::verify(exists(path), "path {} does not exist", path);

              services[path].emplace(line_from, line_to);
              std::cout << "Service \"" << path << "\": " << line_from << " - "
                        << line_to << "\n";
            }

            ++c_idx;
          }
        });
  }
  return services;
}

void create_base_copy(schedule_format const format, fs::path const& src,
                      fs::path const& dest) {
  auto const link_if_exists = [&](fs::path const& from, fs::path const& to) {
    if (exists(from)) {
      create_symlink(from, to);
    }
  };

  auto const abs_src = absolute(src);
  auto ec = std::error_code{};
  switch (format) {
    case schedule_format::kGTFS:
      fs::remove_all(dest, ec);
      fs::create_directories(dest);
      link_if_exists(abs_src / AGENCY_FILE, dest / AGENCY_FILE);
      link_if_exists(abs_src / STOPS_FILE, dest / STOPS_FILE);
      link_if_exists(abs_src / CALENDAR_FILE, dest / CALENDAR_FILE);
      link_if_exists(abs_src / CALENDAR_DATES_FILE, dest / CALENDAR_DATES_FILE);
      link_if_exists(abs_src / TRANSFERS_FILE, dest / TRANSFERS_FILE);
      link_if_exists(abs_src / FREQUENCIES_FILE, dest / FREQUENCIES_FILE);
      link_if_exists(abs_src / ROUTES_FILE, dest / ROUTES_FILE);
      link_if_exists(abs_src / TRIPS_FILE, dest / TRIPS_FILE);
      std::ofstream{dest / STOP_TIMES_FILE}.write("", 0);
      break;

    case schedule_format::kHRD:
      fs::remove_all(dest, ec);
      fs::create_directories(dest / SCHEDULE_DATA);
      fs::create_symlink(abs_src / CORE_DATA, dest / CORE_DATA);
      std::ofstream{dest / SCHEDULE_DATA / "services.txt"}.write("", 0);
      break;
  }
}

void xtract_services(fs::path const& old_path, fs::path const& new_path,
                     std::set<std::pair<int, int>> const& line_ranges) {
  std::cout << "writing " << old_path << " -> " << new_path << "\n";
  std::ofstream services_file{new_path.c_str()};
  cista::mmap f{old_path.generic_string().c_str(),
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

struct xtract_settings : public conf::configuration {
  xtract_settings() : conf::configuration{"xtract"} {
    param(response_files_, "responses", "list of response files");
    param(new_schedule_path_, "new_schedule", "new schedule path");
  }
  std::vector<fs::path> response_files_;
  fs::path new_schedule_path_;
};

int xtract(int argc, char const** argv) {
  if (argc < 4) {
    std::cout << "usage: " << argv[0]
              << " src-schedule target-schedule response [response, ...]";
    return 0;
  }

  mb::import_settings import_opt;
  xtract_settings xtract_opt;
  auto const confs =
      std::vector<conf::configuration*>{&import_opt, &xtract_opt};
  conf::options_parser parser(confs);
  try {
    parser.read_environment("MOTIS_");
    parser.read_command_line_args(argc, argv, true);

    if (parser.help()) {
      parser.print_help(std::cout);
      return 0;
    } else if (parser.version()) {
      return 0;
    }

    parser.read_configuration_file(true);

    parser.print_used(std::cout);
  } catch (std::exception const& e) {
    std::cout << "options error: " << e.what() << "\n";
    return 1;
  }

  auto const dest_path = fs::path{xtract_opt.new_schedule_path_};
  auto const schedule_paths =
      utl::all(import_opt.import_paths_)  //
      | utl::transform([](auto&& s) { return mb::split_import_path(s); })  //
      |
      utl::remove_if([](auto&& s) { return std::get<0>(s) != "schedule"; })  //
      | utl::transform([](auto&& s) -> fs::path { return std::get<2>(s); })  //
      | utl::transform([&](fs::path const& s)
                           -> std::tuple<schedule_format, fs::path, fs::path> {
          if (hrd_parser{}.applicable(s)) {
            return {schedule_format::kHRD, s, dest_path / s.filename()};
          } else if (gtfs_parser{}.applicable(s)) {
            return {schedule_format::kGTFS, s, dest_path / s.filename()};
          } else {
            throw utl::fail("no timetable format detected: {}", s);
          }
        })  //
      | utl::vec();

  for (auto const& [format, path, new_schedule_path] : schedule_paths) {
    create_base_copy(format, path, new_schedule_path);
  }

  auto const services = get_services(xtract_opt.response_files_);
  auto const number_of_trips = std::accumulate(
      begin(services), end(services), 0U, [](unsigned const sum, auto&& entry) {
        return sum + entry.second.size();
      });
  std::cout << "number of trips: " << number_of_trips << "\n";

  auto const is_subpath = [](fs::path const& path, fs::path const& base) {
    auto const rel = std::filesystem::relative(path, base);
    return !rel.empty() && rel.native()[0] != '.';
  };
  for (auto const& [path, lines] : services) {
    auto const p = path;
    auto const sched_it = utl::find_if(schedule_paths, [&](auto&& sched_p) {
      return is_subpath(p, std::get<1>(sched_p));
    });
    if (sched_it == end(schedule_paths)) {
      std::cout << "no schedule found for " << path << "\n";
      return 1;
    }

    auto const [format, old_path, new_path] = *sched_it;
    auto const rel = std::filesystem::relative(path, old_path);
    xtract_services(path, new_path / rel, lines);
  }

  create_symlink(absolute(fs::path{parser.file()}),
                 xtract_opt.new_schedule_path_ / ".." / "config.ini");

  return 0;
}

}  // namespace motis::routing::eval
