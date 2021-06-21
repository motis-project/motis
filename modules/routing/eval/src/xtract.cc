#include <iostream>
#include <map>
#include <set>
#include <string>

#include "boost/filesystem.hpp"

#include "cista/mmap.h"
#include "cista/reflection/comparable.h"

#include "utl/enumerate.h"
#include "utl/for_each_line_in_file.h"
#include "utl/parser/arg_parser.h"
#include "utl/verify.h"

#include "motis/core/schedule/footpath.h"
#include "motis/module/message.h"
#include "motis/loader/hrd/hrd_parser.h"
#include "motis/loader/hrd/parse_config.h"

namespace fs = boost::filesystem;
using namespace motis;
using namespace motis::loader::hrd;
using namespace motis::module;
using motis::routing::RoutingResponse;

std::tuple<std::string, int, int> get_service_source(std::string const& d) {
  auto const file_end = d.find_first_of(':');
  auto const line_end = d.find_last_of(':');
  utl::verify(file_end != std::string::npos && file_end != line_end,
              "malformed service_source info");
  return {d.substr(0, file_end),
          std::stoi(d.substr(file_end + 1, line_end - file_end)),
          std::stoi(d.substr(line_end + 1))};
}

struct hrd_footpath {
  CISTA_COMPARABLE()
  std::string from_id_, to_id_;
  int duration_;
};

auto const parser_config = hrd_5_20_26;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "usage: " << argv[0]
              << " src-schedule target-schedule response [response, ...]\n";
    return 0;
  }

  auto const schedule_path = fs::path{argv[1]};
  auto const new_schedule_path = fs::path{argv[2]};
  auto const response_files = [&]() {
    std::vector<std::string> r;
    for (auto i = 3U; i != argc; ++i) {
      r.emplace_back(argv[i]);
    }
    return r;
  }();

  hrd_parser parser;
  if (!parser.applicable(schedule_path)) {
    std::cout << "No schedule in " << schedule_path << " found\n";
    std::cout << "Missing files:\n";
    for (auto const& missing_file : parser.missing_files(schedule_path)) {
      std::cout << "  " << missing_file << "\n";
    }
    return 1;
  }

  std::set<hrd_footpath> footpaths;
  std::map<std::string /* filename */,
           std::set<std::pair<int, int>> /* first/last line */>
      services;
  std::set<std::string> meta_candidates;
  for (auto const& [res_idx, path] : utl::enumerate(response_files)) {
    int i = res_idx;
    utl::for_each_line_in_file(path, [&](std::string const& response_str) {
      auto const msg = make_msg(response_str);
      auto const routing_res = motis_content(RoutingResponse, msg);
      auto service_sources = std::vector<std::string>{};

      auto c_idx = 0U;
      for (auto const& c : *routing_res->connections()) {
        meta_candidates.emplace(c->stops()->Get(0)->station()->id()->str());
        meta_candidates.emplace(
            c->stops()->Get(c->stops()->size() - 1)->station()->id()->str());

        for (auto const& fp : *c->transports()) {
          if (fp->move_type() != Move_Walk) {
            continue;
          }

          auto const walk = static_cast<Walk const*>(fp->move());
          auto const from = c->stops()->Get(walk->range()->from());
          auto const to = c->stops()->Get(walk->range()->to());
          footpaths.emplace(hrd_footpath{
              from->station()->id()->str(), to->station()->id()->str(),
              static_cast<int>(
                  (to->arrival()->time() - from->departure()->time()) / 60)});
        }

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
          if (!fs::is_regular_file(schedule_path / SCHEDULE_DATA / filename)) {
            std::cout << "Error: Schedule file " << filename << " not found\n";
            continue;
          }

          services[filename].emplace(line_from, line_to);
          std::cout << "Service " << filename << ": " << line_from << " - "
                    << line_to << "\n";
        }
        ++c_idx;
      }
    });
  }

  std::set<std::string> station_ids, categories, bitfields, providers;
  {
    fs::create_directories(new_schedule_path / SCHEDULE_DATA);
    std::cout << "writing services.txt\n";
    std::ofstream services_file{
        (new_schedule_path / SCHEDULE_DATA / "services.txt").c_str()};
    for (auto const& [filename, line_ranges] : services) {
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

        if (line.starts_with("*A VE")) {
          bitfields.emplace(
              line.substr(parser_config.s_info_.traff_days_).to_str());
        } else if (line.starts_with("*A") || line.starts_with("*R")) {
          continue;
        } else if (line.starts_with("*Z")) {
          providers.emplace(line.substr(9, utl::size(6)).to_str());
        } else if (line.starts_with("*G ")) {
          categories.emplace(line.substr(parser_config.s_info_.cat_).to_str());
        } else if (std::isdigit(line[0]) != 0) {
          station_ids.emplace(line.substr(0, utl::size(7)).to_str());
        }

        services_file << line.view() << "\n";
      }
    }
  }

  {
    cista::mmap metabhf_file{
        (schedule_path / CORE_DATA / parser_config.files(FOOTPATHS))
            .generic_string()
            .c_str(),
        cista::mmap::protection::READ};
    auto const metabhf_file_content =
        utl::cstr{metabhf_file.data(), metabhf_file.size()};
    fs::create_directories(new_schedule_path / CORE_DATA);
    auto meta_bhf_out = std::ofstream{
        (new_schedule_path / CORE_DATA / parser_config.files(FOOTPATHS))
            .c_str()};
    for (auto const& line : utl::lines(metabhf_file_content)) {
      if (line[7] == ':' &&
          std::any_of(begin(meta_candidates), end(meta_candidates),
                      [&](std::string const& s) { return line.contains(s); })) {
        meta_bhf_out << line.view() << "\n";

        station_ids.emplace(
            line.substr(parser_config.meta_.meta_stations_.eva_).to_str());
        utl::for_each_token(line.substr(8), ' ',
                            [&](utl::cstr meta_station_id) {
                              if (meta_station_id.starts_with("F")) {
                                return;
                              }
                              station_ids.emplace(meta_station_id.to_str());
                            });
      }
    }
    for (auto const& fp : footpaths) {
      meta_bhf_out << fp.from_id_ << " " << fp.to_id_ << " " << std::setw(3)
                   << std::setfill('0') << fp.duration_ << "\n";
    }
  }

  {
    cista::mmap stations_file{
        (schedule_path / CORE_DATA / parser_config.files(STATIONS))
            .generic_string()
            .c_str(),
        cista::mmap::protection::READ};
    auto const stations_file_content =
        utl::cstr{stations_file.data(), stations_file.size()};
    auto stations_out = std::ofstream{
        (new_schedule_path / CORE_DATA / parser_config.files(STATIONS))
            .c_str()};
    for (auto const& line : utl::lines(stations_file_content)) {
      if (station_ids.find(
              line.substr(parser_config.st_.names_.eva_).to_str()) !=
          end(station_ids)) {
        stations_out << line.view() << "\n";
      }
    }
  }

  {
    cista::mmap station_coords_file{
        (schedule_path / CORE_DATA / parser_config.files(COORDINATES))
            .generic_string()
            .c_str(),
        cista::mmap::protection::READ};
    auto const station_coords_file_content =
        utl::cstr{station_coords_file.data(), station_coords_file.size()};
    auto station_coords_out = std::ofstream{
        (new_schedule_path / CORE_DATA / parser_config.files(COORDINATES))
            .c_str()};
    for (auto const& line : utl::lines(station_coords_file_content)) {
      if (station_ids.find(
              line.substr(parser_config.st_.names_.eva_).to_str()) !=
          end(station_ids)) {
        station_coords_out << line.view() << "\n";
      }
    }
  }

  {
    cista::mmap categories_file{
        (schedule_path / CORE_DATA / parser_config.files(CATEGORIES))
            .generic_string()
            .c_str(),
        cista::mmap::protection::READ};
    auto const categories_file_content =
        utl::cstr{categories_file.data(), categories_file.size()};
    auto categories_out = std::ofstream{
        (new_schedule_path / CORE_DATA / parser_config.files(CATEGORIES))
            .c_str()};
    for (auto const& line : utl::lines(categories_file_content)) {
      if (categories.find(line.substr(parser_config.cat_.code_).to_str()) !=
          end(categories)) {
        categories_out << line.view() << "\n";
      }
    }
  }

  {
    cista::mmap bitfields_file{
        (schedule_path / CORE_DATA / parser_config.files(BITFIELDS))
            .generic_string()
            .c_str(),
        cista::mmap::protection::READ};
    auto const bitfields_file_content =
        utl::cstr{bitfields_file.data(), bitfields_file.size()};
    auto bitfields_out = std::ofstream{
        (new_schedule_path / CORE_DATA / parser_config.files(BITFIELDS))
            .c_str()};
    for (auto const& line : utl::lines(bitfields_file_content)) {
      if (bitfields.find(line.substr(parser_config.bf_.index_).to_str()) !=
          end(bitfields)) {
        bitfields_out << line.view() << "\n";
      }
    }
  }

  {
    cista::mmap providers_file{
        (schedule_path / CORE_DATA / parser_config.files(PROVIDERS))
            .generic_string()
            .c_str(),
        cista::mmap::protection::READ};
    auto const providers_file_content =
        utl::cstr{providers_file.data(), providers_file.size()};
    auto providers_out = std::ofstream{
        (new_schedule_path / CORE_DATA / parser_config.files(PROVIDERS))
            .c_str()};
    auto prev_line = utl::cstr{};
    for (auto const& line : utl::lines(providers_file_content)) {
      if (line[6] != 'K' &&
          providers.find(line.substr(utl::field{8, 6}).to_str()) !=
              end(providers)) {
        providers_out << prev_line.view() << "\n";
        providers_out << line.view() << "\n";
      }
      prev_line = line;
    }
  }

  for (auto const& file : {BASIC_DATA, TIMEZONES}) {
    fs::copy_file(schedule_path / CORE_DATA / parser_config.files(file),
                  new_schedule_path / CORE_DATA / parser_config.files(file),
                  fs::copy_option::overwrite_if_exists);
  }

  for (auto const& file : {ATTRIBUTES, TRACKS, INFOTEXT, THROUGH_SERVICES,
                           MERGE_SPLIT_SERVICES, DIRECTIONS, MIN_CT_FILE}) {
    std::ofstream out{
        (new_schedule_path / CORE_DATA / parser_config.files(file)).c_str()};
  }
}