#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "boost/program_options.hpp"

namespace motis {

inline void add_help_opt(boost::program_options::options_description& desc) {
  desc.add_options()("help,h", "print this help message");
}

inline void add_data_path_opt(boost::program_options::options_description& desc,
                              std::filesystem::path& p) {
  desc.add_options()(
      "data,d", boost::program_options::value(&p)->default_value(p),
      "The data path contains all preprocessed data as well as a `config.yml`. "
      "It will be created by the `motis import` command. After the import has "
      "finished, `motis server` only needs the `data` folder and can run "
      "without the input files (such as OpenStreetMap file, GTFS datasets, "
      "tiles-profiles, etc.)");
}

inline void add_config_path_opt(
    boost::program_options::options_description& desc,
    std::filesystem::path& p) {
  desc.add_options()(
      "config,c", boost::program_options::value(&p)->default_value(p),
      "Configuration YAML file. Legacy INI files are still supported but this "
      "support will be dropped in the future.");
}

inline void add_trip_id_opt(boost::program_options::options_description& desc) {
  desc.add_options()(
      "trip-id,t",
      boost::program_options::value<std::vector<std::string> >()->composing(),
      "Add trip-id to analyze.\n"
      "If the trip-id is encoded, it will be decoded automatically.\n"
      "This option can be used multiple times.\n"
      "\n"
      "Will search the shape corresponding to each trip-id. "
      "If a shape is found, the index of the shape point, that is "
      "matched with each stop, will be printed.\n"
      "Notice that the first and last stop of a trip will always be "
      "matched with the first and last shape point respectively.\n"
      "If a shape contains less points than stops in the trip, this "
      "segmentation is not possible.");
}

inline void add_log_level_opt(boost::program_options::options_description& desc,
                              std::string& log_lvl) {
  desc.add_options()(
      "log-level",
      boost::program_options::value(&log_lvl)->default_value(log_lvl),
      "Set the log level.\n"
      "Supported log levels: ERROR, INFO, DEBUG");
}

inline boost::program_options::variables_map parse_opt(
    int ac, char** av, boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  auto vm = po::variables_map{};
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);
  return vm;
}

}  // namespace motis
