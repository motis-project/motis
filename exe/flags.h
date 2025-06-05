#pragma once

#include <filesystem>

#include "boost/program_options.hpp"

namespace motis {

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

inline boost::program_options::variables_map parse_opt(
    int ac, char** av, boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  auto vm = po::variables_map{};
  po::store(po::command_line_parser(ac, av).options(desc).run(), vm);
  po::notify(vm);
  return vm;
}

}  // namespace motis