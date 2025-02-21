#include "boost/program_options.hpp"

#include <filesystem>
#include <iostream>

#include "google/protobuf/stubs/common.h"

#include "utl/progress_tracker.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/import.h"
#include "motis/server.h"

#include "./flags.h"

#if !defined(MOTIS_VERSION)
#define MOTIS_VERSION "unknown"
#endif

namespace po = boost::program_options;
using namespace std::string_view_literals;
namespace fs = std::filesystem;

namespace motis {
int generate(int, char**);
int batch(int, char**);
int compare(int, char**);
}  // namespace motis

using namespace motis;

int main(int ac, char** av) {
  if (ac > 1 && av[1] == "--help"sv) {
    fmt::println(
        "MOTIS {}\n\n"
        "Usage:\n"
        "  --help    print this help message\n"
        "  --version print program version\n\n"
        "Commands:\n"
        "  generate   generate random queries and write them to a file\n"
        "  batch      run queries from a file\n"
        "  compare    compare results from different batch runs\n"
        "  config     generate a config file from a list of input files\n"
        "  import     prepare input data, creates the data directory\n"
        "  server     starts a web server serving the API\n",
        MOTIS_VERSION);
    return 0;
  } else if (ac <= 1 || (ac >= 2 && av[1] == "--version"sv)) {
    fmt::println("{}", MOTIS_VERSION);
    return 0;
  }

  // Skip program argument, quit if no command.
  --ac;
  ++av;

  auto return_value = 0;

  // Execute command.
  auto const cmd = std::string_view{av[0]};
  switch (cista::hash(cmd)) {
    case cista::hash("generate"): return_value = generate(ac, av); break;
    case cista::hash("batch"): return_value = batch(ac, av); break;
    case cista::hash("compare"): return_value = compare(ac, av); break;

    case cista::hash("config"): {
      auto paths = std::vector<std::string>{};
      for (auto i = 1; i != ac; ++i) {
        paths.push_back(std::string{av[i]});
      }
      if (paths.empty() || paths.front() == "--help") {
        fmt::println(
            "usage: motis config [PATHS...]\n\n"
            "Generates a config.yml file in the current working "
            "directory.\n\n"
            "File type will be determined based on extension:\n"
            "  - \".osm.pbf\" will be used as OpenStreetMap file.\n"
            "    This enables street routing, geocoding and map tiles\n"
            "  - the rest will be interpreted as static timetables.\n"
            "    This enables transit routing."
            "\n\n"
            "Example: motis config germany-latest.osm.pbf "
            "germany.gtfs.zip\n");
        return_value = paths.empty() ? 1 : 0;
        break;
      }
      std::ofstream{"config.yml"} << config::read_simple(paths) << "\n";
      return_value = 0;
      break;
    }

    case cista::hash("server"):
      try {
        auto data_path = fs::path{"data"};

        auto desc = po::options_description{"Server Options"};
        add_data_path_opt(desc, data_path);
        auto vm = parse_opt(ac, av, desc);
        if (vm.count("help")) {
          std::cout << desc << "\n";
          return_value = 0;
          break;
        }

        auto const c = config::read(data_path / "config.yml");
        return_value = server(data{data_path, c}, c);
        break;
      } catch (std::exception const& e) {
        std::cerr << "unable to start server: " << e.what() << "\n";
        return_value = 1;
        break;
      }

    case cista::hash("import"): {
      auto c = config{};
      try {
        auto data_path = fs::path{"data"};
        auto config_path = fs::path{"config.yml"};

        auto desc = po::options_description{"Import Options"};
        add_data_path_opt(desc, data_path);
        add_config_path_opt(desc, config_path);
        auto vm = parse_opt(ac, av, desc);
        if (vm.count("help")) {
          std::cout << desc << "\n";
          return_value = 0;
          break;
        }

        c = config_path.extension() == ".ini" ? config::read_legacy(config_path)
                                              : config::read(config_path);
        auto const bars = utl::global_progress_bars{false};
        import(c, std::move(data_path));
        return_value = 0;
        break;
      } catch (std::exception const& e) {
        fmt::println("unable to import: {}", e.what());
        fmt::println("config:\n{}", fmt::streamed(c));
        return_value = 1;
        break;
      }
    }

    default:
      fmt::println("Invalid command. Type motis --help for a list of commands.");
      return_value = 1;
      break;
  }

  google::protobuf::ShutdownProtobufLibrary();
  return return_value;
}