#include "boost/program_options.hpp"

#include <filesystem>
#include <iostream>

#include "google/protobuf/stubs/common.h"

#include "utl/progress_tracker.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/import.h"
#include "motis/server.h"

#if !defined(MOTIS_VERSION)
#define MOTIS_VERSION "unknown"
#endif

namespace po = boost::program_options;
using namespace std::string_view_literals;
namespace fs = std::filesystem;

using namespace motis;

int main(int ac, char** av) {
  auto data_path = fs::path{"data"};
  auto config_path = fs::path{"config.yml"};

  auto desc = po::options_description{"Global options"};
  desc.add_options()  //
      ("version", "Prints the MOTIS version")  //
      ("help", "Prints this help message")  //
      ("data,d", po::value(&data_path)->default_value(data_path),
       "The data path contains all preprocessed data as well as a `config.yml` "
       "and is required by `motis server`. It will be created by the `motis "
       "import` command. After the import has finished, `motis server` only "
       "needs the `data` folder and can run without the input files (such as "
       "OpenStreetMap file, GTFS datasets, tiles-profiles, etc.)")  //
      ("config,c", po::value(&config_path)->default_value(config_path),
       "Configuration YAML file. Legacy INI files are still supported but this "
       "support will be dropped in the future.")  //
      ("command", po::value<std::string>(),
       "Command to execute:\n"
       "  - \"import\": preprocesses the input data\n"
       "    and creates the `data` folder.\n"
       "  - \"server\": serves static files\n"
       "    and all API endpoints such as\n"
       "    routing, geocoding, tiles, etc.")  //
      ("paths", po::value<std::vector<std::string>>(),
       "List of paths to import for the simple mode. File type will be "
       "determined based on extension:\n"
       "  - \".osm.pbf\" will be used as\n"
       "    OpenStreetMap file.\n"
       "    This enables street routing,\n"
       "    geocoding and map tiles\n"
       "  - the rest will be interpreted as\n"
       "    static timetables.\n"
       "    This enables transit routing");

  auto const help = [&]() {
    std::cout << "MOTIS " << MOTIS_VERSION << "\n\n"
              << "Usage:\n"
                 "  - simple:   motis         [PATHS...]\n"
                 "  - import:   motis import  [-c config.yml] [-d data_dir]\n"
                 "  - server:   motis server  [-d data_dir]\n\n"
              << desc << "\n";
  };

  enum mode { kImport, kServer, kSimple } mode = kSimple;
  if (ac > 1) {
    auto const cmd = std::string_view{av[1]};
    switch (cista::hash(cmd)) {
      case cista::hash("import"):
        mode = kImport;
        --ac;
        ++av;
        break;
      case cista::hash("server"):
        mode = kServer;
        --ac;
        ++av;
        break;
    }
  } else {
    help();
    return 1;
  }

  auto pos = po::positional_options_description{}.add("paths", -1);
  auto vm = po::variables_map{};
  po::store(po::command_line_parser(ac, av).options(desc).positional(pos).run(),
            vm);
  po::notify(vm);

  if (vm.count("version")) {
    std::cout << MOTIS_VERSION << "\n";
    return 0;
  } else if (vm.count("help")) {
    help();
    return 0;
  }

  switch (mode) {
    case kServer:
      try {
        auto const c = config::read(data_path / "config.yml");
        return server(data{data_path, c}, c);
      } catch (std::exception const& e) {
        std::cerr << "unable to start server: " << e.what() << "\n";
        return 1;
      }

    case kImport: {
      auto c = config{};
      try {
        c = config_path.extension() == ".ini" ? config::read_legacy(config_path)
                                              : config::read(config_path);
        auto const bars = utl::global_progress_bars{false};
        import(c, std::move(data_path));
        return 0;
      } catch (std::exception const& e) {
        fmt::println("unable to import: {}", e.what());
        fmt::println("config:\n{}", fmt::streamed(c));
        return 1;
      }
    }

    case kSimple:
      try {
        auto const bars = utl::global_progress_bars{false};
        auto args = vm.count("paths")
                        ? vm.at("paths").as<std::vector<std::string>>()
                        : std::vector<std::string>{};

        auto const c = config::read_simple(args);
        server(import(c, data_path), c);
      } catch (std::exception const& e) {
        std::cerr << "error: " << e.what() << "\n";
      }
      return 0;
  }

  google::protobuf::ShutdownProtobufLibrary();
}