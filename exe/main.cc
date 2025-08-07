#include "boost/program_options.hpp"
#include "boost/url/decode_view.hpp"

#include <cctype>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include "google/protobuf/stubs/common.h"

#include "utl/logging.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"

#include "nigiri/logging.h"
#include "nigiri/rt/util.h"

#include "motis/analyze_shapes.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/import.h"
#include "motis/server.h"

#include "./flags.h"

#if defined(USE_MIMALLOC) && defined(_WIN32)
#include "mimalloc-new-delete.h"
#endif

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
int extract(int, char**);
}  // namespace motis

using namespace motis;

int main(int ac, char** av) {
  auto const motis_version = std::string_view{MOTIS_VERSION};
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
        "  server     starts a web server serving the API\n"
        "  extract    trips from a Itinerary to GTFS timetable\n"
        "  pb2json    convert GTFS-RT protobuf to JSON\n"
        "  json2pb    convert JSON to GTFS-RT protobuf\n"
        "  shapes     print shape segmentation for trips\n",
        motis_version);
    return 0;
  } else if (ac <= 1 || (ac >= 2 && av[1] == "--version"sv)) {
    fmt::println("{}", motis_version);
    return 0;
  }

  // Skip program argument, quit if no command.
  --ac;
  ++av;

  auto return_value = 0;

  // Execute command.
  auto const cmd = std::string_view{av[0]};
  switch (cista::hash(cmd)) {
    case cista::hash("extract"): return_value = extract(ac, av); break;
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
        auto log_lvl = std::string{"DEBUG"};

        auto desc = po::options_description{"Server Options"};
        add_data_path_opt(desc, data_path);
        add_log_level_opt(desc, log_lvl);
        add_help_opt(desc);
        auto vm = parse_opt(ac, av, desc);
        if (vm.count("help")) {
          std::cout << desc << "\n";
          return_value = 0;
          break;
        }
        if (vm.count("log-level")) {
          std::transform(log_lvl.begin(), log_lvl.end(), log_lvl.begin(),
                         [](unsigned char const c) { return std::toupper(c); });
          if (log_lvl == "ERROR"sv) {
            utl::log_verbosity = utl::log_level::error;
            nigiri::s_verbosity = nigiri::log_lvl::error;
          } else if (log_lvl == "INFO"sv) {
            utl::log_verbosity = utl::log_level::info;
            nigiri::s_verbosity = nigiri::log_lvl::info;
          } else if (log_lvl == "DEBUG"sv) {
            utl::log_verbosity = utl::log_level::debug;
            nigiri::s_verbosity = nigiri::log_lvl::debug;
          } else {
            fmt::println(std::cerr, "Unsupported log level '{}'\n", log_lvl);
            return_value = 1;
            break;
          }
        }

        auto const c = config::read(data_path / "config.yml");
        return_value = server(data{data_path, c}, c, motis_version);
      } catch (std::exception const& e) {
        std::cerr << "unable to start server: " << e.what() << "\n";
        return_value = 1;
      }
      break;

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

        c = config::read(config_path);
        auto const bars = utl::global_progress_bars{false};
        import(c, std::move(data_path));
        return_value = 0;
      } catch (std::exception const& e) {
        fmt::println("unable to import: {}", e.what());
        fmt::println("config:\n{}", fmt::streamed(c));
        return_value = 1;
      }
      break;
    }

    case cista::hash("pb2json"): {
      try {
        auto p = fs::path{};

        auto desc = po::options_description{"GTFS-RT Protobuf to JSON"};
        desc.add_options()  //
            ("path,p", boost::program_options::value(&p)->default_value(p),
             "Path to Protobuf GTFS-RT file");
        auto vm = parse_opt(ac, av, desc);
        if (vm.count("help")) {
          std::cout << desc << "\n";
          return_value = 0;
          break;
        }

        auto const protobuf = cista::mmap{p.generic_string().c_str(),
                                          cista::mmap::protection::READ};
        fmt::println("{}", nigiri::rt::protobuf_to_json(protobuf.view()));
        return_value = 0;
      } catch (std::exception const& e) {
        fmt::println("error: ", e.what());
        return_value = 1;
      }
      break;
    }

    case cista::hash("json2pb"): {
      try {
        auto p = fs::path{};

        auto desc = po::options_description{"GTFS-RT JSON to Protobuf"};
        desc.add_options()  //
            ("path,p", boost::program_options::value(&p)->default_value(p),
             "Path to GTFS-RT JSON file");
        auto vm = parse_opt(ac, av, desc);
        if (vm.count("help")) {
          std::cout << desc << "\n";
          return_value = 0;
          break;
        }

        auto const protobuf = cista::mmap{p.generic_string().c_str(),
                                          cista::mmap::protection::READ};
        fmt::println("{}", nigiri::rt::json_to_protobuf(protobuf.view()));
        return_value = 0;
      } catch (std::exception const& e) {
        fmt::println("error: ", e.what());
        return_value = 1;
      }
      break;
    }

    case cista::hash("shapes"): {
      try {
        auto data_path = fs::path{"data"};

        auto desc = po::options_description{"Analyze Shapes Options"};
        add_trip_id_opt(desc);
        add_data_path_opt(desc, data_path);
        add_help_opt(desc);

        auto vm = parse_opt(ac, av, desc);
        if (vm.count("help")) {
          std::cout << desc << "\n";
          return_value = 0;
          break;
        }

        if (vm.count("trip-id") == 0) {
          std::cerr << "missing trip-ids\n";
          return_value = 2;
          break;
        }
        auto const c = config::read(data_path / "config.yml");
        auto const ids = utl::to_vec(
            vm["trip-id"].as<std::vector<std::string> >(),
            [](auto const& trip_id) {
              auto const decoded = boost::urls::decode_view{trip_id};
              return std::string{decoded.begin(), decoded.end()};
            });

        return_value = analyze_shapes(data{data_path, c}, ids) ? 0 : 1;
      } catch (std::exception const& e) {
        std::cerr << "unable to analyse shapes: " << e.what() << "\n";
        return_value = 1;
      }
      break;
    }

    default:
      fmt::println(
          "Invalid command. Type motis --help for a list of commands.");
      return_value = 1;
      break;
  }

  google::protobuf::ShutdownProtobufLibrary();
  return return_value;
}
