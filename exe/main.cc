#include "boost/program_options.hpp"

#include <iostream>

#include "utl/progress_tracker.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/import.h"

#if !defined(MOTIS_VERSION)
#define MOTIS_VERSION "unknown"
#endif

namespace po = boost::program_options;
using namespace std::string_view_literals;

namespace motis {
int import(int, char**);
int server(int, char**);
int server(data d, config const& c);
}  // namespace motis

using namespace motis;

int main(int ac, char** av) {
  auto global = po::options_description{"Global options"};
  global.add_options()  //
      ("help", "print help message")  //
      ("version", "print version")  //
      ("command", po::value<std::string>(), "command to execute")  //
      ("subargs", po::value<std::vector<std::string>>(),
       "arguments for command");

  auto pos = po::positional_options_description{};
  pos.add("command", 1).add("subargs", -1);

  auto parsed = po::command_line_parser(ac, av)
                    .options(global)
                    .positional(pos)
                    .allow_unregistered()
                    .run();

  auto vm = po::variables_map{};
  po::store(parsed, vm);

  if (!vm.count("command")) {
    if (vm.count("version")) {
      std::cout << MOTIS_VERSION << "\n";
      return 0;
    } else if (vm.count("help")) {
      std::cout << "MOTIS " << MOTIS_VERSION << "\n\n" << global << "\n";
      return 0;
    }
  }

  --ac;
  ++av;

  if (vm.count("command")) {
    auto const cmd = vm["command"].as<std::string>();
    if (cmd == "server") {
      return server(ac, av);
    } else if (cmd == "import") {
      return import(ac, av);
    } else {
      try {
        auto const bars = utl::global_progress_bars{false};
        auto args = vm.count("subargs")
                        ? vm.at("subargs").as<std::vector<std::string>>()
                        : std::vector<std::string>{};
        args.insert(begin(args), cmd);
        auto const c = config::read_simple(args);
        server(import(c, "data"), c);
      } catch (std::exception const& e) {
        std::cerr << "error: " << e.what() << "\n";
      }
    }
  }
}