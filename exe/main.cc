#include "boost/program_options.hpp"

#include <iostream>

#if !defined(MOTIS_VERSION)
#define MOTIS_VERSION "unknown"
#endif

namespace po = boost::program_options;
using namespace std::string_view_literals;

namespace motis {
int import(int, char**);
int server(int, char**);
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
    }
  } else {
    auto const exit_code = import(ac, av);
    if (exit_code == 0) {
      server(ac, av);
    }
  }
}