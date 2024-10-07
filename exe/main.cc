#include "boost/program_options.hpp"

namespace bpo = boost::program_options;
using namespace std::string_view_literals;

namespace motis {
int import(int, char**);
int server(int, char**);
}  // namespace motis

using namespace motis;

int main(int ac, char** av) {
  auto const subcommand = std::string_view{ac <= 1 ? "server" : av[1]};

  if (ac > 1) {
    --ac;
    ++av;
  }

  if (subcommand == "server") {
    return server(ac, av);
  } else if (subcommand == "import") {
    return import(ac, av);
  }
}