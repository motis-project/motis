#include "motis/raptor/eval/commands.h"

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "motis/core/schedule/time.h"
#include "conf/options_parser.h"

#include "motis/bootstrap/dataset_settings.h"
#include "motis/bootstrap/motis_instance.h"

#include "motis/raptor/get_raptor_timetable.h"
#include "motis/raptor/print_raptor.h"
#include "motis/raptor/raptor_timetable.h"
#include "motis/raptor/types.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::bootstrap;
using namespace motis::module;
using namespace motis::routing;

namespace motis::raptor::eval {

struct print_raptor_route_options : public conf::configuration {
  print_raptor_route_options() : configuration{"Print options"} {
    param(in_r_ids_, "r_ids", "Space separated list of Routes to print");
  }

  std::string in_r_ids_{};
};

int print_raptor_route(int argc, const char** argv) {
  print_raptor_route_options print_opt;
  dataset_settings dataset_opt;

  conf::options_parser parser({&dataset_opt, &print_opt});
  parser.read_command_line_args(argc, argv);

  if (parser.help()) {
    std::cout << "\n\tPrint Raptor Routes\n\n";
    parser.print_help(std::cout);
    return 0;
  } else if (parser.version()) {
    std::cout << "Print Raptor Routes\n";
    return 0;
  }

  parser.read_configuration_file();

  std::cout << "\n\tPrint Raptor Routes\n\n";
  parser.print_unrecognized(std::cout);
  parser.print_used(std::cout);

  std::vector<route_id> route_ids{};
  std::stringstream routeStr{print_opt.in_r_ids_};
  std::string val{};
  while(std::getline(routeStr,val, ' ')) {
    route_id r_id = std::stoi(val);
    route_ids.push_back(r_id);
  }

  motis_instance instance;
  instance.import(module_settings{}, dataset_opt, import_settings{{dataset_opt.dataset_}});

    auto const& sched = instance.sched();
    auto const [_, tt] = get_raptor_timetable(sched);

    for (auto const& r_id : route_ids) {
      std::cout << "Printing Route " << +r_id << "\n";
      print_route(r_id, *tt);
      std::cout << "\n\n";
    }

    return 0;
}

}  // namespace motis::raptor::eval