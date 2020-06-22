#include <iostream>

#include "conf/options_parser.h"

#include "utl/progress_tracker.h"

#include "motis/core/common/logging.h"
#include "motis/path/prepare/prepare.h"

#include "version.h"

namespace m = motis;
namespace mp = motis::path;
namespace ml = motis::logging;

int main(int argc, char const** argv) {
  try {
    mp::prepare_settings opt;

    try {
      conf::options_parser parser({&opt});
      parser.read_command_line_args(argc, argv, false);

      if (parser.help()) {
        std::cout << "\n\tpath-prepare (v" << m::short_version() << ")\n\n";
        parser.print_help(std::cout);
        return 0;
      } else if (parser.version()) {
        std::cout << "path-prepare (v" << m::long_version() << ")\n";
        return 0;
      }

      parser.read_configuration_file(false);
      parser.print_used(std::cout);
    } catch (std::exception const& e) {
      LOG(ml::emrg) << "options error: " << e.what();
      return 1;
    }

    utl::activate_progress_tracker("path");
    mp::prepare(opt);
    return 0;

  } catch (std::exception const& e) {
    LOG(ml::emrg) << "exception caught: " << e.what();
    return 1;
  } catch (...) {
    LOG(ml::emrg) << "unknown exception caught";
    return 1;
  }
}
