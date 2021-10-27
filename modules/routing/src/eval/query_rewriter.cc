#include "motis/routing/eval/commands.h"

#include <fstream>
#include <iostream>

#include "boost/algorithm/string.hpp"

#include "motis/module/message.h"

#include "conf/configuration.h"
#include "conf/options_parser.h"

#include "version.h"

using namespace motis;
using motis::module::make_msg;

namespace motis::routing::eval {

struct rewrite_options : public conf::configuration {
  rewrite_options() : configuration{"Rewrite options"} {
    param(in_path_, "in", "Input file path");
    param(out_path_, "out", "Output file path");
    param(new_target_, "target", "New target");
  }
  std::string in_path_, out_path_, new_target_;
};

int rewrite_queries(int argc, char const** argv) {
  rewrite_options opt;

  try {
    conf::options_parser parser({&opt});
    parser.read_command_line_args(argc, argv, false);

    if (parser.help()) {
      std::cout << "\n\tMOTIS v" << short_version() << "\n\n";
      parser.print_help(std::cout);
      return 0;
    } else if (parser.version()) {
      std::cout << "MOTIS v" << long_version() << "\n";
      return 0;
    }

    parser.read_configuration_file(true);
    parser.print_used(std::cout);
  } catch (std::exception const& e) {
    std::cout << "options error: " << e.what() << "\n";
    return 1;
  }

  std::ifstream in{opt.in_path_.c_str()};
  std::ofstream out{opt.out_path_.c_str()};

  in.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  out.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  auto count = 0U;
  std::string json;
  while (!in.eof() && in.peek() != EOF) {
    std::getline(in, json);
    auto const target = make_msg(json)->get()->destination()->target()->str();

    boost::replace_all(json, target, opt.new_target_);
    out << json << "\n";

    ++count;
  }

  std::cout << "rewrote " << count << " queries\n";
  return 0;
}

}  // namespace motis::routing::eval