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
    auto const msg_in = make_msg(json);

    motis::module::message_creator fbb;

    flatbuffers::Offset<void> content;
    switch (msg_in->get()->content_type()) {
      case MsgContent_RoutingResponse: {
        content =
            motis_copy_table(RoutingResponse, fbb, msg_in->get()->content())
                .Union();
      } break;
      case MsgContent_RoutingRequest: {
        content =
            motis_copy_table(RoutingRequest, fbb, msg_in->get()->content())
                .Union();
      } break;
      case MsgContent_IntermodalRoutingRequest: {
        using motis::intermodal::IntermodalRoutingRequest;
        content = motis_copy_table(IntermodalRoutingRequest, fbb,
                                   msg_in->get()->content())
                      .Union();
      } break;
      default: std::cerr << "unsupported message content type\n"; return 1;
    }
    fbb.create_and_finish(msg_in->get()->content_type(), content.Union(),
                          opt.new_target_, DestinationType_Module,
                          msg_in->id());
    out << make_msg(fbb)->to_json(true) << "\n";

    ++count;
  }

  std::cout << "rewrote " << count << " queries\n";
  return 0;
}

}  // namespace motis::routing::eval
