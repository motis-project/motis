#include <iostream>

#include "boost/filesystem.hpp"

#include "utl/for_each_line_in_file.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/message.h"

#include "motis/paxmon/tools/convert/journey_converter.h"

using namespace motis;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::paxmon;
using namespace motis::paxmon::tools::convert;

namespace fs = boost::filesystem;

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "usage: " << argv[0]
              << " journeys_input.txt journeys_output.csv" << std::endl;
    return 1;
  }

  auto const input_path = argv[1];
  auto const output_path = argv[2];

  if (!fs::is_regular_file(input_path)) {
    std::cerr << "Input file not found: " << input_path << "\n";
    return 1;
  }

  auto converter = journey_converter{output_path};

  auto line_nr = 0ULL;
  auto journey_id = 0ULL;
  utl::for_each_line_in_file(input_path, [&](std::string const& line) {
    ++line_nr;
    try {
      auto const res_msg = make_msg(line);
      switch (res_msg->get()->content_type()) {
        case MsgContent_RoutingResponse: {
          auto const res = motis_content(RoutingResponse, res_msg);
          auto const journeys = message_to_journeys(res);
          for (auto const& j : journeys) {
            converter.write_journey(j, ++journey_id);
          }
          break;
        }
        case MsgContent_Connection: {
          auto const res = motis_content(Connection, res_msg);
          auto const j = convert(res);
          converter.write_journey(j, ++journey_id);
          break;
        }
        default: break;
      }
    } catch (std::system_error const& e) {
      std::cerr << "Invalid message: " << e.what() << ": line " << line_nr
                << "\n";
    }
  });

  std::cout << "Converted " << journey_id << " journeys\n";

  return 0;
}
