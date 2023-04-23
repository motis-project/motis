#include "motis/routing/eval/commands.h"

#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>

#include "boost/program_options.hpp"

#include "utl/for_each_line_in_file.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/core/journey/print_journey.h"
#include "motis/module/message.h"
#include "motis/routing/eval/is_terminal.h"

using namespace motis;
using namespace motis::module;
using namespace motis::routing;
using namespace flatbuffers;

namespace fs = std::filesystem;
namespace po = boost::program_options;

namespace motis::routing::eval {

int print(int argc, char const** argv) {
  bool help = false;
  bool utc = false;
  bool local = false;
  realtime_format rt_format = realtime_format::TIME;
  std::vector<std::string> files;
  po::options_description desc("Print Journey");
  // clang-format off
  desc.add_options()
    ("help,h", po::bool_switch(&help), "show help")
    ("utc,u", po::bool_switch(&utc), "print timestamps in UTC")
    ("local,l", po::bool_switch(&local), "print timestamps in local time")
    ("i", po::value<std::vector<std::string>>(&files), "input file")
    ("rt,r", po::value<realtime_format>(&rt_format)->default_value(rt_format),
        "format for realtime information (none/offset/time)")
  ;
  // clang-format on
  po::positional_options_description pod;
  pod.add("i", -1);
  po::variables_map vm;
  po::store(
      po::command_line_parser(argc, argv).options(desc).positional(pod).run(),
      vm);
  po::notify(vm);

  if (!help && files.empty() && !is_terminal(stdin)) {
    files.emplace_back("-");
  }

  if (help || files.empty()) {
    std::cout << desc << std::endl;
    return 0;
  }

  if (utc) {
    local = false;
  }

  auto const multi_file = files.size() > 1;

  auto first = true;
  for (auto const& f : files) {
    if (multi_file) {
      if (!first) {
        std::cout << "\n\n\n\n";
      } else {
        first = false;
      }
      auto const bar = std::string(f.size(), '=');
      std::cout << bar << "\n" << f << "\n" << bar << "\n\n";
    }
    if (f != "-" && !fs::exists(f)) {
      std::cout << "File does not exist" << std::endl;
      continue;
    }

    utl::for_each_line_in_file(f, [&](std::string const& line) {
      auto const msg = make_msg(line);
      auto const res = motis_content(RoutingResponse, msg);
      auto const journeys = message_to_journeys(res);

      std::cout << "Response " << msg->id() << " contains " << journeys.size()
                << (journeys.size() == 1 ? " journey" : " journeys")
                << std::endl;

      for (auto const& j : journeys) {
        std::cout << "\n\n";
        print_journey(j, std::cout, local, rt_format);
      }
      std::cout << "\n\n";
    });
  }

  return 0;
}

}  // namespace motis::routing::eval