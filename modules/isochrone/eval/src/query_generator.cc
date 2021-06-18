//TODO: Generate queries for isochrone queries

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

#include "boost/algorithm/string.hpp"
#include "boost/date_time/gregorian/gregorian_types.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/program_options.hpp"

#include "utl/erase.h"
#include "utl/to_vec.h"

#include "conf/options_parser.h"

#include "motis/core/schedule/time.h"
#include "motis/core/access/time_access.h"
#include "motis/module/message.h"
#include "motis/bootstrap/dataset_settings.h"
#include "motis/bootstrap/motis_instance.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::bootstrap;
using namespace motis::module;
using namespace motis::isochrone;

struct generator_settings : public conf::configuration {
  generator_settings() : configuration("Generator Settings") {
    param(query_count_, "query_count", "number of queries to generate");
    param(target_file_, "target_file",
          "file to write generated departure time queries to. ${target} is "
          "replaced by the target url");
  }

  int query_count_{10000};
  std::string target_file_{"queries-isochrone.txt"};
};

std::string query(int id,
                  std::time_t interval_start, std::time_t interval_end,
                  std::string const& from_eva) {
  message_creator fbb;
  fbb.create_and_finish(
          MsgContent_IsochroneRequest,
          CreateIsochroneRequest(
                  fbb,
                  CreateInputStation(fbb, fbb.CreateString(from_eva),
                                     fbb.CreateString("")),
                  interval_start,interval_end)
                  .Union(),
          "/isochrone");
  auto msg = make_msg(fbb);
  msg->get()->mutate_id(id);

  auto json = msg->to_json();
  utl::erase(json, '\n');
  return json;
}

int main(int argc, char const** argv) {
  generator_settings generator_opt;
  dataset_settings dataset_opt;
  dataset_opt.adjust_footpaths_ = true;

  conf::options_parser parser({&dataset_opt, &generator_opt});
  parser.read_command_line_args(argc, argv);

  parser.read_configuration_file();

  std::cout << "\n\tQuery Generator\n\n";
  parser.print_unrecognized(std::cout);
  parser.print_used(std::cout);

  motis_instance instance;
  instance.import(module_settings{}, dataset_opt,
                  import_settings({dataset_opt.dataset_}));


  return 0;
}
