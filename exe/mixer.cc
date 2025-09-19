#include <fstream>
#include <iostream>

#include "conf/configuration.h"

#include "boost/json/serialize.hpp"
#include "boost/json/value_from.hpp"

#include "utl/init_from.h"
#include "utl/parallel_for.h"
#include "utl/parser/cstr.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/routing.h"

#include "./flags.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace json = boost::json;

namespace motis {

int mixer(int ac, char** av) {
  auto cfg_path = fs::path{"mixer.json"};
  auto in_path = fs::path{"journeys.csv"};
  auto out_path = fs::path{"."};

  auto desc = po::options_description{"Options"};
  desc.add_options()  //
      ("help", "Prints this help message")  //
      ("cfg,c", po::value(&cfg_path)->default_value(cfg_path))  //
      ("in,i", po::value(&in_path)->default_value(in_path),
       "path to input journeys csv file")  //
      ("out,o", po::value(&out_path)->default_value(out_path),
       "folder to write output csv files to");

  auto vm = parse_opt(ac, av, desc);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }

  return 0U;
}

}  // namespace motis