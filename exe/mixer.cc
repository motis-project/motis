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
#include "motis/odm/mixer.h"

#include "./flags.h"
#include "motis/odm/journeys.h"
#include "utl/read_file.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;
namespace json = boost::json;

namespace motis {

int mixer(int ac, char** av) {
  auto cfg_path = std::string{"mixer.json"};
  auto in_path = std::string{"journeys.csv"};
  auto out_path = std::string{"."};

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
  auto const m = [&]() {
    auto cfg_file = std::ifstream{cfg_path};
    return json::value_to<odm::mixer>(json::parse(cfg_file));
  }();

  auto const in_file = utl::read_file(in_path.c_str());
  if (!in_file) {
    fmt::println("Failed to read input file");
    return 1;
  }
  auto odm_journeys = odm::from_csv(*in_file);
  auto const pt_journeys = odm::separate_pt(odm_journeys);

  auto ride_share_journeys = std::vector<nigiri::routing::journey>{};
  m.mix(pt_journeys, odm_journeys, ride_share_journeys, nullptr,
        std::string_view{out_path});

  auto out_file = std::ofstream{fs::path{out_path} / "journeys.csv"};
  out_file << odm::to_csv(odm_journeys);

  return 0U;
}

}  // namespace motis