
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
using namespace motis::module;
using namespace motis::isochrone;
using namespace motis::routing;

int main(int argc, char const** argv) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0]
              << " {iso_responses.txt} {routing_responses.txt}\n";
    return 0;
  }
  std::ifstream in_i(argv[1]), in_r(argv[2]);
  std::ofstream failed_routes("failed_routes.txt");
  std::string line_i, line_r;
  std::map<int, uint64_t> times;
  int mismatches = 0;
  while (in_i.peek() != EOF && !in_i.eof()){

    std::getline(in_i, line_i);
    auto const i_msg = make_msg(line_i);
    auto const i_res = motis_content(IsochroneResponse, i_msg);
    for(int i = 0; i<i_res->arrival_times()->size(); ++i) {
      times[i_msg->id()*1000000+i] = i_res->arrival_times()->Get(i);
    }


  }
  while (in_r.peek() != EOF && !in_r.eof()) {

    std::getline(in_r, line_r);
    auto const r_msg = make_msg(line_r);
    auto const r_res = motis_content(RoutingResponse, r_msg);
    uint64_t at = 0;
    for(auto it : *r_res->connections()) {
      auto at_new = it->stops()->Get(it->stops()->size()-1)->arrival()->time();
      if(at_new < at || at == 0) {
        at = at_new;
      }
    }
    auto const remaining_time = 3600 - (at - r_res->interval_begin());
    if (times.at(r_msg->id()) != remaining_time) {
      auto x = times.at(r_msg->id());
      mismatches++;
      failed_routes << "id:" << r_msg->id() << " iso time:"<< x << " routing time:" << remaining_time << std::endl;
    }
  }

  failed_routes.flush();

  std::cout << mismatches << std::endl;
  return 0;
};

